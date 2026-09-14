"""Check saved coupled equations and isolate propagation in a same-grid replay.

Every check uses the unchanged saved heat source. Local thermal solves hold the
saved velocity fixed; they diagnose the discrepancy without establishing the
uniqueness or accuracy of the full coupled trajectory.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_optimize import load_problem
from .coupled_resolution import forward_model
from .coupled_saved import file_digest, load_saved_solution, require_matching_baseline
from .reporting import environment, write_report
from .validation import integer, real_array


def check_saved_trajectory(problem, state, control, velocity, pressure, replay=None, callback=None):
    """Independently evaluate every saved time slab, with optional replay errors."""
    if not len(problem.physical_steps):
        raise ValueError("Saved-trajectory checks require a transient problem")
    shapes = {
        "state": (problem.slabs, problem.spatial_size),
        "control": (problem.slabs, problem.spatial_size),
        "velocity": (problem.slabs, problem.flow.nv, 2),
        "pressure": (problem.slabs, problem.flow.np),
    }
    arrays = {}
    for name, value in zip(shapes, (state, control, velocity, pressure), strict=True):
        array = real_array(value, "Saved " + name).copy()
        if name in {"state", "control"} and array.shape == (problem.size,):
            array = array.reshape(shapes[name])
        if array.shape != shapes[name] or not np.isfinite(array).all():
            raise ValueError(f"Saved {name} must be finite with shape {shapes[name]}")
        arrays[name] = array
    state, control, velocity, pressure = arrays.values()
    if replay is not None:
        replay = real_array(replay, "Replay state").copy()
        if replay.shape != state.shape or not np.isfinite(replay).all():
            raise ValueError("Replay must contain every same-grid time slab")
    model = forward_model(problem)
    rows, start = [], time.perf_counter()
    I, J = problem.free, problem.mesh.dirichlet
    for n in range(problem.slabs):
        previous = problem.full_temperature(problem.initial if n == 0 else state[n - 1])
        full = problem.full_temperature(state[n])
        source = np.zeros_like(full)
        source[I] = control[n]
        dt = float(problem.physical_steps[n])
        assembly, K, rhs = model._thermal_system(velocity[n], source, previous, dt)
        boundary = K[I][:, J] @ full[J]
        scale = max(np.linalg.norm(rhs[I]) + np.linalg.norm(boundary), 1e-30)
        B = K[I][:, I].tocsc()
        solved = spsolve(B, rhs[I] - boundary)
        flow = FlowResult(velocity[n], pressure[n], "saved", [])
        previous_velocity = problem.initial_flow.velocity if n == 0 else velocity[n - 1]
        metrics = problem.flow.verify(
            flow,
            model._force(full, True),
            problem.boundary_indices,
            problem.boundary_values,
            previous=previous_velocity,
            time_step=dt,
            pressure_gauge=problem.pressure_gauge,
        )
        balance = model.energy_balance(assembly, full, velocity[n], source, previous, dt)
        flux = problem.flow.boundary_flux(velocity[n])
        row = {
            "slab_zero_based": n,
            "time_s": float(problem.physical_steps[: n + 1].sum()),
            "thermal_relative_residual": float(np.linalg.norm((K @ full - rhs)[I]) / scale),
            **{key: float(value) for key, value in metrics.items()},
            "mass_relative_imbalance": float(abs(flux.sum()) / max(np.abs(flux).sum() / 2, 1e-30)),
            "energy_relative_defect": float(balance["relative_defect"]),
            "exact_previous_frozen_velocity_error_K": float(
                np.max(np.abs(solved - state[n])) * problem.temperature_scale
            ),
        }
        if replay is not None:
            replay_previous = problem.initial if n == 0 else replay[n - 1]
            capacity = assembly.capacity[I] / problem.steps[n]
            propagated = spsolve(B, rhs[I] - boundary + capacity * (replay_previous - previous[I]))
            row.update(
                replayed_previous_frozen_velocity_error_K=float(
                    np.max(np.abs(propagated - state[n])) * problem.temperature_scale
                ),
                observed_coupled_replay_error_K=float(
                    np.max(np.abs(replay[n] - state[n])) * problem.temperature_scale
                ),
                remaining_replay_difference_K=float(
                    np.max(np.abs(replay[n] - propagated)) * problem.temperature_scale
                ),
            )
        if not np.isfinite(list(row.values())).all():
            raise ValueError("Nonfinite saved-trajectory diagnostic")
        rows.append(row)
        if callback is not None:
            callback(row)
    return {"rows": rows, "seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("reference", "jacobi", "recycling"), default="reference"
    )
    parser.add_argument("--target-position", type=int)
    parser.add_argument(
        "--replay", type=Path, help="Same-grid replay directory, including states.npz"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        record, cfg, fields, digest = load_saved_solution(
            args.optimization, args.method, args.target_position
        )
        problem, baseline = load_problem({**cfg, "baseline_directory": str(args.baseline)})
        require_matching_baseline(record, baseline)
        filename = (
            (args.method + "-fields.npz")
            if args.target_position is None
            else f"target-{args.target_position:02d}.npz"
        )
        field_path = args.optimization / filename
        with np.load(field_path, allow_pickle=False) as data:
            velocity, pressure = data["velocity"].copy(), data["pressure"].copy()
        if file_digest(field_path) != digest:
            raise ValueError("Saved fields changed while loading")
        replay = None
        replay_digest = None
        if args.replay is not None:
            replay_record = json.loads((args.replay / "record.json").read_text())
            if (
                replay_record.get("optimization_field_sha256") != digest
                or replay_record.get("subdivision") != 1
            ):
                raise ValueError("Replay must use the same saved fields and original time grid")
            replay_path = args.replay / "states.npz"
            with np.load(replay_path, allow_pickle=False) as data:
                replay = data["state"].copy()
                times = data["times_s"].copy()
            expected = np.cumsum(problem.physical_steps)
            if times.shape != expected.shape or not np.allclose(
                times, expected, rtol=1e-12, atol=1e-12
            ):
                raise ValueError("Replay physical time levels differ from the saved trajectory")
            replay_digest = file_digest(replay_path)
        args.output.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema": "coupled-saved-equation-check-v1",
            "environment": environment(),
            "optimization_source": record["environment"]["git_head"],
            "optimization_field_sha256": digest,
            "replay_field_sha256": replay_digest,
            "target_position": args.target_position,
            "scope": "Saved equations and fixed-velocity propagation; no reoptimization or coupled-trajectory accuracy certificate.",
        }
        write_report(args.output / "record.json", {**metadata, "status": "running"})
        result = check_saved_trajectory(
            problem,
            fields["state"],
            fields["control"],
            velocity,
            pressure,
            replay,
            callback=lambda row: write_report(args.output / "progress.json", row),
        )
        write_report(args.output / "record.json", {**metadata, **result, "status": "complete"})


if __name__ == "__main__":
    main()
