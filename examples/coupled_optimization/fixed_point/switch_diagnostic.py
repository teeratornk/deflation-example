"""Diagnose or re-solve one immutable continuation step near a coefficient switch.

The residual, control, previous fields and final accuracy remain unchanged.
The optional active branch is a forward linearization choice; it supplies no
classical derivative at a kink and no global convergence guarantee.
"""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from step_diagnostics import checked_field
from deflation_example.axisymmetric_flow import FlowResult
from deflation_example.coupled_fixed_point import solve_forward
from deflation_example.coupled_newton_replay import newton_step, step_equations, criteria_met
from deflation_example.coupled_resolution import forward_model
from deflation_example.coupled_saved import file_digest
from deflation_example.coupled_step_spectrum import step_linearization
from deflation_example.fixed_point_study import prepare
from deflation_example.reporting import environment, write_arrays, write_report


def load_continuation(args):
    original = json.loads((args.trajectory / "record.json").read_text())
    continued = json.loads((args.continuation / "record.json").read_text())
    repair = json.loads((args.repair / "record.json").read_text())
    if continued["status"] not in {"converged", "numerical_failure"}:
        raise ValueError("Inspect a terminal continuation with saved fields")
    if original["time_scheme"] != "backward_euler":
        raise ValueError("Only backward Euler is supported")
    original_hash = file_digest(args.trajectory / "record.json")
    if (
        continued["trajectory_record_sha256"] != original_hash
        or continued["repair_record_sha256"] != file_digest(args.repair / "record.json")
        or repair["trajectory_record_sha256"] != original_hash
        or not repair["verified"]
        or repair["status"] != "complete"
        or repair["slab_zero_based"] != continued["corrected_slab"]
        or repair["initial_field_sha256"]
        != original["steps"][continued["corrected_slab"]]["field_sha256"]
    ):
        raise ValueError("Continuation input identity differs")
    setup = SimpleNamespace(
        **vars(args),
        policy=original["policy"],
        family="forward",
        repetition=original["repetition"],
        target_position=None,
    )
    problem, controls, metadata = prepare(setup, original["subdivision"])
    for key in (
        "optimization_field_sha256",
        "baseline_sha256",
        "fine_baseline_sha256",
        "slabs",
        "spatial_state_dofs",
        "configuration",
    ):
        if original[key] != metadata[key]:
            raise ValueError("Original input identity differs: " + key)
    if (
        continued["configuration"] != metadata["configuration"]
        or continued["slabs"] != problem.slabs
    ):
        raise ValueError("Continuation problem differs")
    n = len(continued["steps"]) - 1 if args.slab is None else args.slab
    if not 0 <= n < len(continued["steps"]) or n >= problem.slabs:
        raise ValueError("Select a saved continuation step")
    for i, row in enumerate(continued["steps"][: n + 1]):
        if row["slab_zero_based"] != i or not np.isclose(
            row["time_s"], problem.physical_steps[: i + 1].sum(), rtol=0, atol=1e-9
        ):
            raise ValueError("Continuation time ordering differs")
        if i < n and not row["verified"]:
            raise ValueError("The preceding continuation must be verified")
    state, flow = checked_field(args.continuation, continued["steps"][n], problem)
    previous, previous_flow = (
        checked_field(args.continuation, continued["steps"][n - 1], problem)
        if n
        else (problem.full_temperature(problem.initial), problem.initial_flow)
    )
    # Independently check the immediately preceding equation as well as its hash.
    if n:
        older, older_flow = (
            checked_field(args.continuation, continued["steps"][n - 2], problem)
            if n > 1
            else (problem.full_temperature(problem.initial), problem.initial_flow)
        )
        source_previous = np.zeros(len(problem.mesh.nodes))
        source_previous[problem.free] = controls[n - 1]
        _, checks = step_equations(
            problem,
            forward_model(problem),
            previous,
            previous_flow,
            source_previous,
            older,
            older_flow,
            n - 1,
        )
        if not criteria_met(checks, 1e-12):
            raise ValueError("Previous fields fail the independent original equations")
    source = np.zeros(len(problem.mesh.nodes))
    source[problem.free] = controls[n]
    return problem, source, previous, previous_flow, state, flow, n, continued


def branch_values(problem, velocity):
    flow = problem.flow
    vertices = problem.mesh.nodes[problem.mesh.cells[flow.fluid_cells]]
    h = np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=3).max(axis=(1, 2))
    shape = np.array([-1, -1, -1, 4, 4, 4]) / 9
    center = problem.velocity_scale * np.einsum("a,ead->ed", shape, velocity[flow.p2])
    speed = np.linalg.norm(center, axis=1)
    capacity = problem.capacity[flow.fluid_cells]
    k = np.linalg.eigvalsh(problem.conductivity[flow.fluid_cells])[:, 0]
    diffusion = h**2 / (12 * k)
    advection = np.full_like(speed, np.inf)
    np.divide(h, 2 * capacity * speed, out=advection, where=speed > 0)
    # This bounded signed distance is -1 at zero velocity and zero at a tie.
    distance = (h * capacity * speed - 6 * k) / (h * capacity * speed + 6 * k)
    return {
        "distance": distance,
        "advective": advection < diffusion,
        "strict_switch": np.isclose(advection, diffusion, rtol=1e-12, atol=0),
        "center": vertices.mean(axis=1),
    }


def inspect(problem, source, previous, previous_flow, state, flow, slab):
    model = forward_model(problem)
    residual, checks = step_equations(
        problem, model, state, flow, source, previous, previous_flow, slab
    )
    branches = branch_values(problem, flow.velocity)
    cell = int(np.argmin(np.abs(branches["distance"])))
    report = {
        "initial_checks": checks,
        "strict_switch_cells": int(branches["strict_switch"].sum()),
        "closest_fluid_cell": int(problem.flow.fluid_cells[cell]),
        "closest_cell_centroid_m": branches["center"][cell].tolist(),
        "minimum_absolute_branch_distance": float(abs(branches["distance"][cell])),
        "distance_definition": "(h c |v_scaled| - 6 k_min) / (h c |v_scaled| + 6 k_min)",
    }
    H = step_linearization(
        problem, state[problem.free], flow.velocity, slab, branch_policy="active"
    )[0]
    scale = 1 / np.maximum(abs(H).max(axis=1).toarray().ravel(), np.finfo(float).tiny)
    factor = splu((sparse.diags(scale) @ H).tocsc())
    update = factor.solve(-scale * residual)
    denominator = max(np.linalg.norm(scale * residual), np.finfo(float).tiny)
    for _ in range(2):
        defect = residual + H @ update
        if np.linalg.norm(scale * defect) / denominator <= 1e-10:
            break
        update += factor.solve(-scale * defect)
    report["linear_relative_residual"] = float(
        np.linalg.norm(scale * (residual + H @ update)) / denominator
    )
    direction = np.zeros(problem.flow.size)
    direction[problem.flow_free] = update[: len(problem.flow_free)]
    dv = np.column_stack(
        (direction[: problem.flow.nv], direction[problem.flow.nv : 2 * problem.flow.nv])
    )
    report["trials"] = []
    for alpha in (1.0, 0.5, 0.125, 0.01, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, -1e-6):
        trial = state.copy()
        trial[problem.free] += alpha * update[len(problem.flow_free) :]
        trial_flow = FlowResult(
            flow.velocity + alpha * dv,
            flow.pressure + alpha * direction[2 * problem.flow.nv :],
            "diagnostic",
            [],
        )
        candidate, metrics = step_equations(
            problem, model, trial, trial_flow, source, previous, previous_flow, slab
        )
        new_branches = branch_values(problem, trial_flow.velocity)
        finite_difference = (candidate - residual) / alpha
        report["trials"].append(
            {
                "step": alpha,
                "fixed_scaled_residual_ratio": float(
                    np.linalg.norm(scale * candidate) / denominator
                ),
                "linearization_relative_difference": float(
                    np.linalg.norm(scale * (finite_difference - H @ update)) / denominator
                ),
                "changed_branch_cells": int(
                    np.count_nonzero(new_branches["advective"] != branches["advective"])
                ),
                "checks": metrics,
            }
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline",
        "fine-baseline",
        "optimization",
        "trajectory",
        "repair",
        "continuation",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--slab", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--policy",
        dest="diagnostic_policy",
        choices=("inspect", "active_newton", "coupled_anderson"),
        default="inspect",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    policy = args.diagnostic_policy
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {
        "status": "running",
        "policy": policy,
        "environment": environment(),
        "driver_sha256": file_digest(Path(__file__)),
        "loader_sha256": file_digest(Path(__file__).with_name("step_diagnostics.py")),
        "accuracy": {"equation": 1e-12, "mass": 1e-6, "energy": 1e-6},
        "scope": "One unchanged saved step; not a complete trajectory, optimizer timing or physical resolution study.",
    }
    write_report(args.output / "record.json", record)
    try:
        with threadpool_limits(args.threads):
            problem, source, previous, previous_flow, state, flow, n, saved = load_continuation(
                args
            )
            record.update(
                continuation_record_sha256=file_digest(args.continuation / "record.json"),
                trajectory_record_sha256=file_digest(args.trajectory / "record.json"),
                repair_record_sha256=file_digest(args.repair / "record.json"),
                initial_field_sha256=saved["steps"][n]["field_sha256"],
                previous_field_sha256=saved["steps"][n - 1]["field_sha256"] if n else None,
                slab_zero_based=n,
                time_s=saved["steps"][n]["time_s"],
            )
            write_report(args.output / "record.json", record)
            originals = [
                x.copy() for x in (source, previous, previous_flow.velocity, previous_flow.pressure)
            ]
            if policy == "inspect":
                record["diagnosis"] = inspect(
                    problem, source, previous, previous_flow, state, flow, n
                )
            else:
                if policy == "active_newton":
                    result = newton_step(
                        problem,
                        source,
                        previous,
                        previous_flow,
                        n,
                        initial_state=state,
                        initial_flow=flow,
                        tolerance=1e-12,
                        max_iterations=100,
                        line_search="fixed_scaled",
                        backtrack_cap=40,
                        branch_policy="active",
                    )
                else:
                    result = solve_forward(
                        forward_model(problem),
                        source,
                        state,
                        flow,
                        previous_state=previous,
                        previous_velocity=previous_flow.velocity,
                        time_step=float(problem.physical_steps[n]),
                        policy="anderson",
                        depth=5,
                        relaxation=0.5,
                        mixing="coupled",
                        tolerance=1e-12,
                        max_iterations=300,
                        flow_cap=100,
                        callback=lambda row: write_report(args.output / "progress.json", row),
                    )
                _, checks = step_equations(
                    problem,
                    forward_model(problem),
                    result.state,
                    result.flow,
                    source,
                    previous,
                    previous_flow,
                    n,
                )
                write_arrays(
                    args.output / "fields.npz",
                    state=result.state[problem.free],
                    velocity=result.flow.velocity,
                    pressure=result.flow.pressure,
                )
                record.update(
                    termination=result.status,
                    verified=bool(result.status == "converged" and criteria_met(checks, 1e-12)),
                    checks=checks,
                    history=result.history,
                    solve_seconds=result.seconds,
                    field_sha256=file_digest(args.output / "fields.npz"),
                )
            for before, after in zip(
                originals,
                (source, previous, previous_flow.velocity, previous_flow.pressure),
                strict=True,
            ):
                if not np.array_equal(before, after):
                    raise RuntimeError("Diagnostic changed the fixed source or preceding fields")
            record["status"] = "complete"
    except Exception as error:
        record.update(status="execution_failure", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["process_seconds"] = time.perf_counter() - started
        write_report(args.output / "record.json", record)


if __name__ == "__main__":
    main()
