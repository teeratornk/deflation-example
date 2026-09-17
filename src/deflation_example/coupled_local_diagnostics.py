"""Fixed-source Newton diagnostics and common-initial-state time comparisons.

These local windows diagnose the declared thermal formulation. They are not a
substitute for a complete-trajectory or spatial resolution test.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_forward_checkpoint import save_snapshot
from .coupled_newton_replay import newton_step, step_equations
from .coupled_optimize import load_problem
from .coupled_resolution import forward_model
from .coupled_saved import load_saved_solution, require_matching_baseline
from .coupled_step_spectrum import step_linearization
from .meshes import simplex_geometry
from .reporting import environment, file_sha256, write_fields, write_report
from .validation import integer


POLICIES = {
    "original": {"backtrack_cap": 21, "trust_region": None},
    "deep": {"backtrack_cap": 40, "trust_region": None},
    "limited": {"backtrack_cap": 21, "trust_region": 0.05},
}


def displaced(problem, state, flow, direction, step):
    """Perturb only the free unknowns in the Jacobian's declared order."""
    packed = np.zeros(problem.flow.size)
    packed[problem.flow_free] = step * direction[: len(problem.flow_free)]
    candidate = state.copy()
    candidate[problem.free] += step * direction[len(problem.flow_free) :]
    nv = problem.flow.nv
    return candidate, FlowResult(
        flow.velocity + np.column_stack((packed[:nv], packed[nv : 2 * nv])),
        flow.pressure + packed[2 * nv :],
        "diagnostic",
        [],
    )


def stabilization_branches(problem, velocity):
    """Identify the diffusive, advective, or row-limit branch in each fluid cell."""
    fluid = problem.flow.fluid_cells
    grad, lump = simplex_geometry(problem.mesh)
    grad, lump = grad[fluid], lump[fluid]
    vertices = problem.mesh.nodes[problem.mesh.cells[fluid]]
    h = np.max(np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=3), axis=(1, 2))
    center = np.array([-1, -1, -1, 4, 4, 4]) / 9
    v = problem.velocity_scale * np.einsum("a,ead->ed", center, velocity[problem.flow.p2])
    c = problem.capacity[fluid]
    k = np.linalg.eigvalsh(problem.conductivity[fluid])[:, 0]
    speed = np.linalg.norm(v, axis=1)
    advective = np.full_like(speed, np.inf)
    np.divide(h, 2 * c * speed, out=advective, where=speed > 0)
    reach = np.linalg.norm(np.einsum("eid,ed->ei", grad, v), axis=1)
    row = np.full_like(speed, np.inf)
    if problem.consistent_stabilization:
        share = (lump / lump.sum(axis=1)[:, None]).min(axis=1)
        np.divide(share, c * reach, out=row, where=reach > 0)
    choices = np.column_stack((h * h / (12 * k), advective, row))
    return np.argmin(choices, axis=1)


def derivative_diagnostic(problem, source, previous, previous_flow, slab, state, flow):
    """Compare exact Jacobian products with centered differences of original equations."""
    model = forward_model(problem)
    residual, metrics = step_equations(
        problem, model, state, flow, source, previous, previous_flow, slab
    )
    H = step_linearization(
        problem, state[problem.free], flow.velocity, slab, control=source, previous=previous
    )[0]
    scaling = 1 / np.maximum(abs(H).max(axis=1).toarray().ravel(), np.finfo(float).tiny)
    update = splu((sparse.diags(scaling) @ H).tocsc()).solve(-scaling * residual)
    denominator = max(np.linalg.norm(scaling * residual), np.finfo(float).tiny)
    original_branches = stabilization_branches(problem, flow.velocity)
    rng = np.random.default_rng(4817)
    random = rng.normal(size=H.shape[1])
    nf = len(problem.flow_free)
    velocity_mask = problem.flow_free < 2 * problem.flow.nv
    speed = max(float(np.max(np.abs(flow.velocity))), 1e-6)
    random[:nf][velocity_mask] *= speed
    # Pressure uses its own kinematic-pressure scale, never a velocity unit.
    random[:nf][~velocity_mask] *= max(float(np.max(np.abs(flow.pressure))), speed * speed)
    rows = []
    for name, direction in (
        ("scaled_random", random),
        ("newton", update / max(np.max(np.abs(update)), 1e-30)),
    ):
        action = H @ direction
        for epsilon in (1e-4, 1e-5, 1e-6, 1e-7):
            plus, plus_flow = displaced(problem, state, flow, direction, epsilon)
            minus, minus_flow = displaced(problem, state, flow, direction, -epsilon)
            rp = step_equations(
                problem, model, plus, plus_flow, source, previous, previous_flow, slab
            )[0]
            rm = step_equations(
                problem, model, minus, minus_flow, source, previous, previous_flow, slab
            )[0]
            fd = (rp - rm) / (2 * epsilon)
            rows.append(
                {
                    "direction": name,
                    "epsilon": epsilon,
                    "scaled_relative_error": float(
                        np.linalg.norm(scaling * (fd - action))
                        / max(np.linalg.norm(scaling * action), 1e-30)
                    ),
                    "relative_error": float(
                        np.linalg.norm(fd - action) / max(np.linalg.norm(action), 1e-30)
                    ),
                    "plus_branch_changes": int(
                        np.count_nonzero(
                            stabilization_branches(problem, plus_flow.velocity) != original_branches
                        )
                    ),
                    "minus_branch_changes": int(
                        np.count_nonzero(
                            stabilization_branches(problem, minus_flow.velocity)
                            != original_branches
                        )
                    ),
                }
            )
    return {
        "original_equations": metrics,
        "linear_scaled_relative_residual": float(
            np.linalg.norm(scaling * (residual + H @ update)) / denominator
        ),
        "maximum_velocity_increment_m_s": float(
            np.max(np.abs(update[:nf][velocity_mask]), initial=0)
        ),
        "maximum_pressure_increment_m2_s2": float(
            np.max(np.abs(update[:nf][~velocity_mask]), initial=0)
        ),
        "maximum_temperature_increment_K": float(
            np.max(np.abs(update[nf:])) * problem.temperature_scale
        ),
        "branch_counts_diffusive_advective_row_limit": np.bincount(
            original_branches, minlength=3
        ).tolist(),
        "finite_differences": rows,
    }


def preceding_fields(problem, directory, physical_time, source_sha256):
    """Use only a verified saved preceding state with the same frozen source."""
    if physical_time == 0:
        return problem.full_temperature(problem.initial), problem.initial_flow
    record = json.loads((directory / "record.json").read_text())
    if record["optimization_field_sha256"] != source_sha256:
        raise ValueError("Local initial fields came from a different source")
    if not record.get("forward_formulation", {}).get("consistent_stabilization"):
        raise ValueError("Local initial fields require the corrected thermal formulation")
    with np.load(directory / "states.npz", allow_pickle=False) as stored:
        indices = np.flatnonzero(np.isclose(stored["times_s"], physical_time, rtol=0, atol=1e-10))
        if len(indices) != 1:
            raise ValueError("The common initial time must be an existing saved time level")
        index = int(indices[0])
        if any(row["status"] != "converged" for row in record["steps"][: index + 1]):
            raise ValueError("An unverified trajectory cannot supply local initial fields")
        return problem.full_temperature(stored["state"][index]), FlowResult(
            stored["velocity"][index].copy(), stored["pressure"][index].copy(), "saved_verified", []
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--target-position", type=int, default=0)
    parser.add_argument(
        "--interval", type=int, required=True, help="Zero-based original source interval"
    )
    parser.add_argument("--policy", choices=tuple(POLICIES), required=True)
    parser.add_argument("--mode", choices=("diagnose", "window"), required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        saved, cfg, fields, digest = load_saved_solution(
            args.optimization, "reference", args.target_position
        )
        if not cfg["transient"]:
            raise ValueError("The local study requires a complete transient source")
        interval = integer(args.interval, "Source interval", 0)
        if interval >= cfg["slabs"]:
            raise ValueError("Source interval is outside the saved trajectory")
        start = time.perf_counter()
        protocol = {
            "schema": "coupled-local-diagnostic-v1",
            "environment": environment(),
            "optimization_field_sha256": digest,
            "history_record_sha256": file_sha256(args.history / "record.json"),
            "history_fields_sha256": file_sha256(args.history / "states.npz"),
            "source_interval_zero_based": interval,
            "policy": args.policy,
            "settings": {
                **POLICIES[args.policy],
                "line_search": "fixed_scaled",
                "newton_cap": 100,
                "tolerance": 1e-12,
                "mass_tolerance": 1e-6,
                "energy_tolerance": 1e-6,
            },
            "consistent_stabilization": True,
            "mode": args.mode,
            "scope": "Fixed source; one common initial state per window; no clipping or reoptimization",
        }
        outcomes = []
        for subdivision in (1,) if args.mode == "diagnose" else (1, 2, 4):
            problem, baseline = load_problem(
                {
                    **cfg,
                    "baseline_directory": str(args.baseline),
                    "slabs": cfg["slabs"] * subdivision,
                    "consistent_stabilization": True,
                }
            )
            require_matching_baseline(saved, baseline)
            beginning = interval * cfg["horizon_s"] / cfg["slabs"]
            state, flow = preceding_fields(problem, args.history, beginning, digest)
            source = np.zeros(len(problem.mesh.nodes))
            source[problem.free] = fields["control"].reshape(cfg["slabs"], -1)[interval]
            rows = []
            for local in range(subdivision):
                slab = interval * subdivision + local
                before_state, before_flow = state.copy(), flow
                result = newton_step(
                    problem,
                    source,
                    state,
                    flow,
                    slab,
                    tolerance=1e-12,
                    max_iterations=100,
                    line_search="fixed_scaled",
                    **POLICIES[args.policy],
                )
                details = {
                    "slab_zero_based": slab,
                    "time_s": float(problem.physical_steps[: slab + 1].sum()),
                    "history": result.history,
                    "status": result.status,
                }
                snapshot = save_snapshot(
                    args.output,
                    f"subdivision-{subdivision}-step-{local}",
                    result,
                    details,
                    protocol,
                )
                row = {**details, "snapshot": snapshot, "seconds": result.seconds}
                rows.append(row)
                write_report(
                    args.output / "record.json",
                    {
                        **protocol,
                        "status": "running",
                        "completed_subdivisions": outcomes,
                        "current_subdivision": subdivision,
                        "current_steps": rows,
                    },
                )
                if args.mode == "diagnose":
                    # Persist the returned stalled state before performing any
                    # additional diagnostics, which can themselves be interrupted.
                    for label, value, velocity in (
                        ("initial", before_state, before_flow),
                        ("returned", result.state, result.flow),
                    ):
                        try:
                            report = derivative_diagnostic(
                                problem, source, before_state, before_flow, slab, value, velocity
                            )
                        except (ValueError, RuntimeError) as error:
                            report = {
                                "status": "diagnostic_failure",
                                "type": type(error).__name__,
                                "message": str(error),
                            }
                        write_report(args.output / f"derivative-{label}.json", report)
                state, flow = result.state, result.flow
                if result.status != "converged":
                    break
            write_fields(
                args.output / f"endpoint-{subdivision}.npz",
                state=state,
                velocity=flow.velocity,
                pressure=flow.pressure,
            )
            outcomes.append({"subdivision": subdivision, "status": result.status, "steps": rows})
        differences = []
        for coarse, fine in zip(outcomes[:-1], outcomes[1:]):
            if coarse["status"] == fine["status"] == "converged":
                with (
                    np.load(args.output / f"endpoint-{coarse['subdivision']}.npz") as a,
                    np.load(args.output / f"endpoint-{fine['subdivision']}.npz") as b,
                ):
                    differences.append(
                        {
                            "coarse": coarse["subdivision"],
                            "fine": fine["subdivision"],
                            "maximum_temperature_difference_K": float(
                                np.max(np.abs(a["state"] - b["state"])) * problem.temperature_scale
                            ),
                        }
                    )
        write_report(
            args.output / "record.json",
            {
                **protocol,
                "status": "completed",
                "all_steps_converged": all(row["status"] == "converged" for row in outcomes),
                "completed_subdivisions": outcomes,
                "endpoint_comparisons": differences,
                "seconds": time.perf_counter() - start,
            },
        )


if __name__ == "__main__":
    main()
