"""Recompute coupled equations from fixed-source forward fields without relabeling a run."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_newton_replay import criteria_met, replay_configuration, step_equations
from .coupled_optimize import load_problem
from .coupled_resolution import forward_model
from .coupled_saved import load_saved_solution, require_matching_baseline
from .coupled_time_integration import effective_step
from .reporting import environment, file_sha256, write_report
from .validation import integer


def verify_fields(problem, controls, fields, rows, *, time_scheme, subdivision):
    """Check every saved step, including an unsuccessful final candidate."""
    controls = np.asarray(controls)
    count = len(rows)
    shapes = {
        "state": (count, problem.spatial_size),
        "velocity": (count, problem.flow.nv, 2),
        "pressure": (count, problem.flow.np),
        "times_s": (count,),
    }
    if not 0 < count <= problem.slabs or controls.shape != (problem.slabs, problem.spatial_size):
        raise ValueError("Source and saved steps must match the declared trajectory")
    if not np.isfinite(controls).all() or any(
        np.asarray(fields[k]).shape != shape or not np.isfinite(fields[k]).all()
        for k, shape in shapes.items()
    ):
        raise ValueError("Finite saved fields must match the model dimensions")
    times = np.cumsum(problem.physical_steps)[:count]
    if not np.allclose(fields["times_s"], times, rtol=0, atol=1e-10):
        raise ValueError("Saved times differ from the declared discretization")
    if not np.allclose([r["time_s"] for r in rows], times, rtol=0, atol=1e-10):
        raise ValueError("Record and field times differ")
    previous = problem.full_temperature(problem.initial)
    previous_flow = problem.initial_flow
    older, older_flow = None, None
    checks = []
    for n, row in enumerate(rows):
        state = problem.full_temperature(fields["state"][n])
        flow = FlowResult(fields["velocity"][n], fields["pressure"][n], "saved", [])
        effective, history_state, history_flow, _ = effective_step(
            problem,
            n,
            previous,
            previous_flow,
            older,
            older_flow,
            time_scheme,
            restart=n % subdivision == 0,
        )
        source = np.zeros(len(problem.mesh.nodes))
        source[problem.free] = controls[n]
        _, metrics = step_equations(
            effective,
            forward_model(effective),
            state,
            flow,
            source,
            history_state,
            history_flow,
            n,
        )
        boundary_error = float(
            np.max(np.abs(flow.velocity[problem.boundary_indices] - problem.boundary_values))
        )
        gauge_error = 0.0
        if problem.pressure_gauge is not None:
            index, value = problem.pressure_gauge
            gauge_error = float(abs(flow.pressure[index] - value))
        checks.append(
            {
                "slab_zero_based": n,
                "time_s": float(times[n]),
                "recorded_status": row["status"],
                "equations": metrics,
                "velocity_boundary_maximum_absolute_error": boundary_error,
                "pressure_gauge_absolute_error": gauge_error,
                "verified": bool(
                    row["status"] == "converged"
                    and criteria_met(metrics, 1e-12)
                    and boundary_error <= 1e-12
                    and gauge_error <= 1e-12
                ),
            }
        )
        older, older_flow = previous, previous_flow
        previous, previous_flow = state, flow
    return {
        "all_saved_steps_verified": all(r["verified"] for r in checks),
        "complete_trajectory_verified": count == problem.slabs
        and all(r["verified"] for r in checks),
        "saved_steps": count,
        "declared_steps": problem.slabs,
        "steps": checks,
        "maximum_recomputed_equations": {
            k: max(r["equations"][k] for r in checks) for k in checks[0]["equations"]
        },
        "equation_tolerance": 1e-12,
        "conservation_tolerance": 1e-6,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("reference", "jacobi", "recycling"), default="reference"
    )
    parser.add_argument("--target-position", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        original, cfg, source_fields, digest = load_saved_solution(
            args.optimization, args.method, args.target_position
        )
        record = json.loads((args.forward / "record.json").read_text())
        if record["optimization_field_sha256"] != digest or record["status"] == "running":
            raise ValueError("Select a finished forward run of the unchanged saved source")
        if record["configuration"] != cfg:
            raise ValueError("Forward and optimization configurations differ")
        subdivision = integer(record["subdivision"], "Subdivision", 1)
        formulation = record["forward_formulation"]
        if formulation["streamline_rule"] != cfg.get("streamline_rule", "hard_min"):
            raise ValueError("Forward stabilization differs from the saved source model")
        problem, baseline = load_problem(
            replay_configuration(
                cfg, args.baseline, subdivision, formulation["consistent_stabilization"]
            )
        )
        require_matching_baseline(original, baseline)
        require_matching_baseline(record, baseline)
        if record["forward_slabs"] != problem.slabs:
            raise ValueError("Forward dimension differs from the declared subdivision")
        controls = np.repeat(
            source_fields["control"].reshape(cfg["slabs"], -1), subdivision, axis=0
        )
        with np.load(args.forward / "states.npz", allow_pickle=False) as data:
            fields = {k: data[k].copy() for k in ("state", "velocity", "pressure", "times_s")}
        report = verify_fields(
            problem,
            controls,
            fields,
            record["steps"],
            time_scheme=record["forward_solver"]["time_scheme"],
            subdivision=subdivision,
        )
        report["complete_trajectory_verified"] &= record["status"] == "converged"
        write_report(
            args.output,
            {
                **report,
                "schema": "coupled-saved-forward-equation-check-v1",
                "environment": environment(),
                "forward_source": record["environment"]["git_head"],
                "forward_record_sha256": file_sha256(args.forward / "record.json"),
                "forward_fields_sha256": file_sha256(args.forward / "states.npz"),
                "optimization_fields_sha256": digest,
                "baseline_sha256": baseline["baseline_sha256"],
                "forward_formulation": formulation,
                "scope": "Fresh equation and boundary checks on saved fields with the original fixed source; no state changes, clipping, reoptimization or relabeling of the forward run. Temporal resolution remains separate.",
            },
        )


if __name__ == "__main__":
    main()
