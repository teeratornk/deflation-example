"""Compare Newton globalization rules at a saved failed forward step.

Every comparison starts from identical saved fields and uses the original
equations and stopping criteria. Convergence here concerns one step with a
saved preceding state; a complete forward replay is a separate experiment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_newton_replay import criteria_met, newton_step, step_equations
from .coupled_optimize import load_problem
from .coupled_resolution import forward_model
from .coupled_saved import load_saved_solution, require_matching_baseline, file_digest
from .coupled_spatial_resolution import transfer_source
from .coupled_time_integration import replay_time_scheme, saved_history
from .reporting import environment, write_fields, write_report
from .validation import integer


def compare_step(
    problem, source, previous_state, previous_flow, state, flow, slab, *, cap=100, callback=None
):
    rows, fields = [], {}
    for policy, backtracks in (("equation_max", 21), ("fixed_scaled", 40)):
        result = newton_step(
            problem,
            source,
            previous_state,
            previous_flow,
            slab,
            initial_state=state,
            initial_flow=flow,
            tolerance=1e-12,
            max_iterations=cap,
            line_search=policy,
            backtrack_cap=backtracks,
        )
        _, checks = step_equations(
            problem,
            forward_model(problem),
            result.state,
            result.flow,
            source,
            previous_state,
            previous_flow,
            slab,
        )
        rows.append(
            {
                "line_search": policy,
                "backtrack_cap": backtracks,
                "newton_cap": cap,
                "tolerance": 1e-12,
                "status": result.status,
                "seconds": result.seconds,
                "independent_checks": checks,
                "independent_criteria_met": criteria_met(checks, 1e-12),
                "history": result.history,
            }
        )
        fields[f"{policy}_state"] = result.state
        fields[f"{policy}_velocity"] = result.flow.velocity
        fields[f"{policy}_pressure"] = result.flow.pressure
        if callback is not None:
            callback(rows.copy(), fields.copy())
    return rows, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fine-baseline", type=Path, help="Omit for same-mesh temporal replay")
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("reference", "jacobi", "recycling"), default="reference"
    )
    parser.add_argument("--target-position", type=int)
    parser.add_argument("--cap", type=int, default=100)
    parser.add_argument(
        "--protocol", choices=("legacy", "refinement_v7", "continuation_v7"), default="legacy"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cap = integer(args.cap, "Newton cap", 1)
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        saved, config, fields, digest = load_saved_solution(
            args.optimization, args.method, args.target_position
        )
        replay = json.loads((args.replay / "record.json").read_text())
        if replay["configuration"] != config:
            raise ValueError("Replay must retain the saved optimization configuration")
        if replay["status"] in {"converged", "running"}:
            raise ValueError("Select a completed, unsuccessful forward replay")
        subdivision = integer(replay.get("subdivision", 1), "Time subdivision", 1)
        comparison = {**config, "slabs": config["slabs"] * subdivision}
        coarse, baseline = load_problem({**comparison, "baseline_directory": str(args.baseline)})
        fine, fine_baseline = (
            (coarse, baseline)
            if args.fine_baseline is None
            else load_problem({**comparison, "baseline_directory": str(args.fine_baseline)})
        )
        require_matching_baseline(saved, baseline)
        if (
            replay["optimization_field_sha256"] != digest
            or replay["baseline_sha256"] != baseline["baseline_sha256"]
            or replay.get("fine_baseline_sha256", replay["baseline_sha256"])
            != fine_baseline["baseline_sha256"]
        ):
            raise ValueError("Replay, saved control and baselines must match")
        controls = np.repeat(fields["control"].reshape(config["slabs"], -1), subdivision, axis=0)
        if args.fine_baseline is not None:
            controls, _ = transfer_source(coarse, fine, controls)
        formulation = replay.get("forward_formulation", {})
        if (
            formulation.get("consistent_stabilization", fine.consistent_stabilization)
            != fine.consistent_stabilization
            or formulation.get("streamline_rule", fine.streamline_rule) != fine.streamline_rule
        ):
            raise ValueError("Repair must preserve the actual replay formulation")
        with np.load(args.replay / "states.npz", allow_pickle=False) as arrays:
            states, velocities, pressures, times = (
                arrays[key] for key in ("state", "velocity", "pressure", "times_s")
            )
        n = len(states) - 1
        if (
            n < 0
            or len(replay["steps"]) != n + 1
            or any(row["status"] != "converged" for row in replay["steps"][:n])
            or replay["steps"][n]["status"] == "converged"
            or not np.allclose(times, np.cumsum(fine.physical_steps)[: n + 1])
        ):
            raise ValueError(
                "Saved fields must identify the first unsuccessful step at matching times"
            )
        state = fine.full_temperature(states[n])
        flow = FlowResult(velocities[n], pressures[n], "saved", [])
        scheme = replay_time_scheme(replay)
        effective, previous, previous_flow, coefficients = saved_history(
            fine, states, velocities, pressures, n, scheme, restart=n % subdivision == 0
        )
        source = np.zeros(len(fine.mesh.nodes))
        source[fine.free] = controls[n]
        args.output.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema": "coupled-newton-globalization-comparison-v1",
            "environment": environment(),
            "optimization_field_sha256": digest,
            "baseline_sha256": baseline["baseline_sha256"],
            "fine_baseline_sha256": fine_baseline["baseline_sha256"],
            "replay_record_sha256": file_digest(args.replay / "record.json"),
            "replay_fields_sha256": file_digest(args.replay / "states.npz"),
            "saved_replay_status": replay["status"],
            "time_s": float(times[n]),
            "slab_zero_based": n,
            "spatial_state_degrees_of_freedom": fine.spatial_size,
            "time_scheme": scheme,
            "protocol": args.protocol,
            "streamline_rule": fine.streamline_rule,
            "consistent_stabilization": fine.consistent_stabilization,
            "storage_derivative_coefficients_s_inverse": coefficients.tolist(),
            "scope": "Matched local Newton restarts from the saved unsuccessful candidate. The previous state and signed source are fixed; complete trajectory accuracy remains a separate check.",
        }
        write_report(args.output / "record.json", {**metadata, "status": "running"})

        if args.protocol in {"refinement_v7", "continuation_v7"}:
            from .coupled_repair_protocol import derivative_check, compare_repairs, paired_selection
            from .coupled_forward_verify import verify_fields

            audit = verify_fields(
                fine,
                controls,
                {"state": states, "velocity": velocities, "pressure": pressures, "times_s": times},
                replay["steps"],
                time_scheme=scheme,
                subdivision=subdivision,
            )
            metadata["input_equation_audit"] = audit
            if not all(row["verified"] for row in audit["steps"][:-1]):
                write_report(
                    args.output / "record.json",
                    {**metadata, "status": "previous_history_check_failed"},
                )
                return
            metadata["scope"] = (
                "Matched fixed-source repairs of one failed step, from the saved candidate and preceding physical state. Both starts use the same independently verified temporal history. Complete trajectory accuracy remains a separate check."
            )

            metadata["derivative_check"] = derivative_check(
                effective, source, previous, previous_flow, state, flow, n
            )
            if not metadata["derivative_check"]["passed"]:
                write_report(
                    args.output / "record.json", {**metadata, "status": "derivative_check_failed"}
                )
                return
            rows, solutions = [], {}
            initializations = (
                ("saved", state, flow),
                (
                    "previous",
                    fine.full_temperature(fine.initial)
                    if n == 0
                    else fine.full_temperature(states[n - 1]),
                    fine.initial_flow
                    if n == 0
                    else FlowResult(velocities[n - 1], pressures[n - 1], "previous", []),
                ),
            )
            for name, initial_state, initial_flow in initializations:

                def progress(current, arrays):
                    write_fields(args.output / "states.npz", **{**solutions, **arrays})
                    write_report(
                        args.output / "record.json",
                        {**metadata, "status": "running", "rows": rows + current},
                    )

                current, arrays = compare_repairs(
                    effective,
                    source,
                    previous,
                    previous_flow,
                    initial_state,
                    initial_flow,
                    n,
                    initialization=name,
                    cap=cap,
                    callback=progress,
                    continuation=args.protocol == "continuation_v7",
                )
                rows.extend(current)
                solutions.update(arrays)
            write_fields(args.output / "states.npz", **solutions)
            write_report(
                args.output / "record.json",
                {
                    **metadata,
                    "status": "complete",
                    "rows": rows,
                    "selection": paired_selection(
                        rows,
                        solutions,
                        fine.temperature_scale,
                        continuation=args.protocol == "continuation_v7",
                    ),
                },
            )
            return

        def completed_policy(rows, solutions):
            write_fields(args.output / "states.npz", **solutions)
            write_report(
                args.output / "record.json", {**metadata, "status": "running", "rows": rows}
            )

        rows, solutions = compare_step(
            effective,
            source,
            previous,
            previous_flow,
            state,
            flow,
            n,
            cap=cap,
            callback=completed_policy,
        )
        write_fields(args.output / "states.npz", **solutions)
        write_report(args.output / "record.json", {**metadata, "status": "complete", "rows": rows})


if __name__ == "__main__":
    main()
