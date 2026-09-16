"""Matched local correction of an immutable failed coupled step.

All policies retain the physical problem, source, preceding state and final
accuracy. Coupled mixing is a separately versioned Anderson variant. A local
convergence result does not replace the complete trajectory or optimizer study.
"""

import argparse
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from step_diagnostics import load_case
from deflation_example.coupled_fixed_point import solve_forward
from deflation_example.coupled_newton_replay import newton_step, step_equations, criteria_met
from deflation_example.coupled_resolution import forward_model
from deflation_example.coupled_saved import file_digest
from deflation_example.reporting import environment, write_arrays, write_report


def solve(problem, source, previous, previous_flow, state, flow, slab, policy, callback=None):
    if policy in {"equation_max", "fixed_scaled"}:
        return newton_step(
            problem,
            source,
            previous,
            previous_flow,
            slab,
            initial_state=state,
            initial_flow=flow,
            tolerance=1e-12,
            max_iterations=100,
            line_search=policy,
            backtrack_cap=21 if policy == "equation_max" else 40,
        )
    if policy not in {"temperature", "coupled"}:
        raise ValueError("Unknown correction policy")
    return solve_forward(
        forward_model(problem),
        source,
        state,
        flow,
        previous_state=previous,
        previous_velocity=previous_flow.velocity,
        time_step=float(problem.physical_steps[slab]),
        policy="anderson",
        depth=5,
        relaxation=0.5,
        mixing=policy,
        tolerance=1e-12,
        max_iterations=300,
        flow_cap=100,
        callback=callback,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "fine-baseline", "optimization", "trajectory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--slab", type=int)
    parser.add_argument(
        "--correction",
        choices=("equation_max", "fixed_scaled", "temperature", "coupled"),
        required=True,
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    record = {
        "status": "running",
        "policy": args.correction,
        "environment": environment(),
        "driver_sha256": file_digest(Path(__file__)),
        "loader_sha256": file_digest(Path(__file__).with_name("step_diagnostics.py")),
        "accuracy": {"equation": 1e-12, "mass": 1e-6, "energy": 1e-6},
        "scope": "Local solve with a verified preceding state and unchanged signed control; costs exclude the original failed attempt and do not constitute a complete trajectory speedup.",
    }
    write_report(args.output / "record.json", record)
    try:
        with threadpool_limits(args.threads):
            problem, source, previous, previous_flow, state, flow, n, original = load_case(args)
            record.update(
                trajectory_record_sha256=file_digest(args.trajectory / "record.json"),
                initial_field_sha256=original["steps"][n]["field_sha256"],
                slab_zero_based=n,
                time_s=original["steps"][n]["time_s"],
                initial_status=original["steps"][n]["status"],
            )
            before = [
                x.copy() for x in (source, previous, previous_flow.velocity, previous_flow.pressure)
            ]
            write_report(args.output / "record.json", record)
            result = solve(
                problem,
                source,
                previous,
                previous_flow,
                state,
                flow,
                n,
                args.correction,
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
            for a, b in zip(
                before,
                (source, previous, previous_flow.velocity, previous_flow.pressure),
                strict=True,
            ):
                if not np.array_equal(a, b):
                    raise RuntimeError("Correction changed the fixed source or preceding fields")
            write_arrays(
                args.output / "fields.npz",
                state=result.state[problem.free],
                velocity=result.flow.velocity,
                pressure=result.flow.pressure,
            )
            record.update(
                status="complete",
                termination=result.status,
                verified=bool(result.status == "converged" and criteria_met(checks, 1e-12)),
                checks=checks,
                history=result.history,
                solve_seconds=result.seconds,
                field_sha256=file_digest(args.output / "fields.npz"),
            )
    except Exception as error:
        record.update(status="execution_failure", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["process_seconds"] = time.perf_counter() - began
        write_report(args.output / "record.json", record)


if __name__ == "__main__":
    main()
