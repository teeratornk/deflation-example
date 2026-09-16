"""Test the quarter-relaxation factor suggested by the measured iteration mode.

Two predeclared policies, ordinary relaxation and temperature-only Anderson-5,
start from the same failed fields as the earlier four-policy comparison.
The PDE, source, time step, inner flow criterion and final criteria are fixed.
This follow-up is selected using a computed mode, not a complete contraction
certificate. All outcomes are retained separately from the earlier screen.
"""

import argparse
from pathlib import Path
import time

from threadpoolctl import threadpool_limits

from step_diagnostics import load_case
from deflation_example.coupled_fixed_point import solve_forward
from deflation_example.coupled_newton_replay import step_equations, criteria_met
from deflation_example.coupled_resolution import forward_model
from deflation_example.coupled_saved import file_digest
from deflation_example.reporting import environment, write_arrays, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "fine-baseline", "optimization", "trajectory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--slab", type=int)
    parser.add_argument("--acceleration", choices=("relaxed", "anderson"), required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    record = {
        "status": "running",
        "environment": environment(),
        "driver_sha256": file_digest(Path(__file__)),
        "scope": __doc__,
        "policy": args.acceleration,
        "relaxation": 0.25,
        "coupling_cap": 300,
        "depth": 5,
        "flow_cap": 100,
        "accuracy": {"equation": 1e-12, "mass": 1e-6, "energy": 1e-6},
    }
    write_report(args.output / "record.json", record)
    try:
        with threadpool_limits(args.threads):
            problem, source, previous, previous_flow, state, flow, n, original = load_case(args)
            record.update(
                slab_zero_based=n,
                time_s=original["steps"][n]["time_s"],
                initial_field_sha256=original["steps"][n]["field_sha256"],
                trajectory_record_sha256=file_digest(args.trajectory / "record.json"),
            )
            write_report(args.output / "record.json", record)
            result = solve_forward(
                forward_model(problem),
                source,
                state,
                flow,
                previous_state=previous,
                previous_velocity=previous_flow.velocity,
                time_step=float(problem.physical_steps[n]),
                policy=args.acceleration,
                relaxation=0.25,
                depth=5,
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
                status="complete",
                termination=result.status,
                checks=checks,
                verified=bool(result.status == "converged" and criteria_met(checks, 1e-12)),
                solve_seconds=result.seconds,
                history=result.history,
                field_sha256=file_digest(args.output / "fields.npz"),
            )
    except Exception as error:
        record.update(status="execution_failure", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["process_seconds"] = time.perf_counter() - start
        write_report(args.output / "record.json", record)


if __name__ == "__main__":
    main()
