"""Continue an unchanged trajectory after a separately verified local correction.

The prefix and correction retain their input hashes and original costs. This
is a mixed-procedure continuation, not a fresh single-policy timing experiment.
All prefix and new states are independently verified against the same equations.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from step_diagnostics import checked_field, load_case
from deflation_example.coupled_newton_replay import criteria_met, newton_step, step_equations
from deflation_example.coupled_resolution import forward_model
from deflation_example.coupled_saved import file_digest
from deflation_example.fixed_point_study import commit_step, prepare
from deflation_example.reporting import environment, write_report


def verified_row(problem, source, previous, previous_flow, state, flow, n):
    _, checks = step_equations(
        problem, forward_model(problem), state, flow, source, previous, previous_flow, n
    )
    return {
        "slab_zero_based": n,
        "time_s": float(problem.physical_steps[: n + 1].sum()),
        "verified": bool(criteria_met(checks, 1e-12)),
        "checks": checks,
    }


def advance(problem, source, previous, previous_flow, n):
    result = newton_step(
        problem,
        source,
        previous,
        previous_flow,
        n,
        tolerance=1e-12,
        max_iterations=100,
        line_search="fixed_scaled",
        backtrack_cap=40,
    )
    row = verified_row(problem, source, previous, previous_flow, result.state, result.flow, n)
    row.update(
        status=result.status,
        seconds=result.seconds,
        history=result.history,
        procedure="fixed_scaled_newton",
    )
    row["verified"] = bool(row["verified"] and result.status == "converged")
    return row, {
        "state": result.state[problem.free],
        "velocity": result.flow.velocity,
        "pressure": result.flow.pressure,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "fine-baseline", "optimization", "trajectory", "repair", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--slab", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    meta = {
        "status": "running",
        "environment": environment(),
        "driver_sha256": file_digest(Path(__file__)),
        "scope": __doc__,
        "steps": [],
        "attempts": [],
        "policy": {
            "line_search": "fixed_scaled",
            "newton_cap": 100,
            "backtrack_cap": 40,
            "equation_tolerance": 1e-12,
            "balance_tolerance": 1e-6,
        },
    }
    with threadpool_limits(args.threads):
        problem, _, _, _, _, _, corrected, original = load_case(args)
        repair = json.loads((args.repair / "record.json").read_text())
        if not (
            repair["status"] == "complete"
            and repair["verified"]
            and repair["trajectory_record_sha256"] == file_digest(args.trajectory / "record.json")
            and repair["initial_field_sha256"] == original["steps"][corrected]["field_sha256"]
            and repair["slab_zero_based"] == corrected
        ):
            raise ValueError("Correction must verify this unchanged failed step")
        _, controls, _ = prepare(args, original["subdivision"])
        meta.update(
            trajectory_record_sha256=file_digest(args.trajectory / "record.json"),
            repair_record_sha256=file_digest(args.repair / "record.json"),
            configuration=original["configuration"],
            slabs=problem.slabs,
            corrected_slab=corrected,
            original_attempt_process_seconds=original["all_attempt_process_seconds"],
            correction_process_seconds=repair["process_seconds"],
            timing_scope="Prefix work includes its failed attempt; correction and continuation process intervals remain separate. No single-policy performance comparison.",
        )
        if args.resume:
            saved = json.loads((args.output / "record.json").read_text())
            if saved["status"] != "running":
                raise ValueError("Only interrupted running continuations may resume")
            for key in (
                "trajectory_record_sha256",
                "repair_record_sha256",
                "driver_sha256",
                "policy",
                "configuration",
            ):
                if saved[key] != meta[key]:
                    raise ValueError("Continuation restart identity differs: " + key)
            if saved["environment"]["source_sha256"] != meta["environment"]["source_sha256"]:
                raise ValueError("Continuation source changed")
            meta = saved
        meta["attempts"].append({"started_unix": time.time()})
        write_report(args.output / "record.json", meta)
        state, flow = problem.full_temperature(problem.initial), problem.initial_flow
        try:
            # Verify any interrupted committed prefix before continuing it.
            for n, row in enumerate(meta["steps"]):
                if row["slab_zero_based"] != n or not row["verified"]:
                    raise ValueError("Restart requires consecutive verified states")
                current, current_flow = checked_field(args.output, row, problem)
                source = np.zeros(len(problem.mesh.nodes))
                source[problem.free] = controls[n]
                if not verified_row(problem, source, state, flow, current, current_flow, n)[
                    "verified"
                ]:
                    raise ValueError("Stored continuation fails fresh verification")
                state, flow = current, current_flow
            for n in range(len(meta["steps"]), problem.slabs):
                source = np.zeros(len(problem.mesh.nodes))
                source[problem.free] = controls[n]
                write_report(
                    args.output / "progress.json",
                    {
                        "slab_zero_based": n,
                        "time_s": float(problem.physical_steps[: n + 1].sum()),
                        "status": "verifying_prefix" if n <= corrected else "solving",
                    },
                )
                if n <= corrected:
                    directory, descriptor = (
                        (args.trajectory, original["steps"][n])
                        if n < corrected
                        else (
                            args.repair,
                            {"fields": "fields.npz", "field_sha256": repair["field_sha256"]},
                        )
                    )
                    current, current_flow = checked_field(directory, descriptor, problem)
                    row = verified_row(problem, source, state, flow, current, current_flow, n)
                    row.update(
                        status="converged" if row["verified"] else "prefix_verification_failed",
                        procedure="imported_verified_prefix"
                        if n < corrected
                        else "verified_newton_correction",
                        source_field_sha256=descriptor["field_sha256"],
                    )
                    arrays = {
                        "state": current[problem.free],
                        "velocity": current_flow.velocity,
                        "pressure": current_flow.pressure,
                    }
                else:
                    row, arrays = advance(problem, source, state, flow, n)
                meta = commit_step(args.output, meta, row, arrays)
                if not row["verified"]:
                    meta["status"] = "numerical_failure"
                    break
                state, flow = checked_field(args.output, meta["steps"][-1], problem)
            else:
                meta["status"] = "converged"
        except Exception as error:
            meta.update(
                status="execution_failure", error_type=type(error).__name__, error=str(error)
            )
            raise
        finally:
            meta["attempts"][-1]["process_seconds"] = time.perf_counter() - began
            durations = [a.get("process_seconds") for a in meta["attempts"]]
            meta["continuation_process_seconds"] = (
                sum(durations) if all(t is not None for t in durations) else None
            )
            write_report(args.output / "record.json", meta)


if __name__ == "__main__":
    main()
