"""Repeat the corrected forward procedure from the physical initial condition.

This is a fixed-control verification run, not an optimization benchmark. Every
saved state is checked in the original equations. Interrupted runs can resume
only with identical inputs, source, driver, and numerical settings.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from continue_repaired import advance, verified_row
from step_diagnostics import checked_field
from deflation_example.coupled_saved import file_digest
from deflation_example.coupled_targets import desired_temperature
from deflation_example.fixed_point_study import commit_step, prepare, verify_restart
from deflation_example.memory import ProcessMemory
from deflation_example.reporting import write_report


SETTINGS = {
    "line_search": "fixed_scaled",
    "newton_cap": 100,
    "backtrack_cap": 40,
    "equation_tolerance": 1e-12,
    "mass_tolerance": 1e-6,
    "energy_tolerance": 1e-6,
}


def initial_fields(problem):
    return problem.full_temperature(problem.initial), problem.initial_flow


def restore(problem, controls, output, saved, expected):
    """Re-evaluate the complete committed prefix before accepting a restart."""
    if saved["status"] != "running":
        raise ValueError("Only an interrupted running trajectory may resume")
    for key in ("schema", "driver_sha256", "driver_dependencies", "settings", "configuration"):
        if saved[key] != expected[key]:
            raise ValueError("Restart identity differs: " + key)
    verify_restart(saved, expected, output)
    state, flow = initial_fields(problem)
    if len(saved["steps"]) > problem.slabs:
        raise ValueError("Stored trajectory exceeds the declared time grid")
    for n, row in enumerate(saved["steps"]):
        if not np.isclose(row["time_s"], problem.physical_steps[: n + 1].sum(), atol=1e-9, rtol=0):
            raise ValueError("Stored physical time differs")
        current, current_flow = checked_field(output, row, problem)
        source = np.zeros(len(problem.mesh.nodes))
        source[problem.free] = controls[n]
        fresh = verified_row(problem, source, state, flow, current, current_flow, n)
        if not fresh["verified"]:
            raise ValueError("Stored prefix fails independent equation verification")
        state, flow = current, current_flow
    return state, flow


def quantities(problem, output, record):
    """Compute physical quantities without stacking all velocity/pressure fields."""
    if len(record["steps"]) != problem.slabs or not all(r["verified"] for r in record["steps"]):
        raise ValueError("Physical trajectory summaries require every declared step")
    cfg = record["configuration"]
    desired = desired_temperature(
        problem, cfg["query"], cfg["target_count"], cfg.get("target_startup_s", 0)
    ).reshape(problem.slabs, -1)
    weights = problem.weights.reshape(problem.slabs, -1)
    peak, minimum, tracking = -np.inf, np.inf, 0.0
    peak_time, peak_location = None, None
    for n, row in enumerate(record["steps"]):
        full_state, _ = checked_field(output, row, problem)
        state = full_state[problem.free]
        physical = problem.temperature_offset + problem.temperature_scale * state
        index = int(np.argmax(physical))
        if physical[index] > peak:
            peak, peak_time = float(physical[index]), row["time_s"]
            peak_location = problem.mesh.nodes[problem.free[index]].tolist()
        minimum = min(minimum, float(physical.min()))
        tracking += float(np.sum(weights[n] * (state - desired[n]) ** 2))
    return {
        "domain": "Constrained free temperature nodes at every declared physical time level",
        "maximum_temperature_K": peak,
        "minimum_temperature_K": minimum,
        "maximum_time_s": peak_time,
        "maximum_location_rz": peak_location,
        "upper_bound_violation_K": max(0.0, peak - cfg["upper_K"]),
        "lower_bound_violation_K": max(0.0, cfg["lower_K"] - minimum),
        "weighted_tracking_squared_K2": problem.temperature_scale**2 * tracking / weights.sum(),
        "scope": "Discrete forward response to an unchanged saved source; resolution is a separate test.",
    }


def run(args):
    started = time.perf_counter()
    if args.resume:
        if not (args.output / "record.json").is_file():
            raise ValueError("Resume requires an existing manifest")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "corrected-fresh-trajectory-v1",
        "status": "running",
        "driver_sha256": file_digest(Path(__file__)),
        "driver_dependencies": {
            name: file_digest(Path(__file__).with_name(name))
            for name in ("continue_repaired.py", "step_diagnostics.py")
        },
        "settings": SETTINGS,
        "steps": [],
        "attempts": [],
    }
    sampler = ProcessMemory(interval=0.05)
    sampler.start()
    previous_attempts = []
    try:
        problem, controls, metadata = prepare(args, args.subdivision)
        record = {
            **metadata,
            **record,
            "time_scheme": "backward_euler",
            "numerical_policy": "fresh_fixed_scaled_newton_original_equation_checks_v1",
            "scope": "Fresh single-policy fixed-control trajectory from the physical initial fields",
            "timing_boundary": "Step intervals include fresh equation checks. Process intervals additionally include input preparation, restart verification, field serialization and summary. Neither includes Python startup. Restarted attempts are distinct from uninterrupted repetitions.",
        }
        if args.resume:
            saved = json.loads((args.output / "record.json").read_text())
            # Never overwrite a manifest when restart validation fails.
            state, flow = restore(problem, controls, args.output, saved, record)
            record = saved
        else:
            state, flow = initial_fields(problem)
        previous_attempts = record["attempts"]
        record["attempts"] = [
            *previous_attempts,
            {"started_unix": time.time(), "preparation_seconds": time.perf_counter() - started},
        ]
        write_report(args.output / "record.json", record)
        for n in range(len(record["steps"]), problem.slabs):
            write_report(
                args.output / "progress.json",
                {"slab": n, "time_s": float(problem.physical_steps[: n + 1].sum())},
            )
            source = np.zeros(len(problem.mesh.nodes))
            source[problem.free] = controls[n]
            step_started = time.perf_counter()
            row, arrays = advance(problem, source, state, flow, n)
            row["nonlinear_solve_seconds"] = row["seconds"]
            row["seconds"] = time.perf_counter() - step_started
            record = commit_step(args.output, record, row, arrays)
            if not row["verified"]:
                record["status"] = "numerical_failure"
                break
            state, flow = checked_field(args.output, record["steps"][-1], problem)
        else:
            record["quantities"] = quantities(problem, args.output, record)
            record["status"] = "converged"
    except Exception as error:
        # Failed validation leaves an existing restart manifest unchanged.
        if args.resume and record["attempts"] == previous_attempts:
            raise
        record.update(status="execution_failure", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        memory = sampler.finish()
        if not args.resume or record["attempts"] != previous_attempts:
            if record["attempts"]:
                record["attempts"][-1].update(
                    finished_unix=time.time(),
                    process_seconds=time.perf_counter() - started,
                    memory=memory,
                )
            record["sum_step_seconds"] = sum(row["seconds"] for row in record["steps"])
            record["all_attempt_process_seconds"] = (
                sum(row["process_seconds"] for row in record["attempts"])
                if record["attempts"]
                and all("process_seconds" in row for row in record["attempts"])
                else None
            )
            write_report(args.output / "record.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "optimization", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fine-baseline", type=Path)
    parser.add_argument("--target-position", type=int)
    parser.add_argument("--subdivision", type=int, default=1)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.subdivision < 1 or args.threads < 1 or args.repetition < 0:
        parser.error("Subdivision and threads must be positive; repetition must be nonnegative")
    args.policy, args.family = "newton", "forward"
    with threadpool_limits(args.threads):
        result = run(args)
    return 0 if result["status"] == "converged" else 2


if __name__ == "__main__":
    raise SystemExit(main())
