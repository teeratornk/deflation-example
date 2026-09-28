"""Compare zero and declared optimization initial states before reference construction."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_bounds import temperature_bounds
from .coupled_initial_state import snapshot_initial_guess
from .coupled_optimize import equation_acceptance, load_problem
from .reporting import environment, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.optimization / "record.json").read_text())["configuration"]
    cfg = {**cfg, "baseline_directory": str(args.baseline)}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-reference-verification-v1",
        "environment": environment(),
        "status": "running",
        "cases": [],
        "scope": "Initial-state equation checks only; no reference or optimization timing.",
    }
    write_report(args.output / "record.json", report)
    with threadpool_limits(cfg["threads"]):
        problem, baseline = load_problem(cfg)
        problem.evaluation_callback = lambda row: write_report(args.output / "progress.json", row)
        guess, metadata = snapshot_initial_guess(cfg, problem, baseline, 0)
        bounds = temperature_bounds(cfg, upper_K=cfg["queries"][0]["upper_K"])
        lower = (
            bounds["optimization_lower_K"] - problem.temperature_offset
        ) / problem.temperature_scale
        upper = (
            bounds["optimization_upper_K"] - problem.temperature_offset
        ) / problem.temperature_scale
        for name, state, initial in (
            ("clipped_zero", np.clip(np.zeros(problem.size), lower, upper), None),
            ("declared_optimizer_initial", guess.state, guess),
        ):
            row = {"policy": name, "status": "running"}
            report["cases"].append(row)
            write_report(args.output / "record.json", report)
            start = time.perf_counter()
            try:
                evaluation = problem.evaluate(state.copy(), initial=initial)
                checks = problem.verify(evaluation, local_mass=True)
                row.update(
                    status="verified"
                    if equation_acceptance(checks, cfg)
                    else "verification_failed",
                    equations=checks,
                    equation_tolerance=cfg.get("equation_acceptance_tolerance", 1e-8),
                    conservation_tolerance=cfg.get("conservation_tolerance", 1e-6),
                    initial_state=metadata if initial is not None else None,
                )
            except Exception as error:
                row.update(status="error", error_type=type(error).__name__, error=str(error))
            row["seconds"] = time.perf_counter() - start
            write_report(args.output / "record.json", report)
    report["status"] = "complete"
    write_report(args.output / "record.json", report)


if __name__ == "__main__":
    main()
