"""Directional and transpose checks at a frozen coupled checkpoint."""

import argparse
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_control import FlowEvaluationError
from .coupled_derivatives import StabilizationBranchError
from .coupled_optimize import load_problem
from .coupled_review import read_snapshot
from .coupled_sequence import RestoredEvaluation
from .coupled_targets import desired_temperature
from .reporting import environment, write_report
from .validation import integer


def direction_checks(problem, evaluation, desired, direction, progress=None):
    """Center differences of source and objective; keep failed perturbations visible.

    Objective differences use the exact difference of squares to avoid subtracting
    two large totals near a stationary point. Perturbations are derivative tests,
    not constrained optimization updates, and are not clipped to the bounds.
    """
    direction = np.asarray(direction, dtype=float)
    if (
        direction.shape != (problem.size,)
        or not np.isfinite(direction).all()
        or not np.any(direction)
    ):
        raise ValueError("A finite nonzero direction must match the state")
    direction = direction / np.linalg.norm(direction)
    _, gradient = problem.objective_gradient(evaluation, desired)
    tangent = evaluation.jacobian @ direction
    dual = np.random.default_rng(901).normal(size=problem.size)
    dual /= np.linalg.norm(dual)
    lhs, rhs = float(dual @ tangent), float(direction @ (evaluation.jacobian.T @ dual))
    report = {
        "tangent_transpose_relative_difference": abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30),
        "analytic_directional_derivative": float(gradient @ direction),
        "rows": [],
    }
    for step in (1e-2, 1e-3, 1e-4, 1e-5):
        tick = time.perf_counter()
        try:
            plus = problem.evaluate(evaluation.state + step * direction, initial=evaluation)
            minus = problem.evaluate(evaluation.state - step * direction, initial=evaluation)
            numerical = (plus.control - minus.control) / (2 * step)
            forward = problem.objective_difference(plus, evaluation, desired)
            backward = problem.objective_difference(minus, evaluation, desired)
            derivative = (forward - backward) / (2 * step)
            analytic = report["analytic_directional_derivative"]
            row = {
                "step": step,
                "status": "evaluated",
                "seconds": time.perf_counter() - tick,
                "control_derivative_relative_error": float(
                    np.linalg.norm(numerical - tangent) / max(np.linalg.norm(tangent), 1e-30)
                ),
                "objective_derivative": derivative,
                "objective_derivative_absolute_error": abs(derivative - analytic),
                "objective_derivative_relative_error": abs(derivative - analytic)
                / max(abs(analytic), 1e-30),
                "maximum_temperature_perturbation_K": float(
                    step * np.abs(direction).max() * problem.temperature_scale
                ),
            }
            del plus, minus
        except FlowEvaluationError as failure:
            row = {
                "step": step,
                "status": "flow_" + failure.result.status,
                "failed_slab": failure.slab,
                "metrics": failure.metrics,
                "flow_history": failure.result.history,
                "seconds": time.perf_counter() - tick,
            }
        except StabilizationBranchError:
            row = {
                "step": step,
                "status": "stabilization_branch_switch",
                "seconds": time.perf_counter() - tick,
            }
        report["rows"].append(row)
        if progress is not None:
            progress(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--direction", choices=("random", "worst_stationarity"), default="worst_stationarity"
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    record, meta, arrays, manifest = read_snapshot(args.snapshot)
    report = {
        "schema": "coupled-checkpoint-derivatives-v1",
        "status": "running",
        "snapshot": manifest,
        "environment": environment(),
        "direction_policy": args.direction,
        "scope": "Derivative diagnostics at a saved iterate; no convergence or timing claim.",
    }
    write_report(args.output / "report.json", report)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        cfg = {**record["configuration"], "baseline_directory": str(args.baseline)}
        problem, baseline = load_problem(cfg)
        report["baseline_sha256"] = baseline["baseline_sha256"]
        problem.evaluation_callback = lambda row: write_report(
            args.output / "evaluation-progress.json", row
        )
        evaluation = problem.evaluate(
            arrays["state"],
            initial=RestoredEvaluation(arrays["state"], arrays["velocity"], arrays["pressure"]),
        )
        query = cfg["queries"][meta["position"]]
        desired = desired_temperature(
            problem, query["target"], cfg["target_count"], cfg["target_startup_s"]
        )
        if args.direction == "random":
            direction = np.random.default_rng(900).normal(size=problem.size)
        else:
            from .coupled_review import optimality, stationarity_locations
            from .coupled_bounds import temperature_bounds

            bounds = temperature_bounds(cfg, upper_K=query["upper_K"])
            lower = (
                bounds["optimization_lower_K"] - problem.temperature_offset
            ) / problem.temperature_scale
            upper = (
                bounds["optimization_upper_K"] - problem.temperature_offset
            ) / problem.temperature_scale
            _, gradient = problem.objective_gradient(evaluation, desired)
            _, scale = optimality(problem, evaluation.state, gradient, desired, lower, upper)
            location = stationarity_locations(
                problem, evaluation.state, gradient, lower, upper, scale, count=1
            )[0]
            report["location"] = location
            direction = np.zeros(problem.size)
            direction[location["index"]] = 1

        def progress(checks):
            report["checks"] = checks
            write_report(args.output / "report.json", report)

        report["checks"] = direction_checks(problem, evaluation, desired, direction, progress)
        report["status"] = "review_complete"
        write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
