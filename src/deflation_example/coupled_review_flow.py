"""Reproduce saved outer directions and retain failing momentum systems."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from . import coupled_control
from .coupled_bounds import temperature_bounds
from .coupled_optimize import load_problem
from .coupled_review import read_snapshot, trial_report
from .coupled_sequence import RestoredEvaluation
from .coupled_targets import desired_temperature
from .flow_conservation import mass_diagnostics
from .reporting import environment, file_sha256, write_arrays, write_report
from .validation import integer


def momentum_diagnostics(
    flow, acceleration, fixed, values, result, *, previous=None, time_step=None, pressure_gauge=None
):
    """Check the actual failed momentum state, its Jacobian and Newton model."""
    x = np.r_[result.velocity[:, 0], result.velocity[:, 1], result.pressure]
    constrained = np.r_[fixed, flow.nv + fixed]
    if pressure_gauge is not None:
        constrained = np.r_[constrained, 2 * flow.nv + pressure_gauge[0]]
    free = np.setdiff1d(np.arange(flow.size), constrained)
    force = flow.load(acceleration)
    if time_step is not None:
        force += flow.mass @ previous / time_step
    rhs = np.r_[force[:, 0], force[:, 1], np.zeros(flow.np)]

    def residual(state):
        velocity = np.column_stack((state[: flow.nv], state[flow.nv : 2 * flow.nv]))
        return (flow.operator(velocity, time_step=time_step) @ state - rhs)[free]

    J = flow.operator(result.velocity, time_step=time_step) + flow.convection_derivative(
        result.velocity
    )
    J = J[free][:, free].tocsc()
    direction = np.zeros_like(x)
    direction[free] = np.random.default_rng(191).normal(size=len(free))
    direction /= np.linalg.norm(direction)
    action = J @ direction[free]
    difference = []
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        centered = (residual(x + h * direction) - residual(x - h * direction)) / (2 * h)
        difference.append(
            {
                "step": h,
                "jacobian_relative_error": float(
                    np.linalg.norm(centered - action) / max(np.linalg.norm(action), 1e-30)
                ),
            }
        )
    report = {
        "status": result.status,
        "flow_dofs": flow.size,
        "free_flow_dofs": len(free),
        "metrics": flow.verify(
            result,
            acceleration,
            fixed,
            values,
            previous=previous,
            time_step=time_step,
            pressure_gauge=pressure_gauge,
        ),
        "mass": mass_diagnostics(flow, result.velocity),
        "jacobian_differences": difference,
    }
    r = residual(x)
    try:
        factor = splu(J)
        p = factor.solve(-r)
        step = np.zeros_like(x)
        step[free] = p
        linear = J @ p
        velocity_step = np.column_stack((step[: flow.nv], step[flow.nv : 2 * flow.nv]))
        nonlinear = flow.nonlinear_force(velocity_step)
        quadratic = np.r_[nonlinear[:, 0], nonlinear[:, 1], np.zeros(flow.np)][free]
        norm = max(np.linalg.norm(r), 1e-30)
        report["newton"] = {
            "status": "solved",
            "correction_norm": float(np.linalg.norm(p)),
            "linear_relative_residual": float(np.linalg.norm(linear + r) / norm),
            "normalized_merit_directional_derivative": float((r / norm) @ (linear / norm)),
            "trials": [],
        }
        for length in (1.0, 0.5, 0.25, 2.0**-12, 2.0**-23):
            actual = residual(x + length * step)
            predicted = r + length * linear + length**2 * quadratic
            report["newton"]["trials"].append(
                {
                    "step": length,
                    "actual_residual_ratio": float(np.linalg.norm(actual) / norm),
                    "quadratic_model_residual_ratio": float(np.linalg.norm(predicted) / norm),
                    "quadratic_identity_relative_defect": float(
                        np.linalg.norm(actual - predicted) / max(norm, np.linalg.norm(actual))
                    ),
                }
            )
    except (RuntimeError, np.linalg.LinAlgError):
        report["newton"] = {"status": "factorization_failed"}
    return report


@contextmanager
def capture_failures(directory, reports):
    """Instrument one isolated worker without altering any solver return value."""
    original = coupled_control.solve_momentum

    def traced(flow, acceleration, fixed, values, **kwargs):
        result = original(flow, acceleration, fixed, values, **kwargs)
        if result.status != "converged":
            index = len(reports)
            filename = f"failed-flow-{index}.npz"
            initial = kwargs["initial"]
            previous = kwargs.get("previous")
            write_arrays(
                directory / filename,
                acceleration=acceleration,
                boundary_indices=fixed,
                boundary_values=values,
                velocity=result.velocity,
                pressure=result.pressure,
                initial_velocity=initial.velocity,
                initial_pressure=initial.pressure,
                previous_velocity=np.empty((0, 2)) if previous is None else previous,
            )
            selected = {key: kwargs.get(key) for key in ("previous", "time_step", "pressure_gauge")}
            try:
                report = momentum_diagnostics(flow, acceleration, fixed, values, result, **selected)
            except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as failure:
                report = {
                    "status": result.status,
                    "diagnostic_status": "diagnostic_failed",
                    "diagnostic_error_type": type(failure).__name__,
                }
            report.update(
                file=filename,
                sha256=file_sha256(directory / filename),
                history=result.history,
                time_step=kwargs.get("time_step"),
                pressure_gauge=kwargs.get("pressure_gauge"),
                solver_settings={
                    key: kwargs.get(key) for key in ("tolerance", "max_iterations", "continuation")
                },
            )
            reports.append(report)
            write_report(directory / "failed-systems.json", reports)
        return result

    coupled_control.solve_momentum = traced
    try:
        yield
    finally:
        coupled_control.solve_momentum = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--review", type=Path, required=True, help="Completed QP direction from coupled_review"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    parser.add_argument("--continuation", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if any(not np.isfinite(s) or not 0 < s <= 1 for s in args.steps):
        raise ValueError("Trial lengths must lie in (0,1]")
    record, meta, arrays, manifest = read_snapshot(args.snapshot)
    prior = json.loads((args.review / "review.json").read_text())
    if prior["snapshot"] != manifest or prior["qp"]["status"] != "converged":
        raise ValueError("The direction must belong to this snapshot and a converged QP")
    path = args.review / "direction.npz"
    digest = file_sha256(path)
    with np.load(path, allow_pickle=False) as stored:
        state, direction = stored["state"].copy(), stored["direction"].copy()
    if file_sha256(path) != digest:
        raise ValueError("The saved direction changed while reading")
    if direction.shape != state.shape or not np.isfinite([state, direction]).all():
        raise ValueError("The saved direction must be finite and match the state")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-failed-flow-review-v1",
        "status": "running",
        "snapshot": manifest,
        "direction_sha256": digest,
        "direction_origin": prior.get("direction_origin", {"policy": "newly-solved-checkpoint-QP"}),
        "environment": environment(),
        "trials": [],
        "failures": [],
    }
    write_report(args.output / "report.json", report)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        cfg = {**record["configuration"], "baseline_directory": str(args.baseline)}
        problem, baseline = load_problem(cfg)
        report["baseline_sha256"] = baseline["baseline_sha256"]
        initial = (
            None
            if prior["initial_state_policy"] == "original_zero_state"
            else RestoredEvaluation(arrays["state"], arrays["velocity"], arrays["pressure"])
        )
        evaluation = problem.evaluate(state, initial=initial)
        query = cfg["queries"][meta["position"]]
        desired = desired_temperature(
            problem, query["target"], cfg["target_count"], cfg["target_startup_s"]
        )
        bounds = temperature_bounds(cfg, upper_K=query["upper_K"])
        lower = (
            bounds["optimization_lower_K"] - problem.temperature_offset
        ) / problem.temperature_scale
        upper = (
            bounds["optimization_upper_K"] - problem.temperature_offset
        ) / problem.temperature_scale
        with capture_failures(args.output, report["failures"]):
            for step in args.steps:
                report["trials"].append(
                    trial_report(
                        problem,
                        evaluation,
                        desired,
                        direction,
                        step,
                        lower,
                        upper,
                        args.continuation,
                    )
                )
                write_report(args.output / "report.json", report)
    report["status"] = "review_complete"
    write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
