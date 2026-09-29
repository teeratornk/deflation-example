"""Independent directional and adjoint checks at a saved trust-region state."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_optimize import adjoint_acceptance, equation_acceptance, load_problem
from .coupled_recovery import RecoveryStore, identity
from .coupled_sequence import RestoredEvaluation
from .coupled_targets import desired_temperature
from .reporting import environment, write_report


def check(problem, state, flow_guess, desired, *, seed=20260925, bounds=None):
    start = time.perf_counter()
    evaluation = problem.evaluate(state, initial=flow_guess)
    value, gradient = problem.objective_gradient(evaluation, desired)
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(problem.size)
    direction /= np.linalg.norm(direction)
    test = rng.standard_normal(problem.size)
    exact = evaluation.jacobian @ direction
    transpose = evaluation.jacobian.T @ test
    left, right = float(test @ exact), float(direction @ transpose)
    dot_error = abs(left - right) / max(np.linalg.norm(test) * np.linalg.norm(exact), 1e-30)
    rows = []
    for h in (1e-4, 1e-5, 1e-6):
        plus = problem.evaluate(state + h * direction, initial=evaluation)
        minus = problem.evaluate(state - h * direction, initial=evaluation)
        numerical = (plus.control - minus.control) / (2 * h)
        slope = problem.objective_difference(plus, minus, desired) / (2 * h)
        exact_slope = float(gradient @ direction)
        rows.append(
            {
                "step": h,
                "control_jacobian_relative_error": float(
                    np.linalg.norm(numerical - exact)
                    / max(np.linalg.norm(numerical), np.linalg.norm(exact), 1e-30)
                ),
                "objective_directional_relative_error": abs(slope - exact_slope)
                / max(abs(slope), abs(exact_slope), 1e-14),
            }
        )
    result = {
        "objective": value,
        "rows": rows,
        "transpose_relative_error": dot_error,
        "adjoint": problem.verify_adjoint(evaluation, desired),
        "equations": problem.verify(evaluation),
        "seconds": time.perf_counter() - start,
        "derivatives_passed": dot_error <= 1e-10
        and all(
            min(r[key] for r in rows) <= 1e-5
            for key in ("control_jacobian_relative_error", "objective_directional_relative_error")
        ),
    }
    if bounds is not None:
        from .coupled_trust import optimality

        kkt, scale = optimality(problem, evaluation, desired, gradient, *bounds)
        result.update(
            kkt=kkt, stationarity_scale=scale, stationarity_numerator=kkt["stationarity"] * scale
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing diagnostic output")
    record = json.loads(args.record.read_text())
    cfg, env = record["configuration"], environment()
    fp = identity(cfg, env["source_sha256"])
    payload = RecoveryStore(args.record.parent / "recovery", fp).load()
    if payload is None:
        raise ValueError("A complete checkpoint is required")
    saved = payload["optimizer"] or payload["previous_solution"]
    if saved is None:
        raise ValueError("The checkpoint has no retained temperature")
    position = payload["position"] - (payload["optimizer"] is None)
    if not 0 <= position < len(cfg["queries"]):
        raise ValueError("Checkpoint state has no corresponding target")
    with threadpool_limits(cfg["threads"]):
        problem, _ = load_problem(cfg)
        guess = RestoredEvaluation(saved["state"], saved["velocity"], saved["pressure"])
        desired = desired_temperature(
            problem,
            cfg["queries"][position]["target"],
            cfg["target_count"],
            cfg["target_startup_s"],
        )
        result = check(problem, saved["state"], guess, desired)
    result.update(
        environment=env,
        checkpoint_identity=fp,
        scope="Directional and adjoint verification at one retained temperature; separate diagnostic cost.",
    )
    result["passed"] = (
        result["derivatives_passed"]
        and adjoint_acceptance(result["adjoint"])
        and equation_acceptance(result["equations"], cfg)
    )
    write_report(args.output, result)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
