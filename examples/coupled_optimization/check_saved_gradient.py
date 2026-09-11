"""Recheck the exact reduced gradient at a saved nonlinear optimization iterate."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from deflation_example.axisymmetric_flow import FlowResult
from deflation_example.coupled_optimize import load_problem
from deflation_example.coupled_optimizer import box_kkt
from deflation_example.reporting import environment, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--method", default="jacobi")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(4):
        record = json.loads((args.run / "record.json").read_text())
        config = {
            **record["configuration"],
            "baseline_directory": str(args.baseline),
            "flow_tolerance": 1e-11,
        }
        problem, _ = load_problem(config)
        field = args.run / (args.method + "-fields.npz")
        with np.load(field, allow_pickle=False) as data:
            x, target, control = (data[name].copy() for name in ("state", "desired", "control"))
            flows = tuple(
                FlowResult(v.copy(), p.copy(), "initial", [])
                for v, p in zip(data["velocity"], data["pressure"], strict=True)
            )
        evaluation = problem.evaluate(x, initial=SimpleNamespace(flows=flows))
        value, gradient = problem.objective_gradient(evaluation, target)
        rng = np.random.default_rng(762)
        direction = rng.normal(size=problem.size)
        direction /= np.linalg.norm(direction)
        rows = []
        for h in (0.01, 0.005, 0.0025, 0.00125):
            trial = problem.evaluate(x + h * direction, initial=evaluation)
            f, _ = problem.objective_gradient(trial, target)
            rows.append({"step": h, "remainder": abs(f - value - h * (gradient @ direction))})
        remainder = np.array([row["remainder"] for row in rows])
        lower = (config["lower_K"] - problem.temperature_offset) / problem.temperature_scale
        upper = (config["upper_K"] - problem.temperature_offset) / problem.temperature_scale
        scaled = gradient / problem.weights
        projected = x - np.clip(x - scaled, lower, upper)
        worst = int(np.argmax(np.abs(projected)))
        coordinate = np.zeros(problem.size)
        coordinate[worst] = 1
        local = []
        for h in (1e-5, 1e-6, 1e-7, 1e-8):
            plus = problem.evaluate(x + h * coordinate, initial=evaluation)
            minus = problem.evaluate(x - h * coordinate, initial=evaluation)
            difference = problem.objective_difference(plus, minus, target)
            fplus = problem.objective_gradient(plus, target)[0]
            fminus = problem.objective_gradient(minus, target)[0]
            local.append(
                {
                    "step": h,
                    "derivative_analytic": float(gradient[worst]),
                    "derivative_difference_formula": difference / (2 * h),
                    "derivative_subtracted_totals": (fplus - fminus) / (2 * h),
                }
            )
        write_report(
            args.output,
            {
                "environment": environment(),
                "field_sha256": hashlib.sha256(field.read_bytes()).hexdigest(),
                "objective": value,
                "control_relative_difference": float(
                    np.linalg.norm(control - evaluation.control)
                    / max(np.linalg.norm(control), 1e-30)
                ),
                "gradient_taylor": rows,
                "orders": np.log2(remainder[:-1] / remainder[1:]).tolist(),
                "kkt_at_tighter_flow_tolerance": box_kkt(
                    x,
                    scaled,
                    lower,
                    upper,
                    max(1.0, np.max(np.abs(x - target)), np.max(np.abs(scaled - (x - target)))),
                ),
                "worst_projected_gradient_index": worst,
                "worst_coordinate_weight": float(problem.weights[worst]),
                "coordinate_gradient_checks": local,
                "equations": problem.verify(evaluation),
            },
        )


if __name__ == "__main__":
    main()
