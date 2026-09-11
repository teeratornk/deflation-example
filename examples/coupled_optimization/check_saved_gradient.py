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
                "equations": problem.verify(evaluation),
            },
        )


if __name__ == "__main__":
    main()
