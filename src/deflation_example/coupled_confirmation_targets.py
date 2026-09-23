"""Choose confirmation targets from desired fields, without solver outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .coupled_optimize import load_problem
from .coupled_targets import desired_temperature
from .reporting import file_sha256, write_report
from .validation import integer


def choose_targets(rows, nominal):
    nominal = integer(nominal, "Nominal target", 0)
    identifiers = [integer(r["target"], "Target", 0) for r in rows]
    if len(rows) < 3 or len(set(identifiers)) != len(rows) or nominal not in identifiers:
        raise ValueError("Three distinct targets including the nominal case are required")
    values = np.asarray([[r["peak_K"], r["weighted_fraction_above_bound"]] for r in rows])
    if not np.isfinite(values).all() or np.any(values[:, 1] < 0) or np.any(values[:, 1] > 1):
        raise ValueError("Target measures must be finite and fractions in [0, 1]")
    chosen = [nominal]
    for measure in ("peak_K", "weighted_fraction_above_bound"):
        remaining = [r for r in rows if r["target"] not in chosen]
        chosen.append(min(remaining, key=lambda r: (-r[measure], r["target"]))["target"])
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.optimization / "record.json"
    original = json.loads(source.read_text())
    if not original.get("all_problems_verified") or len(original["configuration"]["queries"]) != 1:
        raise ValueError("A verified nominal single-target configuration is required")
    cfg = {**original["configuration"], "baseline_directory": str(args.baseline)}
    problem, baseline = load_problem(cfg)
    if baseline["baseline_sha256"] != original["baseline_sha256"]:
        raise ValueError("The declared computed-flow baseline differs")
    nominal = cfg["queries"][0]["target"]
    upper = cfg["queries"][0]["upper_K"]
    bound = (upper - problem.temperature_offset) / problem.temperature_scale
    rows = []
    for target in range(cfg["target_count"]):
        field = desired_temperature(
            problem, target, cfg["target_count"], cfg.get("target_startup_s", 0)
        )
        rows.append(
            {
                "target": target,
                "peak_K": float(
                    problem.temperature_offset + problem.temperature_scale * field.max()
                ),
                "weighted_fraction_above_bound": float(
                    problem.weights @ (field > bound) / problem.weights.sum()
                ),
                "desired_sha256": hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest(),
            }
        )
    chosen = choose_targets(rows, nominal)
    if len({r["desired_sha256"] for r in rows if r["target"] in chosen}) != 3:
        raise ValueError("The chosen target fields must be distinct")
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(
        args.output / "targets.json",
        {
            "schema": "coupled-confirmation-targets-v1",
            "targets": chosen,
            "upper_K": upper,
            "lower_K": cfg["lower_K"],
            "baseline_sha256": baseline["baseline_sha256"],
            "nominal_record_sha256": file_sha256(source),
            "rows": rows,
            "rule": "Nominal, highest remaining desired peak, then largest remaining space-time weighted fraction of desired temperature above the physical bound; ties use the smallest target identifier.",
            "scope": "Selection uses desired fields only. The weighted fraction describes the target, not the optimized active set.",
        },
    )


if __name__ == "__main__":
    main()
