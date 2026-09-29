"""Check saved temperatures, source fields, objectives and paired agreement."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_corrected_report import field_agreement
from .coupled_optimize import load_problem
from .coupled_projected_report import audit
from .coupled_small_study import verification_identity
from .coupled_targets import desired_temperature
from .reporting import file_sha256, write_report


def field_metrics(problem, fields, desired, lower_K, upper_K, objective):
    field_agreement(fields, fields, temperature_scale=problem.temperature_scale)
    if fields["state"].shape != (problem.size,) or not np.array_equal(fields["desired"], desired):
        raise ValueError("The saved field dimension or desired trajectory differs")
    temperature = problem.temperature_offset + problem.temperature_scale * fields["state"]
    reconstructed = float(
        0.5
        * problem.objective_scale
        * np.sum(
            problem.weights
            * ((fields["state"] - desired) ** 2 + problem.alpha * fields["control"] ** 2)
        )
    )
    relative = abs(reconstructed - objective) / max(abs(reconstructed), abs(objective), 1e-30)
    return {
        "objective_from_fields": reconstructed,
        "objective_relative_difference": relative,
        "objective_consistent": bool(np.isfinite(relative) and relative <= 1e-10),
        "maximum_upper_violation_K": float(max(0, np.max(temperature - upper_K))),
        "maximum_lower_violation_K": float(max(0, np.max(lower_K - temperature))),
        "minimum_temperature_K": float(np.min(temperature)),
        "maximum_temperature_K": float(np.max(temperature)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-confirmation-fields-v1",
        "runs": [],
        "comparisons": [],
        "scope": "Read-only saved-field, objective and declared discrete bound checks. No fresh coupled PDE solve, physical-resolution assessment or global-optimality certificate.",
    }
    stored, metadata, scale, population = {}, {}, None, None
    for path in args.records:
        record = json.loads(path.read_text()) if path.exists() else None
        row = {"input": path.parent.name, **audit(record), "fields": []}
        report["runs"].append(row)
        if record is None:
            continue
        cfg = record["configuration"]
        identity = (
            verification_identity(cfg),
            cfg["queries"],
            record.get("baseline_sha256"),
            record["environment"]["source_sha256"],
        )
        if population is not None and identity != population:
            raise ValueError("The saved fields belong to different numerical populations")
        population = identity
        row["record_sha256"] = file_sha256(path)
        with threadpool_limits(cfg["threads"]):
            problem, baseline = load_problem(
                {**cfg, "baseline_directory": str(args.baseline), "device": "cpu"}
            )
            if baseline["baseline_sha256"] != record["baseline_sha256"]:
                raise ValueError("The physical baseline differs from the saved fields")
            scale = problem.temperature_scale
            method = "baseline" if cfg["rank"] == 0 else cfg["method"]
            for case in record.get("cases", []):
                position = case["position"]
                key = (cfg["repetition"], method, position)
                field_path = path.parent / f"target-{position:02d}.npz"
                result = {
                    "position": position,
                    "target": case["target"],
                    "status": case["status"],
                    "optimization_verified": case["verified"],
                }
                row["fields"].append(result)
                if not field_path.exists():
                    result["field_status"] = "missing"
                    continue
                with np.load(field_path, allow_pickle=False) as data:
                    fields = {name: data[name].copy() for name in ("state", "control", "desired")}
                desired = desired_temperature(
                    problem, case["target"], cfg["target_count"], cfg["target_startup_s"]
                )
                result.update(
                    field_metrics(
                        problem,
                        fields,
                        desired,
                        cfg["lower_K"],
                        cfg["queries"][position]["upper_K"],
                        case["objective"],
                    )
                )
                result.update(field_status="checked", field_sha256=file_sha256(field_path))
                if key in stored:
                    raise ValueError("Duplicate method, repetition and target slot")
                stored[key], metadata[key] = fields, result
                write_report(args.output / "summary.json", report)
    for (repetition, method, position), fields in stored.items():
        baseline_key = (repetition, "baseline", position)
        if method == "baseline" or baseline_key not in stored:
            continue
        delta = field_agreement(stored[baseline_key], fields, temperature_scale=scale)
        ref_value, value = (
            metadata[k]["objective_from_fields"]
            for k in (baseline_key, (repetition, method, position))
        )
        relative = abs(ref_value - value) / max(abs(ref_value), abs(value), 1e-30)
        report["comparisons"].append(
            {
                "repetition": repetition,
                "method": method,
                "position": position,
                "both_optimization_verified": all(
                    metadata[k]["optimization_verified"]
                    for k in (baseline_key, (repetition, method, position))
                ),
                "maximum_temperature_difference_K": delta,
                "relative_objective_difference": relative,
                "agreement_within_declared_thresholds": delta <= 0.001 and relative <= 1e-6,
            }
        )
    write_report(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
