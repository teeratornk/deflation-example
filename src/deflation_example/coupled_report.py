"""Summarize matched complete coupled sequences, retaining every outcome."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_optimize import equations_verified
from .coupled_saved import file_digest
from .reporting import write_report
from .validation import integer

METHODS = ("jacobi", "reference", "recycling")
LABELS = {"jacobi": "Diagonal CG", "reference": "Reference deflation", "recycling": "Recycling CG"}


def matched_identity(record):
    config = {
        key: value
        for key, value in record["configuration"].items()
        if key not in {"method", "repetition"}
    }
    env = record["environment"]
    return {
        "configuration": config,
        "numerical_policy": record["numerical_policy"],
        "baseline_sha256": record["baseline_sha256"],
        "source_sha256": env["source_sha256"],
        "cpu_model": env["cpu_model"],
        "numpy": env["numpy"],
        "scipy": env["scipy"],
        "device": record["device"],
        "timing_boundary": record["timing_boundary"],
    }


def validate_record(record):
    if record.get("schema") != "coupled-complete-sequence-v1":
        raise ValueError("Use complete-sequence records; pilot intervals have different boundaries")
    cfg = record["configuration"]
    if cfg["method"] not in METHODS:
        raise ValueError("Unknown coupled solver")
    total = record["sequence_seconds"]
    components = np.asarray(list(record["components_seconds"].values()), dtype=float)
    if (
        not np.isfinite(total)
        or total <= 0
        or not np.isfinite(components).all()
        or np.any(components < -1e-8)
    ):
        raise ValueError("Timing values must be finite and nonnegative")
    if not np.isclose(components.sum(), total, rtol=1e-10, atol=1e-8):
        raise ValueError("Timing components do not sum to the complete sequence")
    preparation = np.asarray([record["process_preparation_seconds"], record["calibration_seconds"]])
    if not np.isfinite(preparation).all() or np.any(preparation < 0):
        raise ValueError("Preparation costs must be finite and nonnegative")
    expected = total + preparation.sum()
    if not np.isclose(record["preparation_inclusive_seconds"], expected, rtol=1e-10, atol=1e-8):
        raise ValueError("Preparation-inclusive timing is inconsistent")
    cases = record["cases"]
    if [row["position"] for row in cases] != list(range(len(cfg["queries"]))):
        raise ValueError("Every declared target must have one outcome in sequence order")
    for query, row in zip(cfg["queries"], cases, strict=True):
        if row["target"] != query["target"] or row["upper_K"] != query["upper_K"]:
            raise ValueError("Target or bound differs from the declared sequence")
        if row["verified"]:
            kkt = np.asarray(list(row["kkt"].values()))
            adjoint = row["adjoint"]["maximum_momentum_adjoint_relative_residual"]
            if (
                row["status"] != "converged"
                or not np.isfinite(kkt).all()
                or kkt.max() > cfg["nonlinear_tolerance"]
                or not equations_verified(row["equations"])
                or not np.isfinite(adjoint)
                or adjoint > 1e-8
            ):
                raise ValueError("A verified label disagrees with its independent accuracy checks")
    verified = sum(row["verified"] for row in cases)
    whole = record["status"] == "complete" and verified == len(cases)
    if record["verified_problems"] != verified or record["all_problems_verified"] != whole:
        raise ValueError("Sequence and case verification labels disagree")
    return whole


def summarize(records, repetitions=5):
    repetitions = integer(repetitions, "Declared repetitions", 1)
    if not records:
        raise ValueError("Supply at least one complete-sequence record")
    identity = matched_identity(records[0])
    grouped = {method: [] for method in METHODS}
    seen = set()
    for record in records:
        validate_record(record)
        if matched_identity(record) != identity:
            raise ValueError(
                "Comparisons require matched sources, hardware, accuracy and problem settings"
            )
        cfg = record["configuration"]
        repetition = integer(cfg["repetition"], "Repetition", 0)
        key = cfg["method"], repetition
        if repetition >= repetitions or key in seen:
            raise ValueError(
                "Repetition identifiers must be distinct and within the declared population"
            )
        seen.add(key)
        grouped[cfg["method"]].append(record)
    rows = []
    for method, population in grouped.items():
        passing = [r for r in population if r["all_problems_verified"]]
        times = [r["sequence_seconds"] for r in passing]
        row = {
            "method": method,
            "declared_sequences": repetitions,
            "recorded_sequences": len(population),
            "verified_sequences": len(passing),
            "sequence_seconds_all_outcomes": [r["sequence_seconds"] for r in population],
            "outcomes": [
                {
                    "repetition": r["configuration"]["repetition"],
                    "sequence_status": r["status"],
                    "target_statuses": [c["status"] for c in r["cases"]],
                }
                for r in population
            ],
            "verified_sequence_median_seconds": float(np.median(times)) if times else None,
            "verified_sequence_range_seconds": [min(times), max(times)] if times else None,
            "preparation_inclusive_median_seconds": float(
                np.median([r["preparation_inclusive_seconds"] for r in passing])
            )
            if passing
            else None,
            "sampled_peak_host_rss_bytes": max(
                (
                    r["memory"].get("peak_host_rss_bytes", 0)
                    for r in passing
                    if r["memory"].get("complete")
                ),
                default=None,
            ),
            "sampled_peak_gpu_process_bytes": max(
                (
                    r["memory"].get("peak_gpu_process_bytes", 0)
                    for r in passing
                    if r["memory"].get("complete")
                ),
                default=None,
            ),
            "verified_memory_sequences": sum(bool(r["memory"].get("complete")) for r in passing),
        }
        rows.append(row)
    complete = all(row["verified_sequences"] == repetitions for row in rows)
    speedup = None
    if complete:
        medians = {row["method"]: row["verified_sequence_median_seconds"] for row in rows}
        speedup = min(medians["jacobi"], medians["recycling"]) / medians["reference"]
    return {
        "schema": "coupled-comparison-summary-v1",
        "matched_protocol": identity,
        "all_declared_sequences_verified": complete,
        "fastest_tested_alternative_over_reference": speedup,
        "scope": "Median complete-sequence timings for this fixed discrete coupled protocol. Physical resolution requires separate fixed-source studies. Failed and missing sequences remain visible; a complete-population ratio requires every declared sequence to meet the final checks.",
        "methods": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [root / "record.json" for root in args.runs]
    records = [json.loads(path.read_text()) for path in paths]
    report = summarize(records, args.repetitions)
    report["input_records"] = [
        {"run": path.parent.name, "sha256": file_digest(path)} for path in paths
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", report)
    line_end = chr(92) * 2
    lines = [
        "Method & Verified sequences & Median complete time (s) & Observed range (s) " + line_end
    ]
    for row in report["methods"]:
        median, span = (
            row["verified_sequence_median_seconds"],
            row["verified_sequence_range_seconds"],
        )
        timing = "--- & ---" if median is None else f"{median:.2f} & {span[0]:.2f}--{span[1]:.2f}"
        lines.append(
            f"{LABELS[row['method']]} & {row['verified_sequences']}/{row['declared_sequences']} & {timing} "
            + line_end
        )
    (args.output / "timing_rows.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
