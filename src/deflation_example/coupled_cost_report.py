"""Cost diagnosis and complete-sequence confirmation without selective pooling."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import pstats
import statistics

from .coupled_small_report import summarize
from .reporting import file_sha256, write_report


def profile_group(filename, function):
    name = Path(filename).name
    if "SuperLU" in function or name in {"linsolve.py", "_dsolve.py"}:
        return "sparse_factorization_and_solve"
    if any(s in name for s in ("coarse", "recycling")):
        return "coarse_space_and_recycling_python"
    if "reference" in name or "krylov" in name:
        return "reference_construction_python"
    if name in {"reporting.py", "coupled_trace.py", "zipfile.py", "_npyio_impl.py"}:
        return "serialization_python"
    if "flow" in name or name in {"coupled_control.py", "coupled_derivatives.py"}:
        return "flow_and_control_python"
    if name in {"solvers.py", "refinement.py", "study_solvers.py", "box_projected_cg.py"}:
        return "inner_solver_python"
    if name in {"coupled_trust.py", "coupled_trust_run.py"}:
        return "outer_optimizer_python"
    return "other_python_and_native_calls"


def profile_summary(stats):
    """Exclusive self times sum; cumulative call times are nested diagnostics."""
    groups, functions = defaultdict(float), []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in stats.items():
        if not all(math.isfinite(v) and v >= 0 for v in (own, cumulative)):
            raise ValueError("Profile times must be finite and nonnegative")
        groups[profile_group(filename, function)] += own
        functions.append(
            {
                "file": Path(filename).name,
                "line": line,
                "function": function,
                "primitive_calls": primitive,
                "calls": calls,
                "self_seconds": own,
                "cumulative_seconds": cumulative,
            }
        )
    return {
        "exclusive_total_seconds": sum(groups.values()),
        "exclusive_categories_seconds": dict(groups),
        "largest_nested_calls": sorted(
            functions, key=lambda r: r["cumulative_seconds"], reverse=True
        )[:40],
        "scope": "Exclusive self times partition profiled execution, including native calls attributed by cProfile. Cumulative call times overlap and must not be added. Instrumented times are diagnostic, not performance measurements.",
    }


def method_key(row):
    if row.get("rank") == 0 and row.get("variant") == "standard":
        return "baseline"
    if row.get("rank") == 8 and row.get("variant") == "standard":
        return "reference"
    if row.get("rank") == 8 and row.get("variant") == "recycling":
        return "recycling"
    raise ValueError(
        "The confirmation requires rank zero, full rank eight and recycling rank eight"
    )


def confirmation(records, repetitions=5):
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("Use a positive integer repetition count")
    # The caller supplies three explicit slots per repetition, including None
    # for missing files. Unknown output cannot disappear from a population.
    if len(records) != 3 * repetitions:
        raise ValueError("Retain all three method slots for every declared repetition")
    report = summarize(records)
    slots, rows = {}, report["runs"]
    for record, row in zip(records, rows, strict=True):
        if record is None:
            continue
        method, repetition = method_key(row), row["repetition"]
        if repetition not in range(repetitions) or (method, repetition) in slots:
            raise ValueError("Duplicate or undeclared confirmation slot")
        slots[method, repetition] = row
        cfg = record["configuration"]
        if (
            cfg["queries"] != [{"target": t, "upper_K": 357.3} for t in (7, 8, 9)]
            or cfg["slabs"] != 16
            or cfg["alpha"] != 1e-14
            or cfg.get("capture_linear_systems", False)
            or record.get("resumed", False)
            or record.get("prior_attempt_seconds", 0) != 0
        ):
            raise ValueError("Use the frozen, uninterrupted nearby sequence without trace capture")
        row["inner_iterations"] = sum(
            step.get("linear_iterations", 0)
            for c in record.get("cases", [])
            for h in c.get("history", [])
            for a in h["attempts"]
            for step in a["qp_history"]
        )
        row["outer_updates"] = sum(len(c.get("history", [])) for c in record.get("cases", []))
        row["maximum_final_kkt"] = max(
            (c["maximum_kkt"] for c in row.get("cases", []) if c["maximum_kkt"] is not None),
            default=None,
        )
    methods = {}
    for method in ("baseline", "reference", "recycling"):
        values = [slots.get((method, r)) for r in range(repetitions)]
        complete = all(v is not None and v["verified"] for v in values)
        times = [v["cumulative_attempt_seconds"] for v in values if v is not None and v["verified"]]
        methods[method] = {
            "verified_sequences": len(times),
            "declared_sequences": repetitions,
            "statuses": ["missing" if v is None else v["status"] for v in values],
            "complete_seconds_by_repetition": [
                None if v is None else v.get("cumulative_attempt_seconds") for v in values
            ],
            "median_complete_seconds": statistics.median(times) if complete else None,
            "observed_range_seconds": [min(times), max(times)] if complete else None,
        }
    # Per-pair matching is delegated to the strict existing source/hardware/start
    # comparison; a complete but unmatched population also has no headline ratio.
    paired = all(
        slots.get((method, r), {}).get("speedup") is not None
        for method in ("reference", "recycling")
        for r in range(repetitions)
    )
    all_verified = all(v["verified_sequences"] == repetitions for v in methods.values())
    ratios = None
    if paired and all_verified:
        ref = methods["reference"]["median_complete_seconds"]
        ratios = {
            "baseline_over_reference": methods["baseline"]["median_complete_seconds"] / ref,
            "recycling_over_reference": methods["recycling"]["median_complete_seconds"] / ref,
            "fastest_tested_alternative_over_reference": min(
                methods[m]["median_complete_seconds"] for m in ("baseline", "recycling")
            )
            / ref,
        }
    report.update(
        schema="coupled-complete-confirmation-v1",
        methods=methods,
        all_sequences_verified=all_verified,
        all_comparisons_matched=paired,
        median_time_ratios=ratios,
        scope="Five independently timed repetitions of the same three-target sequence, not fifteen distinct physical cases. All expected slots and failures are retained. Ratios require every matched sequence to meet accuracy; a favorable ratio is not required for completion.",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("profile", "confirmation"))
    parser.add_argument("--profiles", type=Path, nargs="+")
    parser.add_argument("--records", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "profile":
        if not args.profiles:
            parser.error("Use --profiles for locally generated cProfile files")
        report = {"schema": "coupled-profile-summary-v1", "profiles": []}
        for path in args.profiles:
            stats = pstats.Stats(str(path))
            report["profiles"].append(
                {"input": path.name, "sha256": file_sha256(path), **profile_summary(stats.stats)}
            )
    else:
        if not args.records:
            parser.error("Supply every expected record, including missing outputs")
        records = [json.loads(p.read_text()) if p.exists() else None for p in args.records]
        report = confirmation(records)
        for row, path in zip(report["runs"], args.records, strict=True):
            row.update(
                input_name=path.parent.name,
                record_sha256=file_sha256(path) if path.exists() else None,
            )
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
