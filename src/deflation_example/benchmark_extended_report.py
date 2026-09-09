"""Validate complete-sequence records and export reproducible summaries."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from .benchmark_extended import METHODS
from .reporting import atomic_output, write_report
from .timing import PHASES


KKT_COMPONENTS = {"stationarity", "primal", "dual", "complementarity", "projected_gradient"}


def check_partition(parts, total):
    if not parts or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in parts.values()
    ):
        raise ValueError("Invalid timing component")
    if not math.isfinite(total) or not math.isclose(
        sum(parts.values()), total, rel_tol=1e-10, abs_tol=1e-10
    ):
        raise ValueError("Timing components do not sum to the total")


def checked_sequences(index_path):
    index_path = Path(index_path)
    report = json.loads(index_path.read_text())
    spec = report["specification"]
    controls = spec["controls"]
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if report["protocol_sha256"] != spec_hash:
        raise ValueError("Protocol hash differs")
    if not report.get("complete"):
        raise ValueError("Complete-sequence experiment is unfinished")
    if spec["protocol"] not in {
        "extended-cht-pdas-v1",
        "extended-cht-pdas-v2",
        "extended-cht-pdas-v3",
    }:
        raise ValueError("Unknown protocol")
    expected = {
        (n, w, m, r)
        for n in controls["grids"]
        for w in controls["warm_starts"]
        for m in METHODS
        for r in range(controls["repeats"])
    }
    seen, hashes, sequences = set(), {}, []
    for record in report["sequences"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "sequences":
            raise ValueError("Unsafe sequence path")
        path = index_path.parent / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Sequence file hash differs")
        sequence = json.loads(path.read_text())
        key = tuple(sequence[k] for k in ("n", "warm_start", "method", "repetition"))
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected sequence")
        seen.add(key)
        for field in ("n", "warm_start", "method", "repetition", "success", "total_seconds"):
            if sequence[field] != record[field]:
                raise ValueError("Index differs from its sequence")
        if sequence["protocol_sha256"] != spec_hash or sequence["bound"] != report["bound"]:
            raise ValueError("Sequence protocol or physical bound differs")
        check_partition(sequence["components_seconds"], sequence["total_seconds"])
        cases = sequence["cases"]
        if [c["index"] for c in cases] != list(range(controls["targets"])) and sequence[
            "error_type"
        ] is None:
            raise ValueError("Missing target without recorded sequence failure")
        previous_time = 0.0
        for case in cases:
            k = (sequence["n"], case["index"])
            if hashes.setdefault(k, case["target_sha256"]) != case["target_sha256"]:
                raise ValueError("Target changed between methods or repetitions")
            if not previous_time <= case["cumulative_seconds"] <= sequence["total_seconds"]:
                raise ValueError("Invalid cumulative sequence time")
            previous_time = case["cumulative_seconds"]
            if "history" in case and case["outer_iterations"] != len(case["history"]):
                raise ValueError("Outer count omits a visited inactive set")
            if case["inner_iterations"] != sum(i["iterations"] for i in case["inner"]):
                raise ValueError("Inner iteration sum differs")
            for inner in case["inner"]:
                if set(inner["components_seconds"]) != set(PHASES):
                    raise ValueError("Incomplete kernel timing partition")
                check_partition(inner["components_seconds"], inner["total_seconds"])
                if inner["status"] == "converged":
                    if (
                        not isinstance(inner["original_residual"], (int, float))
                        or not 0 <= inner["original_residual"] <= controls["rtol"]
                    ):
                        raise ValueError("Accepted inner solve fails original residual")
                if sequence["method"] == METHODS[0]:
                    if (
                        not 0
                        <= inner["rank"]
                        <= inner["orthogonalized_rank"]
                        <= inner["requested_rank"]
                        == controls["rank"]
                    ):
                        raise ValueError("Requested and deployed ranks disagree")
                    if inner["rank"] and not 1 <= inner["coarse_condition"] <= 1e10:
                        raise ValueError("Deployed coarse space violates its safeguard")
                elif inner["hierarchy_reused"] or not inner["resources_reused"]:
                    raise ValueError("AmgX resource policy differs")
            if case["status"] == "converged":
                if set(case["kkt"]) != KKT_COMPONENTS or any(
                    not isinstance(v, (int, float)) or not 0 <= v <= controls["outer_tolerance"]
                    for v in case["kkt"].values()
                ):
                    raise ValueError("Accepted target fails a KKT component")
                if any(i["status"] != "converged" for i in case["inner"]):
                    raise ValueError("Accepted target contains an unaccepted inner solve")
        expected_success = (
            len(cases) == controls["targets"]
            and sequence["error_type"] is None
            and all(c["status"] == "converged" for c in cases)
        )
        if sequence["success"] != expected_success:
            raise ValueError("Sequence success omits a failed target")
        if sequence["outer_iterations"] != sum(c["outer_iterations"] for c in cases) or sequence[
            "inner_iterations"
        ] != sum(c["inner_iterations"] for c in cases):
            raise ValueError("Sequence iteration counts differ")
        sequences.append(sequence)
    if seen != expected:
        raise ValueError("Missing complete-sequence repetition")
    for n in controls["grids"]:
        if len({h for (grid, _), h in hashes.items() if grid == n}) != controls["targets"]:
            raise ValueError("Duplicate target problems")
    return report, sequences


def summarize(report, sequences):
    controls = report["specification"]["controls"]
    common = sum(
        report[k]
        for k in (
            "calibration_seconds",
            "runtime_initialization_seconds",
            "runtime_finalization_seconds",
        )
    )
    rows = []
    for n in controls["grids"]:
        for warm in controls["warm_starts"]:
            for method in METHODS:
                runs = [
                    s
                    for s in sequences
                    if (s["n"], s["warm_start"], s["method"]) == (n, warm, method)
                ]
                chosen = sorted(runs, key=lambda s: s["total_seconds"])[len(runs) // 2]
                times = [s["total_seconds"] for s in runs]
                components = {
                    k: max(
                        (c["kkt"][k] for s in runs for c in s["cases"] if "kkt" in c), default=None
                    )
                    for k in KKT_COMPONENTS
                }
                rows.append(
                    {
                        "n": n,
                        "warm_start": warm,
                        "method": method,
                        "passed_sequences": sum(s["success"] for s in runs),
                        "repetitions": len(runs),
                        "passed_targets": sum(
                            c["status"] == "converged" for s in runs for c in s["cases"]
                        ),
                        "attempted_targets": sum(len(s["cases"]) for s in runs),
                        "median_seconds": median(times),
                        "min_seconds": min(times),
                        "max_seconds": max(times),
                        "with_common_seconds": common + median(times),
                        "outer_iterations": chosen["outer_iterations"],
                        "inner_iterations": chosen["inner_iterations"],
                        "kkt_maxima": components,
                        "median_run_components": chosen["components_seconds"],
                        "representative_repetition": chosen["repetition"],
                        "fallbacks": sum(
                            i["fallback_reason"] is not None
                            for s in runs
                            for c in s["cases"]
                            for i in c["inner"]
                        ),
                        "maximum_original_residual": max(
                            (
                                i["original_residual"]
                                for s in runs
                                for c in s["cases"]
                                for i in c["inner"]
                                if i["original_residual"] is not None
                            ),
                            default=None,
                        ),
                    }
                )
    return {"protocol": report["specification"]["protocol"], "common_seconds": common, "rows": rows}


def plot_sequences(report, sequences, output):
    import matplotlib.pyplot as plt

    controls = report["specification"]["controls"]
    grids, starts = controls["grids"], controls["warm_starts"]
    fig, axes = plt.subplots(
        len(starts), len(grids), figsize=(7.2, 2.1 * len(starts)), squeeze=False
    )
    for row, warm in enumerate(starts):
        for col, n in enumerate(grids):
            ax = axes[row, col]
            for method, label, color in zip(
                METHODS, ("Deflation", "AmgX / persistent resources"), ("#0072B2", "#D55E00")
            ):
                runs = [
                    s
                    for s in sequences
                    if (s["n"], s["warm_start"], s["method"]) == (n, warm, method)
                ]
                for s in runs:
                    values = [c["cumulative_seconds"] for c in s["cases"]]
                    if len(values) == controls["targets"]:
                        values[-1] = s["total_seconds"]
                    ax.plot(
                        np.arange(1, len(values) + 1),
                        values,
                        color=color,
                        alpha=0.3,
                        linestyle="-" if s["success"] else ":",
                        linewidth=0.7,
                    )
                complete = [s for s in runs if s["success"]]
                if len(complete) == len(runs):
                    values = np.array([[c["cumulative_seconds"] for c in s["cases"]] for s in runs])
                    values[:, -1] = [s["total_seconds"] for s in runs]
                    ax.plot(
                        np.arange(1, controls["targets"] + 1),
                        np.median(values, axis=0),
                        color=color,
                        label=label,
                    )
                else:
                    ax.plot(
                        [],
                        [],
                        color=color,
                        linestyle=":",
                        label=label + " / failed attempts included",
                    )
            ax.set_title(
                f"{n} × {n} × {n} / "
                + {
                    "cold": "Cold start",
                    "outer": "PDAS warm start",
                    "outer_inner": "PDAS and inner warm starts",
                }[warm],
                fontsize=8,
            )
            ax.set_xlabel("Target index")
            ax.set_ylabel("Cumulative elapsed time (s)")
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "complete_cht_cumulative.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report, sequences = checked_sequences(args.input)
    summary = summarize(report, sequences)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", summary)
    with atomic_output(args.output / "sequences.csv") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "n",
                "warm_start",
                "method",
                "repetition",
                "success",
                "total_seconds",
                "outer_iterations",
                "inner_iterations",
            ],
        )
        writer.writeheader()
        writer.writerows({k: s[k] for k in writer.fieldnames} for s in sequences)
    if args.plot:
        plot_sequences(report, sequences, args.output)


if __name__ == "__main__":
    main()
