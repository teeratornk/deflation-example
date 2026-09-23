"""Coverage-checked replay summaries and prospective configuration selection."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .coupled_trace import read_manifest
from .reporting import file_sha256, write_report


def amortization(rows, systems):
    """Constant-cost replay model, separate from complete optimization evidence."""
    controls = [row for row in rows if row["eligible"] and row["policy"] == "jacobi"]
    baseline = min(controls, key=lambda row: row["median_replay_total_seconds"], default=None)
    result = []
    if baseline is not None:
        baseline_online = (
            baseline["median_replay_total_seconds"] - baseline["construction_seconds_once"]
        )
        for row in rows:
            if not row["eligible"] or row["policy"] == "jacobi":
                continue
            online = row["median_replay_total_seconds"] - row["construction_seconds_once"]
            saving = baseline_online - online
            additional = row["construction_seconds_once"] - baseline["construction_seconds_once"]
            blocks = max(1, math.ceil(additional / saving)) if saving > 0 else None
            result.append(
                {
                    "policy": row["policy"],
                    "rank": row["rank"],
                    "preconditioner": row["preconditioner"],
                    "online_replay_median_seconds": online,
                    "online_saving_per_replay_block_seconds": saving,
                    "additional_construction_seconds": additional,
                    "constant_cost_break_even_replay_blocks": blocks,
                    "constant_cost_break_even_inner_solves": None
                    if blocks is None
                    else blocks * systems,
                }
            )
    return {
        "systems_per_replay_block": systems,
        "baseline_preconditioner": None if baseline is None else baseline["preconditioner"],
        "rows": result,
        "scope": "A conditional model repeating this same recorded system mix at its measured median costs. Blocks count linear-system replays, not optimization queries. Nonpositive online savings provide no amortization in this model. No complete-sequence speedup is inferred.",
    }


def summarize(manifest, reports, partition="selection", repetitions=3):
    expected_q = {i for i, q in enumerate(manifest["quadratics"]) if q["partition"] == partition}
    expected = {
        (q, i, rep)
        for i, row in enumerate(manifest["systems"])
        for q in [row["quadratic"]]
        if q in expected_q
        for rep in range(repetitions)
    }
    if not expected:
        raise ValueError("The declared partition contains no systems")
    groups = {}
    hashes = {(r["trace_sha256"], r["bank_sha256"]) for r in reports}
    if len(hashes) != 1:
        raise ValueError("Every comparison must share the same trace and reference bank")
    deployments = {
        json.dumps(
            {
                "device": r.get("device", {}),
                "cpu": r.get("environment", {}).get("cpu_model"),
                "source": r.get("environment", {}).get("source_sha256"),
            },
            sort_keys=True,
        )
        for r in reports
    }
    if len(deployments) != 1:
        raise ValueError("Compare matching implementations and hardware deployments")
    for report in reports:
        if report["feedback"] is not None or report["partition"] != partition:
            continue
        key = (
            report["policy"],
            report["rank"],
            report["width"],
            report.get("preconditioner", "jacobi"),
            report.get("frozen_sweeps", 0),
        )
        groups.setdefault(key, []).append(report)
    rows = []
    for (policy, rank, width, preconditioner, sweeps), records in sorted(groups.items()):
        observed, duplicates = set(), []
        failed = []
        totals = np.zeros(repetitions)
        iterations = np.zeros(repetitions, dtype=np.int64)
        builds = {r["construction_seconds_once"] for r in records}
        if len(builds) != 1:
            raise ValueError("One policy must use one fixed construction cost")
        construction = next(iter(builds))
        if not np.isfinite(construction) or construction < 0:
            raise ValueError("Construction costs must be finite and nonnegative")
        totals += construction
        for report in records:
            for row in report["rows"]:
                identity = (report["quadratic"], row["system"], row["repetition"])
                if identity in observed:
                    duplicates.append(identity)
                observed.add(identity)
                if (
                    not row["verified"]
                    or row["status"] != "converged"
                    or not np.isfinite(row["original_residual"])
                    or row["original_residual"] < 0
                    or row["original_residual"] > report["final_residual_tolerance"]
                ):
                    failed.append(identity)
                if not np.isfinite(row["solve_seconds"]) or row["solve_seconds"] < 0:
                    raise ValueError("Measured solve costs must be finite and nonnegative")
                if 0 <= row["repetition"] < repetitions:
                    totals[row["repetition"]] += row["solve_seconds"]
                    iterations[row["repetition"]] += row["iterations"]
            cleanup = report.get("cleanup_seconds", [])
            if len(cleanup) == repetitions:
                totals += cleanup
            resources = report.get("resource_creation_seconds", [])
            if not np.isfinite(cleanup + resources).all() or any(
                v < 0 for v in cleanup + resources
            ):
                raise ValueError("Resource costs must be finite and nonnegative")
            if len(resources) == repetitions:
                totals += resources
        complete = (
            observed == expected
            and not duplicates
            and not failed
            and {r["quadratic"] for r in records} == expected_q
            and all(
                r["status"] == "complete"
                and len(r.get("cleanup_seconds", [])) == repetitions
                and len(r.get("resource_creation_seconds", [])) == repetitions
                for r in records
            )
        )
        rows.append(
            {
                "policy": policy,
                "rank": rank,
                "width": width,
                "preconditioner": preconditioner,
                "frozen_sweeps": sweeps,
                "eligible": complete,
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
                "duplicates": duplicates,
                "failed": failed,
                "replay_totals_seconds": totals.tolist() if complete else None,
                "median_replay_total_seconds": float(np.median(totals)) if complete else None,
                "total_iterations": iterations.tolist() if complete else None,
                "construction_seconds_once": construction,
                "sampled_peak_gpu_bytes": max(
                    (r.get("memory", {}).get("peak_gpu_process_bytes", 0) for r in records),
                    default=0,
                ),
            }
        )
    eligible = [r for r in rows if r["eligible"]]
    references = [r for r in eligible if r["policy"] != "jacobi"]
    best = min(
        references,
        key=lambda r: (r["median_replay_total_seconds"], r["rank"], r["width"]),
        default=None,
    )
    jacobi = [r for r in eligible if r["policy"] == "jacobi"]
    baseline = min((r["median_replay_total_seconds"] for r in jacobi), default=None)
    return {
        "schema": "coupled-retention-summary-v1",
        "partition": partition,
        "rows": rows,
        "best_reference": best,
        "reference_beats_jacobi": best is not None
        and baseline is not None
        and best["median_replay_total_seconds"] < baseline,
        "baseline_scope": "Fastest verified rank-zero configuration, including frozen preconditioning when present.",
        "amortization": amortization(rows, len(expected) // repetitions),
        "scope": "Sums of matched replay timings by repetition with construction charged once; not independently timed complete optimization.",
    }


def replay_label(row):
    """Display the preconditioner actually applied by the rank-zero host kernel."""
    preconditioner = "Frozen" if row.get("preconditioner", "jacobi") == "frozen" else "Jacobi"
    if row["policy"] == "jacobi":
        return f"CG\n{preconditioner} preconditioner\nr=0"
    names = {
        "thermal": "Thermal reference",
        "nominal_coupled": "Jacobi Ritz",
        "preconditioned_coupled": "Energy Ritz",
        "krylov_coupled": "Coupled Krylov",
    }
    return f"{names[row['policy']]}\nr={row['rank']}\n{preconditioner} preconditioner"


def plot(summary, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in summary["rows"] if r["eligible"]]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    labels = [replay_label(row) for row in rows]
    for i, row in enumerate(rows):
        values = np.array(row["replay_totals_seconds"])
        median = np.median(values)
        axes[0].errorbar(
            i, median, yerr=[[median - values.min()], [values.max() - median]], fmt="o", capsize=3
        )
        axes[0].scatter(i + np.linspace(-0.05, 0.05, len(values)), values, s=12, color="black")
    axes[0].set_ylabel("Setup-inclusive replay sum (s)")
    axes[1].bar(np.arange(len(rows)), [np.median(r["total_iterations"]) for r in rows])
    axes[1].set_ylabel("Total inner iterations")
    for ax in axes:
        ax.set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(directory / "replay_cost.pdf")
    fig.savefig(directory / "replay_cost.png", dpi=180)
    plt.close(fig)


def summarize_feedback(reports):
    """Retain failed diagnostics alongside measured energy fractions."""
    if len({(r["trace_sha256"], r["bank_sha256"]) for r in reports}) != 1:
        raise ValueError("Feedback diagnostics require one trace and fixed reference bank")
    rows = []
    for report in reports:
        if report["feedback"] not in {0, 0.5, 1}:
            raise ValueError("A declared diagnostic feedback multiplier is required")
        for item in report["rows"]:
            diagnostic = item.get("error_diagnostic", {})
            rows.append(
                {
                    "feedback": report["feedback"],
                    "policy": report["policy"],
                    "rank": report["rank"],
                    "quadratic": report["quadratic"],
                    "system": item["system"],
                    "verified": item["verified"],
                    "status": item["status"],
                    "iterations": item["iterations"],
                    "diagnostic_status": diagnostic.get("status", "unavailable"),
                    "energy_fraction_removed": diagnostic.get("energy_fraction_removed"),
                    "deployed_rank": item["deployed_rank"],
                }
            )
    return {
        "schema": "coupled-feedback-summary-v1",
        "rows": rows,
        "outcomes": [
            {
                "policy": r["policy"],
                "rank": r["rank"],
                "quadratic": r["quadratic"],
                "feedback": r["feedback"],
                "status": r["status"],
                "recorded_systems": len(r["rows"]),
            }
            for r in reports
        ],
        "scope": "Fixed recorded masks, loads, and initial guesses; recomputed flow and Gauss-Newton without secants. Not complete optimization.",
    }


def plot_feedback(summary, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
    for policy in sorted({r["policy"] for r in summary["rows"]}):
        for q in sorted({r["quadratic"] for r in summary["rows"]}):
            rows = sorted(
                (
                    r
                    for r in summary["rows"]
                    if r["policy"] == policy
                    and r["quadratic"] == q
                    and r["verified"]
                    and r["diagnostic_status"] == "verified"
                    and r["energy_fraction_removed"] is not None
                ),
                key=lambda r: (r["system"], r["feedback"]),
            )
            for system in sorted({r["system"] for r in rows}):
                selected = [r for r in rows if r["system"] == system]
                label = f"{policy}, system {system}"
                axes[0].plot(
                    [r["feedback"] for r in selected],
                    [r["energy_fraction_removed"] for r in selected],
                    "o-",
                    label=label,
                )
                axes[1].plot(
                    [r["feedback"] for r in selected],
                    [r["iterations"] for r in selected],
                    "o-",
                    label=label,
                )
    for ax in axes:
        ax.set_xlabel("Feedback multiplier (diagnostic)")
        ax.set_xticks([0, 0.5, 1])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Initial error energy removed")
    axes[1].set_ylabel("CG iterations")
    axes[0].legend(fontsize=7)
    fig.savefig(directory / "feedback_correction.pdf")
    fig.savefig(directory / "feedback_correction.png", dpi=180)
    plt.close(fig)


def plot_complete(rows, directory, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
    labels = []
    for i, row in enumerate(rows):
        valid = [v for v in row["outcomes"] if v["verified"]]
        labels.append(f"{row['policy']}\nr={row['rank']} ({len(valid)}/{len(row['outcomes'])})")
        if valid:
            values = np.array([v["sequence_seconds"] for v in valid])
            median = np.median(values)
            axes[0].errorbar(
                i,
                median,
                yerr=[[median - values.min()], [values.max() - median]],
                fmt="o",
                capsize=4,
            )
            axes[0].scatter(np.full(len(values), i), values, s=10, alpha=0.6)
            memory = [v["sampled_gpu_peak_bytes"] / 2**30 for v in valid]
            axes[1].scatter(np.full(len(memory), i), memory, s=15)
    for ax in axes:
        ax.set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Complete initialized optimization (s)")
    axes[1].set_ylabel("Sampled GPU process allocation (GiB)")
    fig.savefig(directory / (name + "_cost_memory.pdf"))
    fig.savefig(directory / (name + "_cost_memory.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--partition", choices=("selection", "held_out"), default="selection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--feedback", action="store_true")
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.records]
    digest = file_sha256(args.trace / "manifest.json")
    if any(r["trace_sha256"] != digest for r in reports):
        raise ValueError("Replay records refer to a different trace")
    summary = (
        summarize_feedback(reports)
        if args.feedback
        else summarize(read_manifest(args.trace), reports, args.partition)
    )
    summary["records"] = [
        {"file": path.parent.name + "/" + path.name, "sha256": file_sha256(path)}
        for path in args.records
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", summary)
    if args.plot:
        (plot_feedback if args.feedback else plot)(summary, args.output)


if __name__ == "__main__":
    main()
