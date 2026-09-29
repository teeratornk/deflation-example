"""Figures generated from complete follow-up records, including unsuccessful runs."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_small_report import summarize
from .reporting import file_sha256, write_report


COLORS = {"baseline": "#5b6573", "reference": "#0072b2", "recycling": "#d55e00"}
LABELS = {
    "baseline": "Frozen-preconditioned CG",
    "reference": "Fixed-reference deflation",
    "recycling": "Retained-vector recycling",
}


def plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def save(fig, output, name):
    for extension in ("pdf", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")


def transfer_figure(report, output):
    if report.get("schema") != "coupled-matched-transfer-v1" or report.get("status") != "complete":
        raise ValueError("Use a completed matched-transfer replay")
    rows = report["rows"]
    if len(rows) != report["expected_systems"]:
        raise ValueError("The transfer figure requires every declared system")
    plt = plotting()
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), layout="constrained")
    x = np.arange(1, len(rows) + 1)
    axes[0, 0].bar(
        x, [r["newly_inactive"] or 0 for r in rows], color="#009e73", label="Newly inactive"
    )
    axes[0, 0].step(
        x,
        [r["newly_active"] or 0 for r in rows],
        where="mid",
        color="#cc79a7",
        label="Newly active",
    )
    axes[0, 0].set_ylabel("Degrees of freedom")
    axes[0, 0].legend()
    for policy, label, color, marker, linestyle in (
        ("full", "Direct restriction", "#0072b2", "o", "-"),
        ("sequential", "Sequential zero extension", "#d55e00", "s", "--"),
    ):
        methods = [r["methods"][policy] for r in rows]
        style = dict(
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=5,
            markerfacecolor="none" if policy == "sequential" else color,
        )
        axes[0, 1].plot(x, [m["deployed_rank"] for m in methods], label=label, **style)
        cumulative = np.cumsum([m["iterations"] for m in methods])
        axes[1, 0].plot(x, cumulative, **style)
        axes[1, 0].annotate(
            str(cumulative[-1]),
            (x[-1], cumulative[-1]),
            xytext=(-4, -15 if policy == "full" else 9),
            textcoords="offset points",
            ha="right",
            color=color,
        )
        energies = [
            m.get("coarse_error", {}).get("energy_fraction_removed", np.nan) for m in methods
        ]
        axes[1, 1].plot(x, energies, **style)
        for i, m in enumerate(methods):
            if not m["verified"]:
                axes[1, 0].plot(
                    x[i],
                    np.cumsum([v["iterations"] for v in methods])[i],
                    "x",
                    color="black",
                    markersize=8,
                )
    axes[0, 1].set_ylabel("Deployed rank")
    axes[0, 1].set_ylim(0, 1 + max(m["deployed_rank"] for r in rows for m in r["methods"].values()))
    axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 1].legend()
    axes[1, 0].set_ylabel("Cumulative CG iterations")
    axes[1, 1].set_ylabel("Initial error energy removed")
    axes[1, 1].axhline(0, color="gray", linewidth=0.5)
    for axis in axes.flat:
        axis.set_xlabel("Captured inactive system")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlim(0.5, len(rows) + 0.5)
    fig.suptitle(
        "Matched quadratic inactive systems\nPhysical and trust-region bounds; recorded initial guesses"
    )
    save(fig, output, "matched_transfer")
    plt.close(fig)


def confirmation_figure(report, output):
    if report.get("schema") != "coupled-complete-confirmation-v1":
        raise ValueError("Use the all-outcome confirmation summary")
    plt = plotting()
    rows = report["runs"]
    gpu = any(r.get("device") in {"hybrid", "cuda"} for r in rows)
    memory_keys = ["peak_host_rss_bytes"] + (["peak_gpu_process_bytes"] if gpu else [])
    fig, axes = plt.subplots(
        1, 1 + len(memory_keys), figsize=(14 if gpu else 10.5, 4.5), layout="constrained"
    )
    for position, method in enumerate(("baseline", "reference", "recycling")):
        selected = [
            r
            for r in rows
            if "rank" in r
            and (
                "baseline"
                if r["rank"] == 0
                else "recycling"
                if r["variant"] == "recycling"
                else "reference"
            )
            == method
        ]
        for row in selected:
            x = position + 0.06 * (row["repetition"] - 2)
            symbol = "o" if row["verified"] else "x"
            elapsed = row.get("cumulative_attempt_seconds")
            if elapsed is None:
                axes[0].text(x, 0, row["status"].replace("_", " "), rotation=90, fontsize=7)
                continue
            axes[0].plot(x, elapsed / 60, symbol, color=COLORS[method])
            memory = row.get("memory") or {}
            if memory.get("complete"):
                for axis, key in zip(axes[1:], memory_keys, strict=True):
                    axis.plot(x, memory[key] / 2**30, symbol, color=COLORS[method])
            if not row["verified"]:
                axes[0].annotate(
                    row["status"].replace("_", " "),
                    (x, row["cumulative_attempt_seconds"] / 60),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                    rotation=35,
                )
        summary = report["methods"][method]
        if summary["median_complete_seconds"] is not None:
            axes[0].plot(
                [position - 0.18, position + 0.18],
                [summary["median_complete_seconds"] / 60] * 2,
                color=COLORS[method],
                linewidth=2,
            )
    labels = ["Frozen CG", "Reference", "Recycling"]
    for axis in axes:
        axis.set_xticks(range(3), labels)
        axis.set_ylim(bottom=0)
    failed = any(not row["verified"] for row in rows)
    axes[0].set_ylabel("Elapsed attempt time (min)" if failed else "Complete sequence time (min)")
    axes[1].set_ylabel("Sampled host process RSS (GiB)")
    if gpu:
        axes[2].set_ylabel("Sampled GPU process allocation (GiB)")
    fig.suptitle(
        "Three-target sequence attempts: all five repetitions\nCircles: verified; crosses: unsuccessful; bars: complete-population medians"
    )
    save(fig, output, "complete_confirmation")
    plt.close(fig)


def backend_figure(records, output):
    report = summarize(records)
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for axis, backend in zip(axes, ("hybrid", "cuda"), strict=True):
        for method, x in (("baseline", 0), ("reference", 1), ("recycling", 2)):
            matching = [
                r
                for r in report["runs"]
                if r.get("device") == backend
                and (
                    "baseline"
                    if r["rank"] == 0
                    else "recycling"
                    if r["variant"] == "recycling"
                    else "reference"
                )
                == method
            ]
            if len(matching) != 1:
                axis.text(x, 0, "missing", rotation=90, ha="center")
                continue
            row = matching[0]
            time = row.get("cumulative_attempt_seconds")
            if time is not None:
                axis.bar(
                    x,
                    time / 60,
                    color=COLORS[method],
                    alpha=1 if row["verified"] else 0.35,
                    hatch=None if row["verified"] else "//",
                )
                axis.text(
                    x,
                    time / 60,
                    f"{time / 60:.1f}\n{row['status'].replace('_', ' ')}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.set_xticks(range(3), ["Frozen CG", "Reference", "Recycling"])
        axis.set_title(
            "CPU sparse / GPU coarse" if backend == "hybrid" else "Resident GPU inner solver"
        )
        axis.set_ylabel("Complete attempt time (min)")
        axis.margins(y=0.25)
    fig.suptitle("Development target: one run per method and backend")
    save(fig, output, "backend_development")
    plt.close(fig)


def device_profile_figure(report, output):
    """Show independent operator costs without treating them as sequence costs."""
    if report.get("status") != "verified":
        raise ValueError("Use a verified CPU/GPU operator comparison")
    groups = (
        ("tangent", "Tangent action", ("cpu", "cuda")),
        ("transpose", "Transpose action", ("cpu", "cuda")),
        ("restricted_normal", "Inactive operator", ("cpu", "cuda")),
        ("frozen_inverse", "Frozen preconditioner", ("cpu", "cuda_serial", "cuda_block_diagonal")),
    )
    entries = {}
    for entry in report["components"]:
        key = (entry["action"], entry["backend"])
        if key in entries or not entry.get("verified", True):
            raise ValueError("Use unique, verified operator measurements")
        values = np.asarray([v["wall_seconds"] for v in entry["timings"]])
        if values.size != report["repetitions"] or not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError("Every operator requires all positive, finite repeated timings")
        entries[key] = values
    required = {(action, backend) for action, _, backends in groups for backend in backends}
    if set(entries) != required:
        raise ValueError("Keep every declared CPU/GPU operator and preconditioner layout")
    plt = plotting()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.9), layout="constrained")
    names = {
        "cpu": "CPU",
        "cuda": "GPU",
        "cuda_serial": "GPU\nserial",
        "cuda_block_diagonal": "GPU\nblock",
    }
    for axis, (action, title, backends) in zip(axes, groups, strict=True):
        for x, backend in enumerate(backends):
            values = entries[action, backend]
            axis.bar(x, np.median(values), color="#5b6573" if backend == "cpu" else "#0072b2")
            axis.plot(x + np.linspace(-0.08, 0.08, len(values)), values, ".", color="black")
        axis.set_xticks(range(len(backends)), [names[b] for b in backends])
        axis.set_title(title)
        axis.set_ylabel("Wall time per action (s)")
        axis.set_ylim(bottom=0)
    fig.suptitle(
        f"Matched operator actions: medians and all {report['repetitions']} repetitions\nSeparate diagnostics; setup excluded"
    )
    save(fig, output, "operator_costs")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--backend-records", type=Path, nargs="+")
    parser.add_argument("--device-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not any((args.transfer, args.confirmation, args.backend_records, args.device_profile)):
        parser.error("Supply at least one declared evidence group")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = []
    for path, plot in (
        (args.transfer, transfer_figure),
        (args.confirmation, confirmation_figure),
        (args.device_profile, device_profile_figure),
    ):
        if path is not None:
            plot(json.loads(path.read_text()), args.output)
            sources.append({"file": path.name, "sha256": file_sha256(path)})
    if args.backend_records:
        backend_figure(
            [json.loads(p.read_text()) if p.exists() else None for p in args.backend_records],
            args.output,
        )
        sources.extend(
            {"file": p.parent.name, "sha256": file_sha256(p) if p.exists() else None}
            for p in args.backend_records
        )
    write_report(
        args.output / "inputs.json",
        {"sources": sources, "scope": "All declared outcomes; no manuscript edits."},
    )


if __name__ == "__main__":
    main()
