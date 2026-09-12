"""Summarize the paired transport study without combining its evidence populations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from deflation_example.reporting import write_report


def validate_pair(original, corrected):
    a, b = original["design"], corrected["design"]
    if a["protocol"]["transport_form"] != "advective" or b["protocol"]["transport_form"] != "skew":
        raise ValueError("Expected the original advective and separate skew pilot")
    for key in ("input_sha256", "queries", "cases", "desired_exceedance_scores", "physical"):
        if a[key] != b[key]:
            raise ValueError(f"Paired designs differ in {key}")
    if {k: v for k, v in a["protocol"].items() if k != "transport_form"} != {
        k: v for k, v in b["protocol"].items() if k != "transport_form"
    }:
        raise ValueError("Transport is the only changed optimization setting")
    for summary in (original, corrected):
        expected = [(c["query"], c["slabs"]) for c in summary["design"]["cases"]]
        if [(r["query"], r["slabs"]) for r in summary["rows"]] != expected:
            raise ValueError("Every declared optimization must remain in the summary")


def generate(original_path, corrected_path, stability_path, replay_paths, output, plots=True):
    paths = [
        Path(original_path),
        Path(corrected_path),
        Path(stability_path),
        *[Path(p) for p in replay_paths],
    ]
    original, corrected, stability, *replays = [json.loads(p.read_text()) for p in paths]
    validate_pair(original, corrected)
    design = corrected["design"]
    if stability["input_sha256"] != design["input_sha256"]:
        raise ValueError("Stability diagnostic uses different thermal inputs")
    endpoints = {min(design["protocol"]["slabs"]), max(design["protocol"]["slabs"])}
    selected = [i for i, c in enumerate(design["cases"]) if c["slabs"] in endpoints]
    if sorted(r["case"] for r in replays) != selected:
        raise ValueError("Keep every selected fixed-source replay exactly once")
    for replay in replays:
        if replay["protocol_sha256"] != design["protocol_sha256"] or (
            replay["optimization_source"]["source_sha256"] != design["environment"]["source_sha256"]
        ):
            raise ValueError("Replay uses a different optimized-source population")
        if replay["optimization_case"] != design["cases"][replay["case"]]:
            raise ValueError("Replay index differs from its selected optimization")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "paired-prescribed-transport-assessment-v1",
        "input_sha256": {
            str(i): hashlib.sha256(p.read_bytes()).hexdigest() for i, p in enumerate(paths)
        },
        "input_order": [
            "original summary",
            "corrected summary",
            "stability",
            "selected replays in command order",
        ],
        "original": original,
        "corrected": corrected,
        "stability": stability,
        "fixed_source_replays": replays,
        "scope": "Separate transport-form pilot and fixed-source time refinement; no replacement of published optimization timings and no solver speedup comparison.",
    }
    write_report(output / "assessment.json", report)
    if plots:
        figures(report, output)
    return report


def figures(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for form, label, color in (
        ("advective", "Original transport", "#d55e00"),
        ("skew", "Energy-consistent pilot", "#0072b2"),
    ):
        rows = [r for r in report["stability"]["rows"] if r["transport_form"] == form]
        axes[0].plot(
            [r["slabs"] for r in rows],
            [
                r["largest_computed_amplification_modulus"]
                if r["status"] == "eigenpair_verified"
                else np.nan
                for r in rows
            ],
            "o-",
            color=color,
            label=label,
        )
        failed = [r for r in rows if r["status"] != "eigenpair_verified"]
        if failed:
            axes[0].text(
                0.03,
                0.97,
                f"Unverified {form} diagnostic: "
                + ", ".join(str(r["slabs"]) for r in failed)
                + " slabs",
                transform=axes[0].transAxes,
                va="top",
                fontsize=8,
            )
    axes[0].axhline(1, color="0.35", ls=":", label="Unit amplification")
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Forward time steps over 600 s",
        ylabel="Largest computed amplification modulus",
        title="(a) Forward thermal modes",
    )
    axes[0].set_xticks([4, 8, 16, 32, 64, 128, 256])
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].legend(fontsize=8, loc="lower right")
    rows = report["corrected"]["rows"]
    queries = report["corrected"]["design"]["queries"]
    for query, color, label in zip(
        queries, ["#0072b2", "#d55e00"], ["Nominal target", "Demanding target"], strict=True
    ):
        data = [r for r in rows if r["query"] == query]
        axes[1].plot(
            [r["slabs"] for r in data],
            [
                r.get("temperature_rms_difference_K", np.nan) if r["verified"] else np.nan
                for r in data
            ],
            "o-",
            color=color,
            label=label,
        )
    axes[1].set(
        xscale="log",
        xlabel="Optimization time slabs",
        ylabel="Weighted temperature difference (K)",
        title="(b) Corrected optimization",
    )
    axes[1].set_xticks(report["corrected"]["design"]["protocol"]["slabs"])
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].legend(fontsize=8)
    axes[1].text(
        0.03,
        0.97,
        "Root-mean-square difference from 64 slabs\nLinear state reconstruction; fixed spatial mesh",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
    )
    fig.savefig(
        output / "transport_and_optimization.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for replay in sorted(report["fixed_source_replays"], key=lambda r: r["case"]):
        case = replay["optimization_case"]
        query = case["query"]
        color = "#0072b2" if query == queries[0] else "#d55e00"
        style = (
            "--"
            if case["slabs"] == min(report["corrected"]["design"]["protocol"]["slabs"])
            else "-"
        )
        label = f"{'Nominal' if query == queries[0] else 'Demanding'}, {case['slabs']}-slab source"
        for axis, key in zip(
            axes, ["temperature_change_max_K", "maximum_bound_violation_K"], strict=True
        ):
            data = replay["rows"]
            axis.plot(
                [r["slabs"] for r in data],
                [r.get(key, np.nan) if r["status"] == "verified" else np.nan for r in data],
                marker="o",
                ls=style,
                color=color,
                label=label,
            )
    axes[0].axhline(
        report["corrected"]["design"]["protocol"]["temperature_change_K"],
        color="0.35",
        ls=":",
        label="0.05 K comparison scale",
    )
    axes[0].set(
        ylabel="Maximum consecutive-grid change (K)", title="(a) Fixed-source temporal sensitivity"
    )
    axes[1].set(ylabel="Maximum bound violation (K)", title="(b) Bound checks at all replay levels")
    for axis in axes:
        axis.set(xscale="log", xlabel="Forward time steps over 600 s")
        axis.set_xticks(
            sorted({n for r in report["fixed_source_replays"] for n in r["replay_slabs"]})
        )
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.legend(fontsize=7)
    fig.savefig(
        output / "fixed_source_resolution.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--corrected", required=True)
    parser.add_argument("--stability", required=True)
    parser.add_argument("--replays", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    generate(
        args.original, args.corrected, args.stability, args.replays, args.output, not args.no_plots
    )


if __name__ == "__main__":
    main()
