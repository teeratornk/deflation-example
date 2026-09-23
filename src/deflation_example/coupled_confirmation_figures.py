"""Plot complete four-way sequences without pooling or omitting repetitions."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_confirmation_report import summarize
from .coupled_report import validate_record
from .reporting import file_sha256, write_report


LABELS = {
    "jacobi": "Jacobi-CG",
    "frozen": "Frozen preconditioner",
    "recycling": "Recycling + frozen",
    "reference": "Reference + frozen",
}
COLORS = dict(zip(LABELS, ("#666666", "#0072B2", "#E69F00", "#AA4499"), strict=True))
QUERY_COMPONENT = "target_optimizations_and_verification"


def measured_intervals(record):
    """Keep measured query intervals and charge unallocated overhead at the end.

    The endpoint is the measured sequence total. No sums of per-query medians
    are used. Preparation outside the sequence timer remains a separate value.
    """
    verified = validate_record(record)
    parts = record["components_seconds"]
    durations = np.asarray([c["seconds"] for c in record["cases"]], dtype=float)
    query_total = float(parts[QUERY_COMPONENT])
    reference = float(parts["reference_construction"])
    setup = sum(
        float(parts[name])
        for name in (
            "model_assembly_and_baseline_verification",
            "reference_construction",
            "solver_resources",
        )
    )
    total = float(record["sequence_seconds"])
    if (
        not np.isfinite(durations).all()
        or np.any(durations < 0)
        or durations.sum() > query_total + 1e-8
        or setup + durations.sum() > total + 1e-8
    ):
        raise ValueError("Query intervals must lie within the measured sequence")
    curve = np.concatenate(([setup], setup + np.cumsum(durations)))
    curve[-1] = total
    other = total - reference - query_total
    if other < -1e-8:
        raise ValueError("Reference and query costs exceed the complete interval")
    return {
        "verified": verified,
        "target_positions": list(range(len(record["cases"]) + 1)),
        "cumulative_seconds": curve.tolist(),
        "sequence_seconds": total,
        "cost_components_seconds": {
            "reference": reference,
            "optimization_and_verification": query_total,
            "other_sequence_work": other,
        },
        "preparation_inclusive_seconds": record["preparation_inclusive_seconds"],
        "statuses": [c["status"] for c in record["cases"]],
        "sampled_peak_host_bytes": record["memory"].get("peak_host_rss_bytes"),
        "sampled_peak_gpu_bytes": record["memory"].get("peak_gpu_process_bytes"),
    }


def figure_data(records, settings, fields=None):
    summary = summarize(records, settings, fields)
    groups = []
    for method in summary["methods"]:
        outcomes = []
        for row in method["outcomes"]:
            outcomes.append(
                {
                    **measured_intervals(records[row["index"]]),
                    "repetition": row["repetition"],
                    "record_index": row["index"],
                }
            )
        # Use one actual middle-ranked run for an additive component display.
        # Medians of separate components need not sum to the median total.
        ordered = sorted(outcomes, key=lambda r: (r["sequence_seconds"], r["repetition"]))
        representative = ordered[len(ordered) // 2] if method["eligible"] else None
        groups.append(
            {
                "arm": method["arm"],
                "outcomes": outcomes,
                "representative_record_index": None
                if representative is None
                else representative["record_index"],
            }
        )
    return {
        "schema": "coupled-confirmation-figures-v1",
        "summary": summary,
        "methods": groups,
        "timing_scope": "Each curve is one independently timed complete sequence. Initial construction and setup are included. The last point includes remaining query bookkeeping and cleanup. Calibration and process preparation are excluded and reported separately.",
        "components_scope": "Components belong to one middle-ranked complete run per method, not separate component medians. No complete component comparison is drawn for an incomplete method population.",
        "memory_scope": "Sampled whole-process host RSS and GPU allocation; all recorded repetitions are shown. Sampling can miss short-lived peaks.",
    }


def plot(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    component_names = ("reference", "optimization_and_verification", "other_sequence_work")
    component_colors = ("#AA4499", "#88CCEE", "#BBBBBB")
    maximum_position = 0
    for position, method in enumerate(data["methods"]):
        arm = method["arm"]
        color = COLORS[arm]
        for repeat, row in enumerate(method["outcomes"]):
            maximum_position = max(maximum_position, row["target_positions"][-1])
            axes[0].plot(
                row["target_positions"],
                np.asarray(row["cumulative_seconds"]) / 60,
                color=color,
                alpha=0.65,
                linewidth=1.1,
                linestyle="-" if row["verified"] else "--",
                marker="o" if row["verified"] else "x",
                markersize=3,
                label=LABELS[arm] if repeat == 0 else None,
            )
            for key, marker, offset in (
                ("sampled_peak_host_bytes", "o", -0.12),
                ("sampled_peak_gpu_bytes", "^", 0.12),
            ):
                value = row[key]
                if value is not None and np.isfinite(value) and value >= 0:
                    axes[2].scatter(
                        position + offset,
                        value / 2**30,
                        color=color,
                        marker=marker,
                        alpha=0.65,
                        s=25,
                    )
        chosen = method["representative_record_index"]
        if chosen is None:
            axes[1].text(position, 0, "Incomplete", rotation=90, ha="center", va="bottom")
            continue
        row = next(r for r in method["outcomes"] if r["record_index"] == chosen)
        bottom = 0.0
        for name, shade in zip(component_names, component_colors, strict=True):
            height = row["cost_components_seconds"][name] / 60
            axes[1].bar(position, height, bottom=bottom, color=shade, width=0.65)
            bottom += height
    axes[0].set(xlabel="Target position", ylabel="Complete-sequence time (min)")
    axes[0].set_xticks(range(maximum_position + 1))
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    names = [LABELS[m["arm"]] for m in data["methods"]]
    for axis in axes[1:]:
        axis.set_xticks(range(len(names)), names, rotation=28, ha="right", fontsize=8)
    axes[1].set_ylabel("Middle-ranked sequence (min)")
    axes[1].legend(
        handles=[
            Patch(color=color, label=label)
            for color, label in zip(
                component_colors,
                ("Reference construction", "Optimization + verification", "Other sequence work"),
                strict=True,
            )
        ],
        fontsize=8,
    )
    axes[2].set_ylabel("Sampled peak process allocation (GiB)")
    for marker, label in (("o", "Host RSS"), ("^", "GPU")):
        axes[2].scatter([], [], marker=marker, color="black", label=label)
    axes[2].legend(fontsize=8)
    fig.suptitle("Complete coupled optimization: " + data["summary"]["phase"], fontsize=11)
    for extension in ("pdf", "png"):
        fig.savefig(output / ("complete_sequences." + extension), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in args.records]
    fields, sources = [], []
    for path, record in zip(args.records, records, strict=True):
        row, saved = [], []
        if record.get("all_problems_verified"):
            for index in range(len(record["cases"])):
                field_path = path.parent / f"target-{index:02d}.npz"
                with np.load(field_path, allow_pickle=False) as field:
                    row.append(field["state"].copy())
                saved.append({"file": field_path.name, "sha256": file_sha256(field_path)})
        fields.append(row)
        sources.append(
            {
                "file": path.parent.name + "/" + path.name,
                "sha256": file_sha256(path),
                "states": saved,
            }
        )
    data = figure_data(records, json.loads(args.settings.read_text()), fields)
    data.update(records=sources, settings_sha256=file_sha256(args.settings))
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "figure_data.json", data)
    plot(data, args.output)


if __name__ == "__main__":
    main()
