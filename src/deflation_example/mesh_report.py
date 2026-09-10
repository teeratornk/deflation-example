"""Generate complete-sequence mesh comparison tables and figures from records."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .mesh_presentation import summarize
from .reporting import atomic_output, write_report, environment


NAMES = {"jacobi": "Jacobi-CG", "reference": "Reference", "recycling": "Recycling", "amgx": "AmgX"}
COLORS = {"jacobi": "#777777", "reference": "#0072b2", "recycling": "#cc79a7", "amgx": "#d55e00"}


def components(record):
    """Partition the complete interval using nonoverlapping nested timers."""
    result = dict.fromkeys(
        ("Construction and setup", "Iteration", "Transfers", "Verification", "Other"), 0.0
    )
    result["Construction and setup"] = sum(
        record["components_seconds"][k]
        for k in ("assembly", "reference_construction", "solver_resources")
    )
    for case in record["cases"]:
        outer = case["outer_timing"]["components_seconds"]
        result["Construction and setup"] += outer["restriction"]
        result["Verification"] += outer["inner_verification"] + outer["kkt_and_update"]
        for inner in case["inner"]:
            p = inner["components_seconds"]
            if not np.isclose(sum(p.values()), inner["total_seconds"], rtol=1e-12):
                raise ValueError("Inner timing components do not sum to their interval")
            result["Construction and setup"] += inner["restriction_and_transfer_seconds"] + sum(
                p[k]
                for k in (
                    "basis_processing",
                    "coarse_or_hierarchy_setup",
                    "initialization",
                    "handle_creation",
                    "resource_creation",
                )
            )
            result["Iteration"] += p["iteration"]
            result["Transfers"] += sum(p[k] for k in ("conversion", "upload", "download"))
            result["Verification"] += p["verification"]
    result["Other"] = record["seconds"] - sum(result.values())
    if result["Other"] < -1e-8:
        raise ValueError("The complete component partition double-counts measured work")
    return result


def load_study(directory):
    directory = Path(directory)
    report = summarize(directory)
    records = {m: [] for m in report["controls"]["methods"]}
    attempts = json.loads((directory / "attempts.json").read_text())
    completed = {(a["method"], a["repetition"]): a for a in attempts if a["status"] == "completed"}
    sources = {}
    for path in sorted(directory.glob("*/record.json")):
        r = json.loads(path.read_text())
        sources[path.relative_to(directory).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        if r["success"] and (r["method"], r["repetition"]) in completed:
            records[r["method"]].append(r)
    report["input_record_sha256"] = sources
    for method in report["methods"]:
        accepted = records[method["method"]]
        if len(accepted) != method["accepted"]:
            raise ValueError("Accepted records differ from the checked attempt population")
        method["components_seconds"] = [components(r) for r in accepted]
        method["startup_seconds"] = [r["startup_after_import_seconds"] for r in accepted]
        method["finalization_seconds"] = [r["finalization_seconds"] for r in accepted]
        method["library_preparation_inclusive_seconds"] = [
            r["seconds"] + r["startup_after_import_seconds"] + r["finalization_seconds"]
            for r in accepted
        ]
        method["whole_worker_process_seconds"] = [
            completed[(r["method"], r["repetition"])]["process_wall_seconds"] for r in accepted
        ]
        method["max_original_residual"] = max(
            (i["original_residual"] for r in accepted for c in r["cases"] for i in c["inner"]),
            default=None,
        )
        method["max_kkt"] = max(
            (max(c["kkt"].values()) for r in accepted for c in r["cases"]), default=None
        )
        ranks = [i["deployed_rank"] for r in accepted for c in r["cases"] for i in c["inner"]]
        method["deployed_rank_range"] = [min(ranks), max(ranks)] if ranks else None
        method["fallback_count"] = sum(
            i["fallback_reason"] is not None
            for r in accepted
            for c in r["cases"]
            for i in c["inner"]
        )
    sample = next((r for group in records.values() for r in group), None)
    report["mesh"] = sample["mesh"] if sample else None
    report["problem_size"] = sample["problem_size"] if sample else None
    return report, records


def _label(report):
    c = report["controls"]
    name = "Transformer" if c["geometry"] == "transformer_2d" else "Bore-in-block"
    form = f"{c['slabs']} slabs" if c["transient"] else "steady"
    return f"{name}, level {c['level']}, {form}, rank {c['rank']}"


def generate(directories, output, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    loaded = [load_study(d) for d in directories]
    reports = [r for r, _ in loaded]
    write_report(
        output / "summary.json",
        {
            "protocol": "mesh-comparison-report-v1",
            "generator_environment": environment(),
            "studies": reports,
        },
    )
    main_rows, detail_rows, accuracy_rows, component_rows, preparation_rows = [], [], [], [], []
    for report, records in loaded:
        c = report["controls"]
        name = "Transformer" if c["geometry"] == "transformer_2d" else f"Bore {c['level']}"
        if c["rank"] != 100:
            name += f", $r={c['rank']}$"
        kind = f"{c['slabs']} slabs" if c["transient"] else "Steady"
        methods = {m["method"]: m for m in report["methods"]}
        entries = []
        for key in NAMES:
            m = methods.get(key)
            if m is None or m["accepted"] != m["requested"]:
                entries.append("---")
            else:
                entries.append(f"{m['median_seconds']:.3f}")
        main_rows.append(" & ".join([name, kind, str(report["problem_size"])] + entries) + r" \\")
        for m in report["methods"]:
            times = m["complete_seconds"]
            if times:
                timing = f"{np.median(times):.3f} [{min(times):.3f}, {max(times):.3f}]"
                inner, outer = (
                    f"{np.median(m[k]):.0f}" for k in ("inner_iterations", "outer_iterations")
                )
                host = f"{max(m['peak_host_bytes']) / 2**30:.3f}" if m["peak_host_bytes"] else "---"
                gpu = f"{max(m['peak_gpu_bytes']) / 2**30:.3f}" if m["peak_gpu_bytes"] else "---"
                accuracy_rows.append(
                    " & ".join(
                        [
                            name,
                            kind,
                            NAMES[m["method"]],
                            f"{m['max_original_residual']:.2e}",
                            f"{m['max_kkt']:.2e}",
                            "--".join(map(str, m["deployed_rank_range"])),
                            str(m["fallback_count"]),
                        ]
                    )
                    + r" \\"
                )
                representative = sorted(records[m["method"]], key=lambda r: r["seconds"])[
                    len(times) // 2
                ]
                parts = components(representative)
                component_rows.append(
                    " & ".join(
                        [name, kind, NAMES[m["method"]]]
                        + [f"{v:.3f}" for v in parts.values()]
                        + [f"{representative['seconds']:.3f}"]
                    )
                    + r" \\"
                )
                preparation_rows.append(
                    " & ".join(
                        [name, kind, NAMES[m["method"]]]
                        + [
                            f"{np.median(m[k]):.3f}"
                            for k in (
                                "complete_seconds",
                                "startup_seconds",
                                "finalization_seconds",
                                "library_preparation_inclusive_seconds",
                                "whole_worker_process_seconds",
                            )
                        ]
                    )
                    + r" \\"
                )
            else:
                timing, inner, outer, host, gpu = ("---",) * 5
            detail_rows.append(
                " & ".join(
                    [
                        name,
                        kind,
                        NAMES[m["method"]],
                        f"{m['accepted']}/{m['requested']}",
                        timing,
                        outer,
                        inner,
                        host,
                        gpu,
                    ]
                )
                + r" \\"
            )
    for name, rows in (
        ("complete_rows.tex", main_rows),
        ("detail_rows.tex", detail_rows),
        ("accuracy_rows.tex", accuracy_rows),
        ("component_rows.tex", component_rows),
        ("preparation_rows.tex", preparation_rows),
    ):
        with atomic_output(output / name) as stream:
            stream.write("\n".join(rows) + "\n")
    if plots:
        plot_sequences(loaded, output / "complete_sequences.pdf")
        plot_components(loaded, output / "complete_components.pdf")
    largest, records = max(loaded, key=lambda pair: pair[0]["problem_size"] or 0)
    decision_rows = []
    for method, data in records.items():
        if len(data) != largest["controls"]["repeats"]:
            continue
        ordered = sorted(data, key=lambda record: record["seconds"])
        record = ordered[len(ordered) // 2]
        parts = components(record)
        item = next(m for m in largest["methods"] if m["method"] == method)
        gpu = f"{max(item['peak_gpu_bytes']) / 2**30:.3f}" if item["peak_gpu_bytes"] else "---"
        remaining = parts["Transfers"] + parts["Verification"] + parts["Other"]
        decision_rows.append(
            " & ".join(
                [
                    NAMES[method],
                    f"{parts['Construction and setup']:.3f}",
                    f"{parts['Iteration']:.3f}",
                    f"{remaining:.3f}",
                    f"{record['seconds']:.3f} [{ordered[0]['seconds']:.3f}, {ordered[-1]['seconds']:.3f}]",
                    gpu,
                ]
            )
            + r" \\"
        )
    with atomic_output(output / "largest_cost_rows.tex") as stream:
        stream.write("\n".join(decision_rows) + "\n")
    return reports


def plot_sequences(loaded, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = (len(loaded) + 1) // 2
    fig, axes = plt.subplots(
        rows, 2, figsize=(8, 2.7 * rows), squeeze=False, constrained_layout=True
    )
    for ax, (report, records) in zip(axes.flat, loaded):
        for method, data in records.items():
            curves = []
            for r in data:
                if r.get("initial_setup_seconds") is None:
                    raise ValueError("Cumulative plots require actual sequence timestamps")
                y = [r["initial_setup_seconds"]] + [c["cumulative_seconds"] for c in r["cases"]]
                y[-1] = r["seconds"]
                curves.append(y)
                ax.plot(range(len(y)), y, color=COLORS[method], alpha=0.23, linewidth=0.8)
            if curves:
                ax.plot(
                    range(len(curves[0])),
                    np.median(curves, axis=0),
                    color=COLORS[method],
                    linewidth=1.7,
                    label=f"{NAMES[method]} ({len(data)}/{report['controls']['repeats']})",
                )
        ref = records.get("reference")
        alternatives = [
            (float(np.median([r["seconds"] for r in data])), method)
            for method, data in records.items()
            if method != "reference" and len(data) == report["controls"]["repeats"]
        ]
        if ref and len(ref) == report["controls"]["repeats"] and alternatives:
            fastest, method = min(alternatives)
            ratio = fastest / np.median([r["seconds"] for r in ref])
            ax.text(
                0.03,
                0.96,
                f"Fastest alternative: {NAMES[method]}\nAlternative / reference: {ratio:.2f}×",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
            )
        ax.set(
            title=_label(report),
            xlabel="Completed targets",
            ylabel="Complete elapsed time (s)",
            xlim=(0, report["controls"]["targets"]),
            xticks=sorted(
                set(range(0, report["controls"]["targets"] + 1, 4))
                | {report["controls"]["targets"]}
            ),
            ylim=(0, None),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="lower right")
    for ax in list(axes.flat)[len(loaded) :]:
        ax.set_visible(False)
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def plot_components(loaded, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        (len(loaded) + 1) // 2,
        2,
        figsize=(10, 2.6 * ((len(loaded) + 1) // 2)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = ["#56b4e9", "#0072b2", "#e69f00", "#009e73", "#aaaaaa"]
    for ax, (report, records) in zip(axes.flat, loaded):
        for position, (method, data) in enumerate(records.items()):
            if not data:
                ax.text(position, 0, "No accepted sequence", rotation=90, fontsize=7)
                continue
            ordered = sorted(data, key=lambda r: r["seconds"])
            representative = ordered[len(ordered) // 2]
            base = 0
            for (name, seconds), color in zip(components(representative).items(), colors):
                ax.bar(
                    position,
                    seconds,
                    bottom=base,
                    color=color,
                    label=name if position == 0 else None,
                )
                base += seconds
        ax.set(
            title=_label(report),
            ylabel="Complete elapsed time (s)",
            xticks=range(len(records)),
            xticklabels=[NAMES[m] for m in records],
        )
        ax.legend(fontsize=6, loc="upper left")
    for ax in list(axes.flat)[len(loaded) :]:
        ax.set_visible(False)
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    generate(args.studies, args.output, not args.no_plots)


if __name__ == "__main__":
    main()
