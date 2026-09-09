"""Validate and plot all GPU repetitions without running a numerical solve."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from .reporting import atomic_output, write_report
from .timing import PHASES

METHODS = ("deflated_gpu_qr", "amgx_fresh_resources", "amgx_persistent_resources")
LABELS = ("Deflation / GPU QR", "AmgX / fresh resources", "AmgX / persistent resources")
COLORS = ("#0072B2", "#D55E00", "#009E73")


def checked_series(report):
    """Cumulative instance medians, not independently timed complete sequences.

    Persistent-session creation and destruction are charged once to every
    plotted prefix. Runtime/library initialization and CPU PDAS are outside
    these per-grid kernel sequences and are reported separately.
    """
    if report["protocol"] != "itemized-gpu-resource-control-v1" or not report["success"]:
        raise ValueError("A complete successful itemized benchmark is required")
    series = {}
    repetitions = report["controls"]["repeats"]
    for grid in report["grids"]:
        n = grid["n"]
        cases = [c for c in report["cases"] if c["n"] == n]
        if [c["index"] for c in cases] != [0, 1, 2]:
            raise ValueError("Expected three ordered cases per grid")
        methods = {}
        for method in METHODS:
            warmup = grid["warmups"][method]["total_seconds"]
            reference = grid["reference_seconds"] if method == "deflated_gpu_qr" else 0.
            session = sum(grid[k]["total_seconds"] for k in
                          ("persistent_session_startup", "persistent_session_cleanup")) \
                if method == "amgx_persistent_resources" else 0.
            cumulative = [warmup + reference + session]
            trials = []
            for case in cases:
                runs = case["repetitions"][method]
                if sorted(r["repetition"] for r in runs) != list(range(repetitions)):
                    raise ValueError("Missing or duplicate timing repetitions")
                for run in runs:
                    if run["status"] != "converged" or not 0 <= run["residual"] <= 1e-10:
                        raise ValueError("An unaccepted solve cannot enter a timing comparison")
                    components = run["components_seconds"]
                    if set(components) != set(PHASES) or any(not math.isfinite(v) or v < 0 for v in components.values()):
                        raise ValueError("Invalid timing components")
                    if not math.isclose(sum(components.values()), run["total_seconds"], rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("Timing components do not sum to the total")
                    if "amgx" in method and run["hierarchy_reused"]:
                        raise ValueError("This control must rebuild each hierarchy")
                times = [r["total_seconds"] for r in runs]
                selected = case["solvers"][method]
                if not math.isclose(selected["total_seconds"], median(times), rel_tol=1e-12):
                    raise ValueError("Summary must select the median-total repetition")
                matching = [r for r in runs if r["total_seconds"] == selected["total_seconds"]]
                if not any(r["components_seconds"] == selected["components_seconds"] for r in matching):
                    raise ValueError("Component entries must come from the same median-total run")
                cost = selected["total_seconds"] + case["restriction_seconds"]
                if method == "deflated_gpu_qr":
                    cost += case["basis_restriction_seconds"]
                cumulative.append(cumulative[-1] + cost)
                trials.append(times)
            methods[method] = {"cumulative_seconds": cumulative, "repetition_seconds": trials,
                               "warmup_seconds": warmup, "session_seconds": session}
        series[n] = {"reference_seconds": grid["reference_seconds"], "methods": methods}
    return series


def plot(series, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.size": 8, "axes.titlesize": 9}):
        fig, axes = plt.subplots(2, len(series), figsize=(7.2, 5.2), squeeze=False, layout="constrained")
        for column, (n, data) in enumerate(series.items()):
            top, bottom = axes[:, column]
            for j, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
                values = data["methods"][method]
                top.plot(range(4), values["cumulative_seconds"], "o-", color=color, label=label, markersize=3)
                for index, trials in enumerate(values["repetition_seconds"], 1):
                    center = index + (j-1)*.2
                    bottom.scatter([center+(k-(len(trials)-1)/2)*.02 for k in range(len(trials))],
                                   [1000*t for t in trials], color=color, s=12, alpha=.8)
                    bottom.plot([center-.06, center+.06], [1000*median(trials)]*2, color=color, lw=2)
            top.set(title=f"{n} x {n}", xlabel="Completed instances", ylabel="Cumulative cost (s)", xticks=range(4), ylim=(0, None))
            bottom.set(xlabel="Instance", ylabel="Kernel time (ms)", xticks=(1, 2, 3), xlim=(.5, 3.5))
            for ax in (top, bottom):
                ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=7)
        fig.savefig(destination, metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    series = checked_series(report)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "cumulative.json", series)
    with atomic_output(args.output / "repetitions.csv") as stream:
        writer = csv.DictWriter(stream, fieldnames=["n", "instance", "method", "repetition", "iterations", "rank", "residual", *PHASES, "total_seconds"])
        writer.writeheader()
        for case in report["cases"]:
            for method, runs in case["repetitions"].items():
                for run in runs:
                    writer.writerow({"n": case["n"], "instance": case["index"]+1, "method": method,
                        **{k: run[k] for k in ("repetition", "iterations", "rank", "residual", "total_seconds")},
                        **run["components_seconds"]})
    if args.plot:
        plot(series, args.output / "cumulative.pdf")


if __name__ == "__main__":
    main()
