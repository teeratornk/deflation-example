"""Plot optimality and inner work from matched coupled optimization pilots.

These figures describe nonlinear convergence. Complete-sequence timing and
physical-resolution assessments have separate records and acceptance tests.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_saved import file_digest
from .reporting import write_report

METHODS = ("jacobi", "reference", "recycling")
LABELS = {"jacobi": "Jacobi-CG", "reference": "Reference deflation", "recycling": "Recycling"}
COLORS = {"jacobi": "#666666", "reference": "#0072B2", "recycling": "#CC79A7"}
KKT_COMPONENTS = {
    "primal_absolute",
    "stationarity",
    "dual_feasibility",
    "lower_complementarity",
    "upper_complementarity",
}


def maximum_kkt(components):
    if set(components) != KKT_COMPONENTS:
        raise ValueError("All five declared optimality components are required")
    values = np.asarray(list(components.values()), dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Optimality residuals must be finite and nonnegative")
    return float(values.max())


def identity(record):
    env = record["environment"]
    return {
        "configuration": {k: v for k, v in record["configuration"].items() if k != "methods"},
        "baseline_sha256": record["baseline_sha256"],
        "source_sha256": env["source_sha256"],
        "cpu_model": env["cpu_model"],
        "numpy": env["numpy"],
        "scipy": env["scipy"],
    }


def convergence_data(records):
    """Retain every method and termination status, including missing histories."""
    if not records:
        raise ValueError("Supply matched pilot records for all three methods")
    shared = identity(records[0])
    rows = {}
    for record in records:
        if record.get("schema") != "coupled-optimization-pilot-v1" or identity(record) != shared:
            raise ValueError("Pilots must share their problem, numerical source and settings")
        methods = record["configuration"]["methods"]
        if len(methods) != 1 or methods[0] not in METHODS or methods[0] in rows:
            raise ValueError("Supply one distinct, single-method pilot for each solver")
        method = methods[0]
        results = [r for r in record.get("results", []) if r["method"] == method]
        if len(results) > 1:
            raise ValueError("A pilot must contain at most one result for its solver")
        if not results:
            rows[method] = {
                "method": method,
                "status": record["status"],
                "history_available": False,
                "iteration": [],
                "cumulative_inner_iterations": [],
                "maximum_kkt": [],
            }
            continue
        result = results[0]
        final_kkt = maximum_kkt(result["kkt"])
        if (
            result["status"] == "converged"
            and final_kkt > record["configuration"]["nonlinear_tolerance"]
        ):
            raise ValueError("A converged status must satisfy the declared optimality tolerance")
        history = result["history"]
        indices = [r["iteration"] for r in history]
        if indices != list(range(len(history))):
            raise ValueError("The nonlinear history must retain every iteration in order")
        work = [0]
        for row in history:
            count = 0
            for attempt in row["attempts"]:
                for step in attempt["qp_history"]:
                    value = step.get("linear_iterations", 0)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError("Inner iteration counts must be nonnegative integers")
                    count += value
            work.append(work[-1] + count)
        rows[method] = {
            "method": method,
            "status": result["status"],
            "optimizer_status": result["optimizer_status"],
            "history_available": True,
            "iteration": indices + [len(history)],
            "cumulative_inner_iterations": work,
            "maximum_kkt": [maximum_kkt(r["kkt"]) for r in history] + [final_kkt],
        }
    if set(rows) != set(METHODS):
        raise ValueError("Retain all three solver outcomes in the comparison")
    tolerance = shared["configuration"]["nonlinear_tolerance"]
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("The declared optimality tolerance must be positive and finite")
    return {
        "schema": "coupled-pilot-convergence-v1",
        "matched_protocol": shared,
        "methods": [rows[m] for m in METHODS],
        "optimality_tolerance": tolerance,
        "scope": "One pilot per method. Each history entry precedes its nonlinear attempt; the final point uses the returned state's residual. Inner work includes unsuccessful quadratic attempts and residual corrections. This figure measures convergence and iteration work, not complete elapsed time or physical resolution.",
    }


def plot_convergence(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tolerance = report["optimality_tolerance"]
    display_floor = tolerance * 1e-3
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8}):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), sharey=True, layout="constrained")
        for row, style in zip(report["methods"], ("--", "-", ":"), strict=True):
            method = row["method"]
            label = LABELS[method]
            if row["status"] != "converged":
                label += " (" + row["status"].replace("_", " ") + ")"
            for axis, key in zip(axes, ("iteration", "cumulative_inner_iterations"), strict=True):
                if row["history_available"]:
                    values = np.maximum(row["maximum_kkt"], display_floor)
                    axis.semilogy(
                        row[key],
                        values,
                        style,
                        color=COLORS[method],
                        label=label,
                        marker="o",
                        markersize=3,
                        linewidth=1.3,
                    )
                    if row["status"] != "converged":
                        axis.plot(row[key][-1], values[-1], "x", color=COLORS[method], ms=7)
                else:
                    axis.plot([], [], style, color=COLORS[method], label=label + "; no history")
        for axis in axes:
            axis.axhline(tolerance, color="black", ls="-.", lw=0.8, label="Final tolerance")
            axis.grid(True, which="major", alpha=0.2)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            axis.set_yscale("log")
        axes[0].set_xlabel("Nonlinear iteration")
        last_iteration = max(
            (max(row["iteration"], default=0) for row in report["methods"]), default=0
        )
        axes[0].set_xticks(np.unique(np.rint(np.linspace(0, last_iteration, 4)).astype(int)))
        axes[1].set_xlabel("Cumulative inner CG iterations")
        axes[0].set_ylabel("Maximum KKT residual")
        axes[0].legend(loc="best")
        fig.savefig(output / "convergence.pdf")
        fig.savefig(output / "convergence.png", dpi=220)
        plt.close(fig)
    write_report(
        output / "convergence.json",
        {
            **report,
            "display_floor": display_floor,
            "display_note": "Residuals below the display floor appear at that floor; stored numerical values remain unchanged.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [r / "record.json" for r in args.runs]
    report = convergence_data([json.loads(p.read_text()) for p in paths])
    report["input_records"] = [{"run": p.parent.name, "sha256": file_digest(p)} for p in paths]
    plot_convergence(report, args.output)


if __name__ == "__main__":
    main()
