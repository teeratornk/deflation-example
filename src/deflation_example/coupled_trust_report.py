"""Summarize all completed or interrupted trust-region diagnostic outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_recovery import RecoveryStore, identity
from .reporting import file_sha256, write_report


def summarize(path):
    path = Path(path)
    record = json.loads(path.read_text())
    if record.get("schema") != "coupled-trust-development-v1":
        raise ValueError("Use a separately versioned trust-region record")
    cfg = record["configuration"]
    cases = record.get("cases", [])
    histories = [row for c in cases for row in c["history"]]
    checkpoint = RecoveryStore(
        path.parent / "recovery", identity(cfg, record["environment"]["source_sha256"])
    ).load()
    if (
        record["status"] == "running"
        and checkpoint is not None
        and checkpoint["optimizer"] is not None
    ):
        opt = checkpoint["optimizer"]
        histories = opt["history"] + [
            {
                "iteration": opt["iteration"],
                "objective": opt["objective"],
                "kkt": opt["kkt"],
                "attempts": opt["attempts"],
            }
        ]
        if opt["qp"] is not None:
            histories[-1]["attempts"] = list(histories[-1]["attempts"]) + [
                {"qp_history": opt["qp"]["history"], "trials": []}
            ]
    attempts = [a for row in histories for a in row["attempts"]]
    steps = [s for a in attempts for s in a["qp_history"]]
    series = []
    cumulative = 0.0
    for row in histories:
        cumulative += sum(
            s.get("timing", {}).get("total_seconds", 0)
            for a in row["attempts"]
            for s in a["qp_history"]
        )
        last = row["attempts"][-1] if row["attempts"] else {}
        series.append(
            {
                "iteration": row["iteration"],
                "objective_before": row["objective"],
                "kkt_before": max(row["kkt"].values()),
                "radius_K": last.get("radius_K"),
                "inner_seconds_cumulative": cumulative,
            }
        )
    return {
        "record_sha256": file_sha256(path),
        "status": record["status"],
        "accuracy": cfg["trust_accuracy"],
        "continuation": cfg["flow_continuation"],
        "method": cfg["method"],
        "inner_preconditioner": cfg["inner_preconditioner"],
        "resumed": record["resumed"],
        "all_problems_verified": record.get("all_problems_verified", False),
        "attempt_seconds": record.get("attempt_seconds"),
        "cumulative_attempt_seconds": record.get("cumulative_attempt_seconds"),
        "verified_problems": record.get("verified_problems", 0),
        "outer_history_rows": len(histories),
        "active_set_solves": len(steps),
        "inner_iterations": sum(s.get("linear_iterations", 0) for s in steps),
        "inner_seconds": cumulative,
        "flow_trial_failures": sum(
            t["status"].startswith("flow_") for a in attempts for t in a["trials"]
        ),
        "maximum_inner_original_residual": max(
            (s.get("linear_residual", 0.0) for s in steps), default=None
        ),
        "series": series,
        "scope": "Development diagnostics; inner intervals include CPU coarse processing. Partial progress and solver failures are retained. No speedup follows from incomplete comparisons.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    rows = [summarize(path) for path in args.records]
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", {"schema": "coupled-trust-summary-v1", "runs": rows})
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
        for row in rows:
            data = row["series"]
            if not data:
                continue
            label = (
                f"{row['method']}/{row['inner_preconditioner']}, {row['accuracy']}, {row['status']}"
            )
            x = np.arange(len(data))
            for ax, field in zip(
                axes.flat,
                ("objective_before", "kkt_before", "radius_K", "inner_seconds_cumulative"),
                strict=True,
            ):
                y = [float("nan") if d[field] is None else d[field] for d in data]
                ax.plot(x, y, marker=".", label=label)
                ax.set_xlabel("Recorded nonlinear step")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for ax, label in zip(
            axes.flat,
            (
                "Scaled objective",
                "Maximum physical KKT residual",
                "Last attempted temperature radius (K)",
                "Cumulative inner time (s)",
            ),
            strict=True,
        ):
            ax.set_ylabel(label)
        axes[0, 1].set_yscale("log")
        axes[0, 1].axhline(1e-8, color="black", linestyle=":", linewidth=1)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Coupled trust-region development diagnostics")
        fig.savefig(args.output / "convergence.pdf")
        fig.savefig(args.output / "convergence.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
