"""Summarize both branch restarts, retaining unsuccessful outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np

from .reporting import environment, file_sha256, write_report


def summarize(records):
    if len(records) != 2:
        raise ValueError("Supply both matched branch restarts")
    common, names, rows = None, set(), []
    for record in records:
        branch = record.get("initial_branch") or {}
        name = branch.get("selected")
        if (
            record.get("schema") != "coupled-radius-restart-v1"
            or record.get("status") in {None, "running"}
            or name not in {"retained", "alternate_seed"}
            or name in names
        ):
            raise ValueError("Supply distinct terminal retained and alternative restarts")
        names.add(name)
        key = [
            record[k]
            for k in (
                "source_configuration",
                "record_sha256",
                "field_sha256",
                "position",
                "target",
                "restart",
            )
        ] + [
            branch["assessment_sha256"],
            branch["trajectory_sha256"],
            record["environment"]["source_sha256"],
        ]
        if common is not None and common != key:
            raise ValueError("The two restarts must have matching inputs, source and policies")
        common = key
        if record.get("gpu_gate_passed") and not (
            record.get("verified")
            and record["status"] == "converged"
            and record.get("independent_derivatives", {}).get("derivatives_passed")
        ):
            raise ValueError("A GPU gate requires verified convergence and derivative checks")
        points = [
            {"objective": r["objective"], "kkt": max(r["kkt"].values())}
            for r in record.get("history", [])
        ]
        if "objective" in record and "kkt" in record:
            final = {"objective": record["objective"], "kkt": max(record["kkt"].values())}
            if not points or points[-1] != final:
                points.append(final)
        if points and not np.isfinite([[p["objective"], p["kkt"]] for p in points]).all():
            raise ValueError("Retained optimization metrics must be finite")
        rows.append(
            {
                "branch": name,
                "status": record["status"],
                "verified": record.get("verified", False),
                "gpu_gate_passed": record.get("gpu_gate_passed", False),
                "initial_verification": record.get("initial_verification"),
                "optimizer_initial_branch": record.get("optimizer_initial_branch"),
                "objective": record.get("objective"),
                "kkt": record.get("kkt"),
                "stationarity_numerator": record.get("stationarity_numerator"),
                "optimizer_seconds": record.get("optimizer_seconds"),
                "total_seconds": record.get("seconds"),
                "retained_states": points,
                "error": record.get("error"),
            }
        )
    return {
        "schema": "coupled-branch-restart-summary-v1",
        "target": records[0]["target"],
        "restart": records[0]["restart"],
        "nonlinear_tolerance": records[0]["source_configuration"]["nonlinear_tolerance"],
        "branches": sorted(rows, key=lambda row: row["branch"] != "retained"),
        "alternative_gpu_gate_passed": next(
            r["gpu_gate_passed"] for r in rows if r["branch"] == "alternate_seed"
        ),
        "scope": "Two optimizer restarts from identical temperatures and separately verified flow branches. Times exclude the preceding sequence and branch diagnosis. No complete-sequence speedup is inferred.",
    }


def plot(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    statuses = []
    for row, color in zip(summary["branches"], ("#0072B2", "#D55E00"), strict=True):
        label = "Retained flow" if row["branch"] == "retained" else "Alternative flow"
        statuses.append(f"{label}: {row['status']}")
        points = row["retained_states"]
        if not points:
            continue
        x = np.arange(len(points))
        axes[0].semilogy(
            x,
            [max(p["kkt"], np.finfo(float).tiny) for p in points],
            "o-",
            color=color,
            label=label,
            markersize=4,
        )
        axes[1].plot(
            x,
            [points[0]["objective"] - p["objective"] for p in points],
            "o-",
            color=color,
            label=label,
            markersize=4,
        )
    axes[0].axhline(
        summary["nonlinear_tolerance"], color="0.4", linestyle="--", label="Final criterion"
    )
    axes[0].set_ylabel("Maximum KKT component")
    axes[1].set_ylabel("Initial minus current normalized objective")
    for ax in axes:
        ax.set_xlabel("Retained-state index")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Target {summary['target']}: matched flow-branch restarts")
    fig.text(0.5, 0.015, " | ".join(statuses), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"branch_restart.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    records = []
    for path in args.records:
        record = json.loads(path.read_text())
        if (
            "result_sha256" in record
            and file_sha256(path.parent / "result.npz") != record["result_sha256"]
        ):
            raise ValueError("The returned-field checksum differs")
        records.append(record)
    summary = summarize(records)
    summary["environment"] = environment()
    summary["record_sha256"] = [file_sha256(path) for path in args.records]
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", summary)
    if args.plot:
        plot(summary, args.output)


if __name__ == "__main__":
    main()
