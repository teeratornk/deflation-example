"""Generate status tables and figures from every retained fixed-point outcome."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from deflation_example.reporting import atomic_output, file_sha256, write_report

COLORS = {
    "newton": "#222222",
    "relaxed025": "#56B4E9",
    "relaxed050": "#0072B2",
    "relaxed100": "#009E73",
    "anderson3": "#D55E00",
    "anderson5": "#CC79A7",
}


def summarize(root, output, plots=True):
    records = []
    for path in sorted(root.glob("**/record.json")):
        record = json.loads(path.read_text())
        if record.get("schema") != "coupled-fixed-point-study-v1":
            continue
        records.append((path, record))
    if not records:
        raise ValueError("No fixed-point records found")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path, record in records:
        local = record.get("row", {})
        checks = local.get("checks", {})
        rows.append(
            {
                "record": str(path.relative_to(root)),
                "sha256": file_sha256(path),
                "family": record["family"],
                "policy": record["policy"],
                "time_s": record.get("time_s"),
                "spatial_dofs": record.get("spatial_state_dofs"),
                "status": local.get("status", record["status"]),
                "verified": local.get("verified", record["status"] == "converged"),
                "step_seconds": local.get("seconds"),
                "sum_step_seconds": record.get("sum_step_seconds"),
                "all_attempt_process_seconds": record.get("all_attempt_process_seconds"),
                "momentum_residual": checks.get("momentum_relative_residual"),
                "thermal_residual": checks.get("thermal_relative_residual"),
                "upper_violation_K": record.get("quantities", {}).get("upper_bound_violation_K"),
            }
        )
    # Field names remain explicit; unknown or unfinished costs stay empty.
    with atomic_output(output / "outcomes.csv") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(
        output / "summary.json",
        {
            "schema": "fixed-point-summary-v1",
            "rows": rows,
            "scope": "All retained local and trajectory outcomes. Local solves do not establish complete optimization speedups.",
        },
    )
    if not plots:
        return rows
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = sorted({p.parent.parent.name for p, r in records if "row" in r})
    for family in ("forward", "momentum"):
        if not cases:
            continue
        fig, axes = plt.subplots(
            1, len(cases), figsize=(3.1 * len(cases), 3.5), squeeze=False, sharey=True
        )
        for ax, case in zip(axes[0], cases):
            for path, record in records:
                if (
                    record["family"] != family
                    or path.parent.parent.name != case
                    or "row" not in record
                ):
                    continue
                residuals = []
                for row in record["row"]["history"]:
                    values = [
                        row[k]
                        for k in (
                            "momentum_relative_residual",
                            "continuity_relative_residual",
                            "thermal_relative_residual",
                        )
                        if row.get(k) is not None
                    ]
                    if values:
                        residuals.append(max(values))
                if residuals:
                    ax.semilogy(
                        np.maximum(residuals, 1e-16),
                        color=COLORS[record["policy"]],
                        label=record["policy"],
                    )
            ax.axhline(1e-12, color="0.7", linewidth=0.7, linestyle="--")
            ax.set_title(case.replace("_", " "))
            ax.set_xlabel("Nonlinear iteration")
        axes[0, 0].set_ylabel("Maximum original-equation residual")
        axes[0, -1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"{family}-residuals.pdf")
        plt.close(fig)
    trajectories = [(p, r) for p, r in records if "steps" in r]
    if trajectories:
        fig, ax = plt.subplots(figsize=(7, 4))
        for path, record in trajectories:
            times, peaks = [], []
            for row in record["steps"]:
                with np.load(path.parent / row["fields"], allow_pickle=False) as data:
                    # The frozen transformer normalization is read from model metadata.
                    peak = record["temperature_offset_K"] + record["temperature_scale_K"] * float(
                        data["state"].max()
                    )
                times.append(row["time_s"])
                peaks.append(peak)
            ax.plot(
                times,
                peaks,
                color=COLORS[record["policy"]],
                alpha=0.5,
                label=f"{record['policy']}, {record['slabs']} slabs"
                if record["repetition"] == 0
                else None,
            )
        upper = trajectories[0][1]["configuration"]["upper_K"]
        ax.axhline(upper, linestyle="--", color="black", label="Upper temperature bound")
        ax.set(xlabel="Physical time (s)", ylabel="Maximum nodal temperature (K)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "temperature-bound.pdf")
        plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)


if __name__ == "__main__":
    main()
