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
LABELS = {
    "newton": "Newton",
    "relaxed025": r"Relaxed, $\omega=0.25$",
    "relaxed050": r"Relaxed, $\omega=0.5$",
    "relaxed100": r"Relaxed, $\omega=1$",
    "anderson3": "Anderson, depth 3",
    "anderson5": "Anderson, depth 5",
}


def residual_trace(history):
    """Plot attempted iterates; restoring a retained state is a separate event."""
    iterations, residuals = [], []
    last = 0
    for row in history:
        if row.get("procedure") == "returned_state_verification":
            continue
        last = row.get("coupling_iteration", row.get("iteration", last))
        values = [
            row[key]
            for key in (
                "momentum_relative_residual",
                "continuity_relative_residual",
                "thermal_relative_residual",
            )
            if row.get(key) is not None
        ]
        if values:
            iterations.append(last)
            residuals.append(max(values))
    return iterations, residuals


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
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    for family in ("forward", "momentum"):
        case_times = {
            p.parent.parent.name: r["time_s"]
            for p, r in records
            if "row" in r and r["family"] == family
        }
        cases = sorted(case_times, key=lambda case: (case_times[case], case))
        if not cases:
            continue
        columns = min(3, len(cases))
        nrows = (len(cases) + columns - 1) // columns
        fig, axes = plt.subplots(
            nrows, columns, figsize=(3.4 * columns, 3.1 * nrows), squeeze=False, sharey=True
        )
        handles = {}
        returned_marker_present = False
        for ax, case in zip(axes.ravel(), cases):
            physical_time = None
            for path, record in records:
                if (
                    record["family"] != family
                    or path.parent.parent.name != case
                    or "row" not in record
                ):
                    continue
                physical_time = record["time_s"]
                iterations, residuals = residual_trace(record["row"]["history"])
                if residuals:
                    (line,) = ax.semilogy(
                        iterations,
                        np.maximum(residuals, 1e-16),
                        color=COLORS[record["policy"]],
                        label=LABELS[record["policy"]],
                        marker="o" if len(residuals) == 1 else None,
                        markersize=4,
                    )
                    handles[record["policy"]] = line
                    if not record["row"]["verified"]:
                        ax.plot(
                            iterations[-1],
                            max(residuals[-1], 1e-16),
                            "x",
                            color=COLORS[record["policy"]],
                            markersize=7,
                        )
                        checks = record["row"].get("checks", {})
                        returned_values = [
                            checks[key]
                            for key in (
                                "momentum_relative_residual",
                                "continuity_relative_residual",
                                "thermal_relative_residual",
                            )
                            if checks.get(key) is not None
                        ]
                        returned = max(returned_values) if returned_values else None
                        if returned is not None and not np.isclose(
                            returned, residuals[-1], rtol=1e-8, atol=0
                        ):
                            # This is a return event at termination, not another
                            # nonlinear update or an inferred earlier iterate.
                            final_index = max(
                                row.get("coupling_iteration", row.get("iteration", 0))
                                for row in record["row"]["history"]
                            )
                            ax.plot(
                                final_index,
                                max(returned, 1e-16),
                                marker="D",
                                linestyle="none",
                                markerfacecolor="none",
                                color=COLORS[record["policy"]],
                                markersize=6,
                            )
                            returned_marker_present = True
            ax.axhline(1e-12, color="0.7", linewidth=0.7, linestyle="--")
            scheme = "BDF2" if case.startswith("bdf2") else "Backward Euler"
            ax.set_title(f"{scheme}, {physical_time:.8g} s")
            ax.set_xlabel("Nonlinear iteration")
            ax.set_xlim(left=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=1))
        for ax in axes[:, 0]:
            ax.set_ylabel("Maximum original-\nequation residual")
        for ax in axes.ravel()[len(cases) :]:
            ax.set_visible(False)
        if handles:
            ordered = [handles[policy] for policy in COLORS if policy in handles]
            if returned_marker_present:
                ordered.extend(
                    [
                        Line2D(
                            [],
                            [],
                            color="0.25",
                            marker="x",
                            linestyle="none",
                            label="Last recorded iterate (failed solve)",
                        ),
                        Line2D(
                            [],
                            [],
                            color="0.25",
                            marker="D",
                            markerfacecolor="none",
                            linestyle="none",
                            label="Returned state",
                        ),
                    ]
                )
            fig.legend(
                ordered,
                [line.get_label() for line in ordered],
                loc="lower center",
                ncol=3,
                fontsize=9,
            )
        bottom = (
            (0.2 if nrows == 1 else 0.13)
            if returned_marker_present
            else (0.12 if nrows == 1 else 0.08)
        )
        fig.tight_layout(rect=(0, bottom, 1, 1))
        fig.savefig(output / f"{family}-residuals.pdf")
        fig.savefig(output / f"{family}-residuals.png", dpi=160)
        plt.close(fig)
    trajectories = [(p, r) for p, r in records if "steps" in r]
    if trajectories:
        definitions = {
            (r["configuration"]["upper_K"], r["temperature_offset_K"], r["temperature_scale_K"])
            for _, r in trajectories
        }
        if len(definitions) != 1:
            raise ValueError("Temperature plots require matched bounds and temperature units")
        fig, ax = plt.subplots(figsize=(7, 4))
        for path, record in trajectories:
            times, peaks = [], []
            for row in record["steps"]:
                if not row["verified"]:
                    break
                field_path = path.parent / row["fields"]
                if (
                    field_path.resolve().parent != path.parent.resolve()
                    or file_sha256(field_path) != row["field_sha256"]
                ):
                    raise ValueError("Trajectory field checksum or location differs")
                with np.load(field_path, allow_pickle=False) as data:
                    # The frozen transformer normalization is read from model metadata.
                    peak = record["temperature_offset_K"] + record["temperature_scale_K"] * float(
                        data["state"].max()
                    )
                times.append(row["time_s"])
                peaks.append(peak)
            if not times:
                continue
            ax.plot(
                times,
                peaks,
                color=COLORS[record["policy"]],
                alpha=0.5,
                linestyle="-" if record["status"] == "converged" else "--",
                label=f"{LABELS[record['policy']]}, {record['slabs']} slabs"
                if record["repetition"] == 0
                else None,
            )
        upper = trajectories[0][1]["configuration"]["upper_K"]
        ax.axhline(upper, linestyle="--", color="black", label="Upper temperature bound")
        ax.set(xlabel="Physical time (s)", ylabel="Maximum nodal temperature (K)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "temperature-bound.pdf")
        fig.savefig(output / "temperature-bound.png", dpi=160)
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
