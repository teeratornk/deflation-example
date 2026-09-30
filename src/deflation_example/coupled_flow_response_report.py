"""Summarize every outcome of a matched local momentum-response diagnostic."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .reporting import environment, file_sha256, write_report


def summarize(records):
    if not records:
        raise ValueError("Supply at least one response record")
    identity, slabs, rows = None, set(), []
    expected = [(name, 0.0, False) for name in ("retained", "preceding_time", "preceding_target")]
    expected += [("retained", sign * h, False) for h in (1e-4, 1e-6, 1e-8) for sign in (-1, 1)]
    expected += [("retained", sign * 1e-6, True) for sign in (-1, 1)]
    for record in records:
        if (
            record.get("schema") != "coupled-local-flow-response-v1"
            or record.get("status") != "complete"
        ):
            raise ValueError("A completed response diagnostic is required")
        key = (
            record["record_sha256"],
            tuple(record["fields_sha256"]),
            record["position"],
            record["target"],
            record["environment"]["git_head"],
            record["budget_seconds_per_case"],
        )
        if identity is not None and identity != key:
            raise ValueError("The selected records change the matched source or configuration")
        identity = key
        slab = record["slab_zero_based"]
        if slab in slabs:
            raise ValueError("Duplicate slab in the response summary")
        slabs.add(slab)
        if [
            (r["initial_guess"], r["step_K"], r["continuation"]) for r in record["cases"]
        ] != expected:
            raise ValueError("The response population differs from the declared cases")
        cases = []
        for case in record["cases"]:
            if case["status"] in {"running", "diagnostic_error"}:
                raise ValueError("Every response must have a numerical terminal status")
            if case["verified"] and case["status"] != "converged":
                raise ValueError("Only a converged numerical result can be verified")
            row = {k: v for k, v in case.items() if k != "history"}
            step = case["step_K"]
            row["measured_velocity_response_m_s_per_K"] = (
                case["velocity_change_norm_m_s"] / abs(step) if case["verified"] and step else None
            )
            if row["measured_velocity_response_m_s_per_K"] is not None and not np.isfinite(
                row["measured_velocity_response_m_s_per_K"]
            ):
                raise ValueError("A verified response must have a finite measurement")
            cases.append(row)
        rows.append(
            {
                "slab_zero_based": slab,
                "tangent": record["tangent"],
                "cases": cases,
                "statuses": dict(Counter(r["status"] for r in cases)),
                "verified_count": sum(r["verified"] for r in cases),
            }
        )
    return {
        "schema": "coupled-local-flow-response-summary-v1",
        "status": "complete",
        "source_record_sha256": identity[0],
        "diagnostic_source": identity[4],
        "case_count": sum(len(r["cases"]) for r in rows),
        "verified_count": sum(r["verified_count"] for r in rows),
        "slabs": sorted(rows, key=lambda r: r["slab_zero_based"]),
        "scope": "Fixed-predecessor single-slab momentum response. The velocity measure is the Euclidean nodal norm, not a spatially weighted physical norm. Failed solves have no sensitivity measurement. This diagnostic neither establishes a trajectory optimum nor a complete-solve speedup.",
    }


def plot(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(summary["slabs"])
    fig, axes = plt.subplots(1, count, figsize=(4.4 * count, 4.7), squeeze=False, sharey=True)
    positive = [row["tangent"]["velocity_derivative_norm_m_s_per_K"] for row in summary["slabs"]]
    positive += [
        case["measured_velocity_response_m_s_per_K"]
        for row in summary["slabs"]
        for case in row["cases"]
        if case["measured_velocity_response_m_s_per_K"] is not None
    ]
    positive = [value for value in positive if value > 0]
    if not positive:
        raise ValueError("A logarithmic response plot requires a positive response")
    limits = (10 ** np.floor(np.log10(min(positive))), 10 ** np.ceil(np.log10(max(positive))))
    if limits[0] == limits[1]:
        limits = (limits[0] / 10, limits[1] * 10)
    for ax, row in zip(axes[0], summary["slabs"], strict=True):
        failures = []
        for sign, color, label in (
            (-1, "#0072B2", "Negative perturbation"),
            (1, "#D55E00", "Positive perturbation"),
        ):
            cases = sorted(
                (
                    c
                    for c in row["cases"]
                    if c["initial_guess"] == "retained"
                    and not c["continuation"]
                    and sign * c["step_K"] > 0
                ),
                key=lambda c: abs(c["step_K"]),
            )
            # NaNs break the line at an unsuccessful solve; they never become a
            # zero response or an interpolated finite measurement.
            ax.loglog(
                [abs(c["step_K"]) for c in cases],
                [
                    c["measured_velocity_response_m_s_per_K"] if c["verified"] else np.nan
                    for c in cases
                ],
                "o-",
                color=color,
                label=label,
            )
            failures.extend(
                f"{c['step_K']:+.0e} K: {c['status'].replace('_', ' ')}"
                for c in cases
                if not c["verified"]
            )
        ax.axhline(
            row["tangent"]["velocity_derivative_norm_m_s_per_K"],
            color="black",
            linestyle="--",
            linewidth=1,
            label="Analytic tangent",
        )
        ax.set_title(f"Time slab {row['slab_zero_based'] + 1}")
        ax.set_ylim(*limits)
        ax.set_xlabel("Maximum temperature perturbation (K)")
        ax.set_xticks([1e-8, 1e-6, 1e-4])
        ax.grid(alpha=0.2, which="both")
        ax.text(
            0,
            -0.26,
            "\n".join(failures) if failures else "All six direct perturbations converged.",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
        )
    axes[0, 0].set_ylabel(r"Nodal velocity response $\|\Delta v\|_2/|h|$ (m s$^{-1}$ K$^{-1}$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.84, bottom=0.30, wspace=0.34)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"local_momentum_response.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    records = []
    for path in args.records:
        record = json.loads(path.read_text())
        if file_sha256(path.parent / "inputs.npz") != record["inputs_sha256"]:
            raise ValueError("The saved direction or tangent changed")
        for row in record["cases"]:
            if file_sha256(path.parent / f"case-{row['index']:02d}.npz") != row["field_sha256"]:
                raise ValueError("A saved terminal momentum field changed")
        records.append(record)
    summary = summarize(records)
    summary["report_environment"] = environment()
    summary["inputs"] = [
        {"file": f"{p.parent.name}/{p.name}", "sha256": file_sha256(p)} for p in args.records
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", summary)
    if args.plot:
        plot(summary, args.output)


if __name__ == "__main__":
    main()
