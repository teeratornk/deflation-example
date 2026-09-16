"""Compare complete fixed-control trajectories on the same discrete problem.

This diagnostic checks saved-field identities and reported equation criteria.
It measures solver agreement, not discretization error, uniqueness or speedup.
Run it on a compute node after both trajectories have finished.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from deflation_example.reporting import file_sha256, write_report


IDENTITY_KEYS = (
    "optimization_field_sha256",
    "baseline_sha256",
    "fine_baseline_sha256",
    "configuration",
    "control_policy",
    "slabs",
    "spatial_state_dofs",
    "subdivision",
    "time_scheme",
    "temperature_offset_K",
    "temperature_scale_K",
    "tolerance",
    "mass_tolerance",
    "energy_tolerance",
)
CRITERIA = {
    "momentum_relative_residual": "tolerance",
    "continuity_relative_residual": "tolerance",
    "thermal_relative_residual": "tolerance",
    "mass_relative_imbalance": "mass_tolerance",
    "energy_relative_defect": "energy_tolerance",
}


def read_complete(folder):
    path = folder / "record.json"
    digest = file_sha256(path)
    record = json.loads(path.read_text())
    if (
        record.get("schema") != "coupled-fixed-point-study-v1"
        or record.get("family") != "forward"
        or record.get("status") != "converged"
        or not record.get("steps")
        or len(record["steps"]) != record.get("slabs")
    ):
        raise ValueError("Agreement requires complete, converged forward trajectories")
    if not record["environment"].get("source_sha256"):
        raise ValueError("Numerical source identity is required")
    for key in IDENTITY_KEYS:
        if key not in record or record[key] is None:
            raise ValueError("Missing trajectory identity: " + key)
    for key in ("temperature_scale_K", "tolerance", "mass_tolerance", "energy_tolerance"):
        if not math.isfinite(record[key]) or record[key] <= 0:
            raise ValueError("Invalid positive physical scale or accuracy criterion")
    if not math.isfinite(record["temperature_offset_K"]):
        raise ValueError("Invalid temperature offset")
    previous_time = 0.0
    for index, row in enumerate(record["steps"]):
        if row.get("verified") is not True or row.get("slab_zero_based") != index:
            raise ValueError("Every trajectory step must be verified and ordered")
        physical_time = row["time_s"]
        if not math.isfinite(physical_time) or physical_time <= previous_time:
            raise ValueError("Physical times must be finite, positive and increasing")
        previous_time = physical_time
        for criterion, threshold in CRITERIA.items():
            value = row.get("checks", {}).get(criterion)
            if value is None or not math.isfinite(value) or not 0 <= value <= record[threshold]:
                raise ValueError("A trajectory step fails its recorded criterion: " + criterion)
    horizon = record["configuration"]["horizon_s"]
    if not math.isfinite(horizon) or not math.isclose(previous_time, horizon, rel_tol=1e-12):
        raise ValueError("The completed trajectory must reach its physical horizon")
    return record, digest


def load_fields(folder, row, dofs):
    path = folder / row["fields"]
    if path.resolve().parent != folder.resolve() or file_sha256(path) != row["field_sha256"]:
        raise ValueError("Trajectory field checksum or location differs")
    with np.load(path, allow_pickle=False) as data:
        state, velocity = data["state"].copy(), data["velocity"].copy()
    if (
        state.shape != (dofs,)
        or velocity.ndim != 2
        or velocity.shape[1] != 2
        or not velocity.shape[0]
        or not np.isrealobj(state)
        or not np.isrealobj(velocity)
        or not np.isfinite(state).all()
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("Fields require finite real state and two-component velocity arrays")
    return state, velocity


def compare(baseline_folder, candidate_folder):
    baseline, baseline_digest = read_complete(baseline_folder)
    candidate, candidate_digest = read_complete(candidate_folder)
    for key in IDENTITY_KEYS:
        if baseline[key] != candidate[key]:
            raise ValueError("Trajectory problems differ: " + key)
    if baseline["environment"]["source_sha256"] != candidate["environment"]["source_sha256"]:
        raise ValueError("Trajectory numerical sources differ")
    rows = []
    for left, right in zip(baseline["steps"], candidate["steps"]):
        if left["time_s"] != right["time_s"]:
            raise ValueError("Trajectory time levels differ")
        a, va = load_fields(baseline_folder, left, baseline["spatial_state_dofs"])
        b, vb = load_fields(candidate_folder, right, candidate["spatial_state_dofs"])
        if va.shape != vb.shape:
            raise ValueError("Trajectory velocity dimensions differ")
        delta = np.abs(a - b) * baseline["temperature_scale_K"]
        velocity_delta = np.linalg.norm(va - vb, axis=1)
        if not np.isfinite(delta).all() or not np.isfinite(velocity_delta).all():
            raise ValueError("Physical field differences must remain finite")
        rows.append(
            {
                "slab_zero_based": left["slab_zero_based"],
                "time_s": left["time_s"],
                "maximum_temperature_difference_K": float(delta.max()),
                "temperature_state_index": int(delta.argmax()),
                "maximum_velocity_vector_difference_m_s": float(velocity_delta.max()),
                "velocity_node_index": int(velocity_delta.argmax()),
                "baseline_field_sha256": left["field_sha256"],
                "candidate_field_sha256": right["field_sha256"],
            }
        )
    for folder, digest in (
        (baseline_folder, baseline_digest),
        (candidate_folder, candidate_digest),
    ):
        if file_sha256(folder / "record.json") != digest:
            raise ValueError("A trajectory record changed during comparison")
    temperature = max(rows, key=lambda row: row["maximum_temperature_difference_K"])
    velocity = max(rows, key=lambda row: row["maximum_velocity_vector_difference_m_s"])
    return {
        "baseline_policy": baseline["policy"],
        "baseline_repetition": baseline["repetition"],
        "candidate_policy": candidate["policy"],
        "candidate_repetition": candidate["repetition"],
        "baseline_record_sha256": baseline_digest,
        "candidate_record_sha256": candidate_digest,
        "optimization_field_sha256": baseline["optimization_field_sha256"],
        "source_sha256": baseline["environment"]["source_sha256"],
        "slabs": baseline["slabs"],
        "maximum_temperature_difference_K": temperature["maximum_temperature_difference_K"],
        "temperature_maximum_time_s": temperature["time_s"],
        "maximum_velocity_vector_difference_m_s": velocity[
            "maximum_velocity_vector_difference_m_s"
        ],
        "velocity_maximum_time_s": velocity["time_s"],
        "steps": rows,
    }


def plot(comparisons, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    names = {"newton": "Newton", "anderson5": "Anderson, depth 5"}
    for comparison in comparisons:
        rows = comparison["steps"]
        label = (
            f"{names.get(comparison['candidate_policy'], comparison['candidate_policy'])}, "
            f"repetition {comparison['candidate_repetition'] + 1}"
        )
        for axis, key in zip(
            axes,
            ("maximum_temperature_difference_K", "maximum_velocity_vector_difference_m_s"),
        ):
            axis.plot([row["time_s"] for row in rows], [row[key] for row in rows], label=label)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    for axis, key in zip(
        axes,
        ("maximum_temperature_difference_K", "maximum_velocity_vector_difference_m_s"),
    ):
        maximum = max(row[key] for comparison in comparisons for row in comparison["steps"])
        axis.set_ylim(0, 1.15 * maximum if maximum else 1.0)
        if maximum == 0:
            axis.text(0.5, 0.5, "Identical stored fields", ha="center", transform=axis.transAxes)
    baseline = comparisons[0]
    axes[0].set_title(
        f"Difference from {names.get(baseline['baseline_policy'], baseline['baseline_policy'])} "
        f"(repetition {baseline['baseline_repetition'] + 1})"
    )
    axes[0].set_ylabel("Maximum temperature\ndifference (K)")
    axes[1].set_ylabel("Maximum velocity-vector\ndifference (m/s)")
    axes[1].set_xlabel("Physical time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    for extension in ("pdf", "png"):
        fig.savefig(output / f"trajectory-agreement.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    comparisons = [compare(args.baseline, folder) for folder in args.candidate]
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(
        args.output / "agreement.json",
        {
            "schema": "fixed-point-trajectory-agreement-v1",
            "diagnostic_source_sha256": file_sha256(Path(__file__)),
            "comparisons": comparisons,
            "scope": "Agreement at every stored time level for completed fixed-source trajectories on the same mesh and time grid. Stored field hashes and reported equation criteria are checked. No equations are re-solved. This is not a uniqueness, physical-resolution or optimization-speedup certificate.",
        },
    )
    if not args.no_plot:
        plot(comparisons, args.output)


if __name__ == "__main__":
    main()
