"""Locate temperature differences in matched, fixed-source time refinements.

This diagnostic supplements the all-time resolution assessment. It reports
temperature histories at the largest difference and velocity differences at
that same physical time. Neither field is shifted in time to improve agreement.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_pilot import transformer_inputs
from .coupled_saved import file_digest
from .coupled_time_resolution_report import summarize
from .reporting import environment, write_report
from .validation import integer


def interpolate_snapshot(values, times, time, initial):
    """Interpolate at a physical time without extrapolation or time alignment."""
    times = np.asarray(times, dtype=float)
    if (
        len(values) != len(times)
        or not len(times)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or times[0] <= 0
        or not np.isfinite(time)
        or not 0 <= time <= times[-1]
    ):
        raise ValueError("Snapshot time must lie inside a strictly increasing trajectory")
    n = int(np.searchsorted(times, time, side="left"))
    if time == times[n]:
        return np.array(values[n], copy=True)
    start = 0.0 if n == 0 else times[n - 1]
    previous = initial if n == 0 else values[n - 1]
    fraction = (time - start) / (times[n] - start)
    return (1 - fraction) * previous + fraction * values[n]


def diagnose(directories, baseline_directory):
    baseline_directory = Path(baseline_directory)
    baseline = json.loads((baseline_directory / "record.json").read_text())
    if baseline["status"] != "converged":
        raise ValueError("The reference flow must have converged")
    baseline_fields = baseline_directory / "baseline.npz"
    if file_digest(baseline_fields) != baseline["baseline_sha256"]:
        raise ValueError("Baseline field checksum differs")
    mesh, parameters, _, _, _, hashes = transformer_inputs(baseline["configuration"]["level"])
    if hashes != baseline["input_sha256"]:
        raise ValueError("The mesh and material data differ from the recorded inputs")
    physical = parameters["physical"]
    scale, offset = physical["temperature_scale_K"], physical["inlet_temperature_K"]
    # The declared assessment starts at the uniform inlet temperature.
    summary = summarize(directories, scale, initial_value=0.0)
    paths = {Path(directory).name: Path(directory) for directory in directories}
    with np.load(baseline_fields, allow_pickle=False) as fields:
        initial_velocity = fields["velocity"].copy()
    measurements, snapshots = [], []
    for pair in summary["pairs"]:
        coarse_path, fine_path = paths[pair["coarse_case"]], paths[pair["fine_case"]]
        record = json.loads((fine_path / "record.json").read_text())
        if record["baseline_sha256"] != baseline["baseline_sha256"]:
            raise ValueError("The replay uses a different baseline")
        peak = pair["all_refined_time_peak"]
        node, time = peak["free_node_index"], peak["time_s"]
        states, velocities = [], []
        for path in (coarse_path, fine_path):
            with np.load(path / "states.npz", allow_pickle=False) as fields:
                times = fields["times_s"]
                state = fields["state"]
                if state.shape[1] != len(mesh.free):
                    raise ValueError("Temperature columns must match the baseline free nodes")
                states.append(interpolate_snapshot(state, times, time, np.zeros(len(mesh.free))))
                velocity = fields["velocity"]
                if velocity.shape != (len(times), *initial_velocity.shape):
                    raise ValueError("Velocity fields must match the baseline and physical times")
                if not np.isfinite(velocity).all():
                    raise ValueError("Velocity fields must be finite")
                velocities.append(interpolate_snapshot(velocity, times, time, initial_velocity))
        temperature_difference = scale * (states[1] - states[0])
        velocity_difference = velocities[1] - velocities[0]
        full_node = int(mesh.free[node])
        touching = np.any(mesh.cells == full_node, axis=1)
        measurements.append(
            {
                "coarse_case": pair["coarse_case"],
                "fine_case": pair["fine_case"],
                "coarse_slabs": pair["coarse_slabs"],
                "fine_slabs": pair["fine_slabs"],
                "peak": {
                    **peak,
                    "full_node_index": full_node,
                    "radial_axial_coordinate_m": mesh.nodes[full_node],
                    "adjacent_material_labels": np.unique(mesh.materials[touching]),
                },
                "peak_node_trace": pair["peak_node_trace"],
                "maximum_velocity_difference_at_temperature_peak_m_s": float(
                    np.linalg.norm(velocity_difference, axis=1).max()
                ),
                "coarse_maximum_speed_at_temperature_peak_m_s": float(
                    np.linalg.norm(velocities[0], axis=1).max()
                ),
                "fine_maximum_speed_at_temperature_peak_m_s": float(
                    np.linalg.norm(velocities[1], axis=1).max()
                ),
                "free_nodes_exceeding_temperature_threshold_at_peak": int(
                    np.count_nonzero(np.abs(temperature_difference) > 0.05)
                ),
                "free_nodes": len(mesh.free),
            }
        )
        full_difference = np.zeros(len(mesh.nodes))
        full_difference[mesh.free] = temperature_difference
        snapshots.append(full_difference)
    return (
        {
            "schema": "coupled-resolution-location-diagnostic-v1",
            "environment": environment(),
            "baseline_sha256": baseline["baseline_sha256"],
            "input_sha256": hashes,
            "temperature_scale_K": scale,
            "temperature_offset_K": offset,
            "resolution_assessment": summary["resolution_assessment"],
            "rows": summary["rows"],
            "pairs": measurements,
            "scope": "Fixed signed control, common mesh and physical times. Velocity differences are sampled at each pair's largest temperature difference. They do not establish the cause of that difference. No phase alignment, reoptimization or clipping is applied.",
        },
        mesh,
        snapshots,
    )


def plot_diagnostic(report, mesh, snapshots, output):
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    triangulation = tri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.cells)
    for pair, difference in zip(report["pairs"], snapshots, strict=True):
        figure, axes = plt.subplots(1, 2, figsize=(9, 4.4), layout="constrained")
        maximum = max(float(np.abs(difference).max()), np.finfo(float).eps)
        field = axes[0].tripcolor(
            triangulation, difference, shading="gouraud", cmap="RdBu_r", vmin=-maximum, vmax=maximum
        )
        point = pair["peak"]["radial_axial_coordinate_m"]
        axes[0].plot(*point, "ko", markersize=4)
        axes[0].set(
            xlabel="Radial coordinate (m)",
            ylabel="Axial coordinate (m)",
            title=f"Fine minus coarse at {pair['peak']['time_s']:.2f} s",
        )
        figure.colorbar(field, ax=axes[0], label="Temperature difference (K)")
        trace = pair["peak_node_trace"]
        for name, style in (("coarse", "--"), ("fine", "-")):
            temperatures = report["temperature_offset_K"] + report[
                "temperature_scale_K"
            ] * np.asarray(trace[f"{name}_dimensionless_temperature"])
            axes[1].plot(
                trace[f"{name}_times_s"],
                temperatures,
                style,
                label=f"{pair[f'{name}_slabs']} time levels",
            )
        axes[1].axvline(pair["peak"]["time_s"], color="0.5", linewidth=0.8)
        axes[1].set(xlabel="Physical time (s)", ylabel="Temperature (K)", title="Same spatial node")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        name = f"resolution_location_{pair['coarse_slabs']}_{pair['fine_slabs']}"
        for suffix in ("pdf", "png"):
            figure.savefig(output / f"{name}.{suffix}", dpi=220)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        report, mesh, snapshots = diagnose(args.replays, args.baseline)
        args.output.mkdir(parents=True, exist_ok=False)
        write_report(args.output / "record.json", report)
        if args.plot:
            plot_diagnostic(report, mesh, snapshots, args.output)


if __name__ == "__main__":
    main()
