"""Predeclared v2 statistics for nested fixed-source time refinements.

The v1 assessment (pointwise maximum against linear interpolation of the
coarse trajectory) is kept and reported unchanged. This module adds the
amendment's criteria: the tracking-functional change, the mass-weighted
space-time RMS temperature difference, pointwise percentiles with the nodes
that ever exceed declared thresholds and their location, window maxima, the
bound excess with location and time, and a Richardson estimate of the excess
limit. No time window is excluded and nothing is aligned in time.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_pilot import transformer_inputs
from .coupled_saved import file_digest
from .coupled_time_resolution_report import summarize
from .meshes import assemble_thermal
from .reporting import environment, write_report
from .validation import integer, positive_real

SCHEMA = "coupled-resolution-assessment-v2"
TRACKING_THRESHOLD = 0.01
RMS_THRESHOLD_K = 0.05
POINTWISE_THRESHOLD_K = 0.05
NODE_THRESHOLDS_K = (0.05, 0.5)
WINDOWS_S = ((0.0, 60.0), (60.0, 120.0), (120.0, 200.0), (200.0, 300.0), (300.0, 600.0))
PERCENTILES = (99.0, 99.9)


def load_states(directory):
    with np.load(Path(directory) / "states.npz", allow_pickle=False) as data:
        state, times = data["state"].copy(), data["times_s"].copy()
    if state.ndim != 2 or times.shape != (len(state),) or np.any(np.diff(times) <= 0):
        raise ValueError("Saved states need one row per strictly increasing physical time")
    return state, times


def interpolate_coarse(coarse_state, coarse_times, fine_times, initial_value):
    """Linear coarse interpolation at every fine time inside (0, horizon]."""
    if np.any(fine_times <= 0) or fine_times[-1] > coarse_times[-1] * (1 + 1e-12):
        raise ValueError("Fine times must lie inside the coarse trajectory")
    previous = np.vstack((np.full((1, coarse_state.shape[1]), initial_value), coarse_state[:-1]))
    starts = np.concatenate(([0.0], coarse_times[:-1]))
    widths = coarse_times - starts
    index = np.minimum(
        np.searchsorted(coarse_times, fine_times, side="left"), len(coarse_times) - 1
    )
    fraction = (fine_times - starts[index]) / widths[index]
    return previous[index] + fraction[:, None] * (coarse_state[index] - previous[index])


def node_location(mesh, free_node):
    full = int(mesh.free[free_node])
    touching = np.any(mesh.cells == full, axis=1)
    return {
        "free_node_index": int(free_node),
        "full_node_index": full,
        "radial_axial_coordinate_m": [float(v) for v in mesh.nodes[full]],
        "adjacent_material_labels": [int(v) for v in np.unique(mesh.materials[touching])],
    }


def pair_statistics(coarse_dir, fine_dir, mesh, mass, scale, initial_value):
    coarse_state, coarse_times = load_states(coarse_dir)
    fine_state, fine_times = load_states(fine_dir)
    if coarse_state.shape[1] != len(mass) or fine_state.shape[1] != len(mass):
        raise ValueError("Temperature columns must match the free nodes of the mesh")
    difference = scale * (
        fine_state - interpolate_coarse(coarse_state, coarse_times, fine_times, initial_value)
    )
    magnitude = np.abs(difference)
    steps = np.diff(np.concatenate(([0.0], fine_times)))
    weights = steps[:, None] * mass[None, :]
    rms_space_time = float(np.sqrt(np.sum(weights * difference**2) / np.sum(weights)))
    spatial_rms = np.sqrt((mass * difference**2).sum(axis=1) / mass.sum())
    peak_time, peak_node = np.unravel_index(int(magnitude.argmax()), magnitude.shape)
    ever = magnitude.max(axis=0)
    exceeding = {}
    for threshold in NODE_THRESHOLDS_K:
        nodes = np.flatnonzero(ever > threshold)
        exceeding[f"{threshold:g}_K"] = {
            "count": int(len(nodes)),
            "fraction_of_free_nodes": float(len(nodes) / len(mass)),
            "nodes": [node_location(mesh, node) for node in nodes[:64]],
            "listed": int(min(len(nodes), 64)),
        }
    windows = []
    for start, end in WINDOWS_S:
        mask = (fine_times > start) & (fine_times <= end)
        windows.append(
            {
                "window_s": [start, end],
                "maximum_K": float(magnitude[mask].max()) if mask.any() else None,
                "levels": int(mask.sum()),
            }
        )
    percentiles = np.percentile(magnitude, PERCENTILES)
    return {
        "coarse_levels": int(len(coarse_state)),
        "fine_levels": int(len(fine_state)),
        "pointwise_maximum_K": float(magnitude.max()),
        "pointwise_peak": {
            "time_s": float(fine_times[peak_time]),
            "signed_fine_minus_coarse_K": float(difference[peak_time, peak_node]),
            **node_location(mesh, peak_node),
        },
        "pointwise_percentiles_K": {f"{p:g}": float(v) for p, v in zip(PERCENTILES, percentiles)},
        "mass_weighted_space_time_rms_K": rms_space_time,
        "weighted_spatial_rms_maximum_over_time_K": float(spatial_rms.max()),
        "weighted_spatial_rms_at_end_K": float(spatial_rms[-1]),
        "endpoint_maximum_K": float(magnitude[-1].max()),
        "nodes_exceeding": exceeding,
        "window_maxima": windows,
        "comparison": "every fine time level against linear interpolation of the coarse trajectory from the uniform initial value",
    }


def bound_excess(directory, mesh, scale, offset, upper_K, horizon_s, original_slabs):
    state, times = load_states(directory)
    temperature = offset + scale * state
    excess = np.maximum(temperature - upper_K, 0.0)
    if not excess.size or excess.max() <= 0:
        return {"maximum_K": 0.0, "time_s": None, "location": None, "on_original_level": None}
    time_index, node = np.unravel_index(int(excess.argmax()), excess.shape)
    level = horizon_s / original_slabs
    ratio = times[time_index] / level
    return {
        "maximum_K": float(excess.max()),
        "time_s": float(times[time_index]),
        "location": node_location(mesh, node),
        "on_original_level": bool(abs(ratio - round(ratio)) < 1e-9),
        "levels_with_excess": int(np.count_nonzero(excess.max(axis=1) > 0)),
    }


def richardson(values):
    """Geometric extrapolation of successive-doubling values; None when not estimable."""
    values = [float(v) for v in values if v is not None]
    if len(values) < 3:
        return {"limit_K": None, "increment_ratio": None, "values_K": values}
    increments = np.diff(values)
    if increments[-2] == 0 or not np.isfinite(increments).all():
        return {"limit_K": None, "increment_ratio": None, "values_K": values}
    ratio = float(increments[-1] / increments[-2])
    limit = None
    if 0 < ratio < 1:
        limit = float(values[-1] + increments[-1] * ratio / (1 - ratio))
    return {
        "limit_K": limit,
        "increment_ratio": ratio,
        "values_K": values,
        "scope": "Assumes a geometric increment sequence; a ratio outside (0, 1) gives no limit.",
    }


def non_increasing(pairs, key):
    if len(pairs) < 2:
        return None
    a, b = pairs[-2][key], pairs[-1][key]
    return bool(a is not None and b is not None and b <= a + 1e-12)


def assess(directories, baseline_directory, upper_K, protocol=None, cross_scheme=False):
    baseline_directory = Path(baseline_directory)
    baseline = json.loads((baseline_directory / "record.json").read_text())
    if baseline["status"] != "converged":
        raise ValueError("The reference flow must have converged")
    mesh, parameters, conductivity, capacity, _, hashes = transformer_inputs(
        baseline["configuration"]["level"]
    )
    if hashes != baseline["input_sha256"]:
        raise ValueError("The mesh and material data differ from the recorded inputs")
    physical = parameters["physical"]
    scale, offset = physical["temperature_scale_K"], physical["inlet_temperature_K"]
    upper_K = positive_real(upper_K, "Upper temperature bound")
    mass = assemble_thermal(mesh, conductivity, capacity).mass[mesh.free]
    summary = summarize(directories, scale, initial_value=0.0, cross_scheme=cross_scheme)
    paths = {Path(directory).name: Path(directory) for directory in directories}
    pairs = []
    for pair in summary["pairs"]:
        statistics = pair_statistics(
            paths[pair["coarse_case"]], paths[pair["fine_case"]], mesh, mass, scale, 0.0
        )
        pairs.append(
            {
                "coarse_case": pair["coarse_case"],
                "fine_case": pair["fine_case"],
                "tracking_relative_change": pair["tracking_relative_change_fine_denominator"],
                "v1_maximum_all_refined_time_difference_K": pair.get(
                    "maximum_all_refined_time_difference_K"
                ),
                **statistics,
            }
        )
    rows = []
    for row in summary["rows"]:
        entry = dict(row)
        for key in ("times_s", "maximum_difference_K"):
            entry.pop(key, None)
        if row.get("complete"):
            record = json.loads((paths[row["case"]] / "record.json").read_text())
            cfg = record["configuration"]
            entry["bound_excess"] = bound_excess(
                paths[row["case"]],
                mesh,
                scale,
                offset,
                upper_K,
                float(cfg["horizon_s"]),
                integer(cfg["slabs"], "Original slabs", 1),
            )
            entry["time_scheme"] = record.get("forward_solver", {}).get(
                "time_scheme", "backward_euler"
            )
        rows.append(entry)
    complete = [row for row in rows if row.get("complete")]
    complete.sort(key=lambda row: row["subdivision"])
    # The replay's own recorded excess is the reported series; the recomputed
    # value from the saved states adds its location and must agree with it.
    recorded = [row.get("maximum_recorded_upper_violation_K") for row in complete]
    recomputed = [row["bound_excess"]["maximum_K"] for row in complete]
    excess_series = richardson(recorded if all(v is not None for v in recorded) else recomputed)
    tracking_met = (
        bool(pairs)
        and pairs[-1]["tracking_relative_change"] is not None
        and (pairs[-1]["tracking_relative_change"] <= TRACKING_THRESHOLD)
    )
    rms_met = bool(pairs) and pairs[-1]["mass_weighted_space_time_rms_K"] <= RMS_THRESHOLD_K
    pointwise_met = bool(pairs) and pairs[-1]["pointwise_maximum_K"] <= POINTWISE_THRESHOLD_K
    tracking_trend = non_increasing(pairs, "tracking_relative_change")
    rms_trend = non_increasing(pairs, "mass_weighted_space_time_rms_K")
    passed = tracking_met and rms_met and tracking_trend is True and rms_trend is True
    assessment = {
        "tracking_relative_change": {
            "threshold": TRACKING_THRESHOLD,
            "values": [p["tracking_relative_change"] for p in pairs],
            "last_met": tracking_met,
            "last_two_non_increasing": tracking_trend,
        },
        "mass_weighted_space_time_rms_K": {
            "threshold": RMS_THRESHOLD_K,
            "values": [p["mass_weighted_space_time_rms_K"] for p in pairs],
            "last_met": rms_met,
            "last_two_non_increasing": rms_trend,
        },
        "pointwise_maximum_K": {
            "threshold": POINTWISE_THRESHOLD_K,
            "values": [p["pointwise_maximum_K"] for p in pairs],
            "last_met": pointwise_met,
            "statement": "met" if pointwise_met else "not met",
        },
        "pointwise_99_9_percentile_K": [p["pointwise_percentiles_K"]["99.9"] for p in pairs],
        "recorded_bound_excess_K": recorded,
        "recomputed_bound_excess_K": recomputed,
        "bound_excess_richardson": excess_series,
        "v2_criteria_met": bool(passed and len(pairs) >= 2),
        "v1_status": summary["resolution_assessment"]["status"],
        "status": "v2_criteria_met" if passed and len(pairs) >= 2 else "v2_criteria_not_met",
        "scope": "Discrete fixed-source assessment of the 64-slab optimum on its own mesh; not a continuum feasibility certificate. The v1 pointwise verdict is reported alongside and is not replaced.",
    }
    report = {
        "schema": SCHEMA,
        "environment": environment(),
        "baseline_sha256": baseline["baseline_sha256"],
        "input_sha256": hashes,
        "temperature_scale_K": scale,
        "temperature_offset_K": offset,
        "upper_bound_K": upper_K,
        "cross_scheme": bool(cross_scheme),
        "protocol_sha256": None if protocol is None else file_digest(protocol),
        "v1_resolution_assessment": summary["resolution_assessment"],
        "rows": rows,
        "pairs": pairs,
        "assessment_v2": assessment,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--upper-K", type=float, default=357.3)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--cross-scheme", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        report = assess(args.replays, args.baseline, args.upper_K, args.protocol, args.cross_scheme)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "record.json", report)


if __name__ == "__main__":
    main()
