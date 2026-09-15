"""Compare nested fixed-source forward trajectories at shared physical times."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_saved import file_digest
from .reporting import write_report
from .validation import finite_real, integer, positive_real


def assess_resolution(rows, pairs):
    """Apply the declared thresholds and check the last two refinement changes.

    A small tracking change alone does not establish temperature resolution.
    Missing or unsuccessful declared refinements prevent a passing assessment;
    they are never replaced with a comparison of the remaining successful runs.
    This is a discrete fixed-source assessment, not a feasibility certificate.
    """
    complete = bool(rows) and all(row["complete"] for row in rows)
    full_trajectory = bool(pairs) and all(
        "maximum_all_refined_time_difference_K" in pair for pair in pairs
    )
    temperature_key = (
        "maximum_all_refined_time_difference_K"
        if full_trajectory
        else "maximum_shared_time_difference_K"
    )
    last = pairs[-1] if pairs else None
    prior = pairs[-2] if len(pairs) >= 2 else None
    trend = None
    if prior is not None:
        changes = [
            (prior[key], last[key])
            for key in (
                temperature_key,
                "tracking_relative_change_fine_denominator",
            )
        ]
        # Equality at the roundoff floor is allowed. This tolerance does not
        # relax either physical-resolution threshold.
        trend = all(a is not None and b is not None and b <= a + 1e-12 for a, b in changes)
    thresholds = bool(
        last is not None and last["shared_time_thresholds_met"] and last[temperature_key] <= 0.05
    )
    passed = complete and thresholds and trend is True
    if not complete:
        status = "incomplete_declared_refinements"
    elif prior is None:
        status = "insufficient_refinements_for_trend"
    elif not thresholds:
        status = "resolution_thresholds_exceeded"
    elif not trend:
        status = "refinement_trend_not_decreasing"
    else:
        status = "discrete_resolution_thresholds_met"
    return {
        "status": status,
        "all_declared_refinements_complete": complete,
        "temperature_threshold_K": 0.05,
        "tracking_relative_threshold": 0.01,
        "last_pair_thresholds_met": thresholds,
        "last_two_changes_nonincreasing": trend,
        "discrete_time_resolution_met": passed,
        "temperature_comparison": "all refined times versus linear coarse interpolation"
        if full_trajectory
        else "shared endpoints only",
        "scope": "The last two temperature and tracking changes must decrease to within roundoff; the finest pair must meet both thresholds. The reported sampling defines the temperature comparison. Spatial resolution, reoptimization and bound satisfaction require separate checks.",
    }


def summarize(directories, temperature_scale, initial_value=None):
    """Retain every run and compute endpoint differences between complete refinements.

    This comparison uses the saved fields rather than the original-optimization
    differences printed by each run. Tracking integrals use the separately
    recorded physical quadrature for each temporal resolution.
    """
    scale = positive_real(temperature_scale, "Temperature scale")
    if initial_value is not None:
        initial_value = finite_real(initial_value, "Uniform initial dimensionless temperature")
    rows, complete, identity = [], [], None
    for directory in map(Path, directories):
        path = directory / "record.json"
        if not path.exists():
            rows.append({"case": directory.name, "status": "missing", "complete": False})
            continue
        record = json.loads(path.read_text())
        key = (
            record["optimization_field_sha256"],
            record["baseline_sha256"],
            record["configuration"],
            record["forward_solver"],
        )
        if identity is not None and key != identity:
            raise ValueError(
                "Time refinements must use the same saved source, physical inputs and forward policy"
            )
        identity = key
        subdivision = integer(record["subdivision"], "Time subdivision", 1)
        original_slabs = integer(record["configuration"]["slabs"], "Original slabs", 1)
        row = {
            "case": directory.name,
            "status": record["status"],
            "complete": False,
            "subdivision": subdivision,
            "forward_solver": record["forward_solver"],
            "source": record["environment"]["git_head"],
            "record_sha256": file_digest(path),
            "seconds": record.get("seconds"),
        }
        for name in ("maximum_recorded_upper_violation_K", "maximum_recorded_lower_violation_K"):
            value = record.get(name)
            if value is not None:
                value = finite_real(value, name)
                if value < 0:
                    raise ValueError("Recorded bound excess must be nonnegative")
            # A missing measurement remains missing, not a zero bound excess.
            row[name] = value
        fields = directory / "states.npz"
        if fields.exists():
            with np.load(fields, allow_pickle=False) as arrays:
                state, times = arrays["state"].copy(), arrays["times_s"].copy()
            steps = record["steps"]
            if (
                state.ndim != 2
                or not state.size
                or not np.isfinite(state).all()
                or times.shape != (len(state),)
                or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)
                or times[0] <= 0
                or len(steps) != len(state)
                or not np.allclose(times, [r["time_s"] for r in steps], rtol=1e-12, atol=1e-12)
            ):
                raise ValueError("Stored fields and recorded physical times must agree")
            is_complete = (
                len(state) == original_slabs * subdivision
                and all(r["status"] == "converged" for r in steps)
                and record["status"] == "converged"
            )
            row.update(
                complete=is_complete, recorded_slabs=len(state), field_sha256=file_digest(fields)
            )
            if is_complete:
                tracking = finite_real(
                    record["tracking_integral_refined_K2_m3_s"], "Physical tracking integral"
                )
                if tracking < 0:
                    raise ValueError("Physical tracking integral must be nonnegative")
                row["tracking_integral_K2_m3_s"] = tracking
                complete.append((row, state, times))
        if record["status"] == "converged" and not row["complete"]:
            raise ValueError("A converged trajectory requires every recorded state")
        rows.append(row)
    if not rows or len({r["case"] for r in rows}) != len(rows):
        raise ValueError("Supply distinct forward computations")
    complete.sort(key=lambda item: item[0]["subdivision"])
    pairs = []
    for coarse, fine in zip(complete, complete[1:]):
        cr, cs, ct = coarse
        fr, fs, ft = fine
        if fr["subdivision"] % cr["subdivision"] or fr["subdivision"] == cr["subdivision"]:
            raise ValueError("Complete temporal refinements must be distinct and nested")
        stride = fr["subdivision"] // cr["subdivision"]
        sampled, shared_times = fs[stride - 1 :: stride], ft[stride - 1 :: stride]
        if sampled.shape != cs.shape or not np.allclose(shared_times, ct, rtol=1e-12, atol=1e-12):
            raise ValueError("Refinements must share the original physical endpoints")
        difference = sampled - cs
        error = np.max(np.abs(difference), axis=1) * scale
        peak_time, peak_node = np.unravel_index(np.abs(difference).argmax(), difference.shape)
        delta = abs(fr["tracking_integral_K2_m3_s"] - cr["tracking_integral_K2_m3_s"])
        denominator = fr["tracking_integral_K2_m3_s"]
        relative = delta / denominator if denominator else (0.0 if delta == 0 else None)
        pair = {
            "coarse_case": cr["case"],
            "fine_case": fr["case"],
            "coarse_slabs": len(cs),
            "fine_slabs": len(fs),
            "maximum_shared_time_difference_K": float(error.max()),
            "tracking_absolute_change_K2_m3_s": delta,
            "tracking_relative_change_fine_denominator": relative,
            "shared_time_thresholds_met": bool(
                error.max() <= 0.05 and relative is not None and relative <= 0.01
            ),
            "times_s": ct,
            "maximum_difference_K": error,
            "shared_time_peak": {
                "time_s": float(ct[peak_time]),
                "free_node_index": int(peak_node),
                "signed_fine_minus_coarse_K": float(difference[peak_time, peak_node] * scale),
            },
        }
        if initial_value is not None:
            # Only a coarse-grid-sized temporary is needed for interpolation,
            # even when the fine replay has many more physical time levels.
            previous = np.vstack((np.full((1, cs.shape[1]), initial_value), cs[:-1]))
            starts = np.concatenate(([0.0], ct[:-1]))
            widths = ct - starts
            trajectory_error = np.empty(len(fs))
            peak_nodes = np.empty(len(fs), dtype=int)
            peak_signed = np.empty(len(fs))
            for substep in range(stride):
                indices = np.arange(substep, len(fs), stride)
                fraction = (ft[indices] - starts) / widths
                if np.any(fraction <= 0) or np.any(fraction > 1 + 1e-12):
                    raise ValueError(
                        "Fine time levels must lie in their corresponding coarse interval"
                    )
                interpolated = previous + fraction[:, None] * (cs - previous)
                delta_state = fs[indices] - interpolated
                nodes = np.abs(delta_state).argmax(axis=1)
                signed = delta_state[np.arange(len(indices)), nodes] * scale
                trajectory_error[indices] = np.abs(signed)
                peak_nodes[indices], peak_signed[indices] = nodes, signed
            peak_time = int(trajectory_error.argmax())
            peak_node = int(peak_nodes[peak_time])
            pair.update(
                maximum_all_refined_time_difference_K=float(trajectory_error.max()),
                refined_times_s=ft,
                maximum_interpolated_difference_K=trajectory_error,
                all_refined_time_peak={
                    "time_s": float(ft[peak_time]),
                    "fine_time_index": peak_time,
                    "free_node_index": peak_node,
                    "signed_fine_minus_coarse_K": float(peak_signed[peak_time]),
                },
                peak_node_trace={
                    "free_node_index": peak_node,
                    "coarse_times_s": np.r_[0.0, ct],
                    "coarse_dimensionless_temperature": np.r_[initial_value, cs[:, peak_node]],
                    "fine_times_s": np.r_[0.0, ft],
                    "fine_dimensionless_temperature": np.r_[initial_value, fs[:, peak_node]],
                },
            )
        pairs.append(pair)
    return {
        "schema": "coupled-nested-time-resolution-summary-v1",
        "temperature_scale_K": scale,
        "initial_dimensionless_value": initial_value,
        "scope": "Pairwise fixed-source comparisons using each grid's physical tracking quadrature. When the uniform initial temperature is supplied, temperatures are also compared at every refined level with linear interpolation of the coarse trajectory. This is not reoptimization or a continuous-time feasibility certificate.",
        "rows": rows,
        "pairs": pairs,
        "resolution_assessment": assess_resolution(rows, pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, nargs="+", required=True)
    parser.add_argument("--temperature-scale", type=float, required=True)
    parser.add_argument(
        "--initial-value",
        type=float,
        help="Uniform initial dimensionless temperature; enables comparison at every refined time level. Use 0 for the declared transformer model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report = summarize(args.replays, args.temperature_scale, args.initial_value)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", report)
    if args.plot:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
        for pair in report["pairs"]:
            axis.plot(
                pair.get("refined_times_s", pair["times_s"]),
                pair.get("maximum_interpolated_difference_K", pair["maximum_difference_K"]),
                label=f"{pair['coarse_slabs']} and {pair['fine_slabs']} time levels",
            )
        axis.set(xlabel="Physical time (s)", ylabel="Maximum temperature difference (K)")
        axis.axhline(0.05, color="black", linestyle=":", linewidth=1, label="0.05 K threshold")
        axis.grid(alpha=0.25)
        if report["pairs"]:
            axis.legend(fontsize=8)
        for extension in ("pdf", "png"):
            figure.savefig(args.output / f"time_resolution.{extension}", dpi=220)
        plt.close(figure)


if __name__ == "__main__":
    main()
