"""Compare nested fixed-source forward trajectories at shared physical times."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_saved import file_digest
from .reporting import write_report
from .validation import finite_real, integer, positive_real


def summarize(directories, temperature_scale):
    """Retain every run and compute endpoint differences between complete refinements.

    This comparison uses the saved fields rather than the original-optimization
    differences printed by each run. Tracking integrals use the separately
    recorded physical quadrature for each temporal resolution.
    """
    scale = positive_real(temperature_scale, "Temperature scale")
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
        )
        if identity is not None and key != identity:
            raise ValueError("Time refinements must use the same saved source and physical inputs")
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
        error = np.max(np.abs(sampled - cs), axis=1) * scale
        delta = abs(fr["tracking_integral_K2_m3_s"] - cr["tracking_integral_K2_m3_s"])
        denominator = fr["tracking_integral_K2_m3_s"]
        relative = delta / denominator if denominator else (0.0 if delta == 0 else None)
        pairs.append(
            {
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
            }
        )
    return {
        "schema": "coupled-nested-time-resolution-summary-v1",
        "temperature_scale_K": scale,
        "scope": "Pairwise fixed-source comparisons at shared time levels and using each grid's physical tracking quadrature; no reoptimization or continuous-time feasibility certificate.",
        "rows": rows,
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, nargs="+", required=True)
    parser.add_argument("--temperature-scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report = summarize(args.replays, args.temperature_scale)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", report)
    if args.plot:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
        for pair in report["pairs"]:
            axis.plot(
                pair["times_s"],
                pair["maximum_difference_K"],
                label=f"{pair['coarse_slabs']} and {pair['fine_slabs']} time levels",
            )
        axis.set(xlabel="Physical time (s)", ylabel="Maximum temperature difference (K)")
        axis.grid(alpha=0.25)
        if report["pairs"]:
            axis.legend(fontsize=8)
        for extension in ("pdf", "png"):
            figure.savefig(args.output / f"time_resolution.{extension}", dpi=220)
        plt.close(figure)


if __name__ == "__main__":
    main()
