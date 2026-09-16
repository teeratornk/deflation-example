"""Freeze three targets using desired fields alone, before final solver outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from deflation_example.coupled_optimize import load_problem
from deflation_example.coupled_targets import desired_temperature
from deflation_example.reporting import environment, file_sha256, write_report
from deflation_example.validation import integer


def choose(rows, nominal=7):
    """Nominal, nearest population-median amplitude, maximum remaining amplitude."""
    nominal = integer(nominal, "Nominal target", 0)
    if len(rows) < 3 or len({r["target"] for r in rows}) != len(rows):
        raise ValueError("At least three uniquely indexed target fields are required")
    if len({r["field_sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate desired fields must be resolved before selection")
    amplitudes = np.asarray([r["weighted_rms_rise_K"] for r in rows])
    if not np.isfinite(amplitudes).all() or np.any(amplitudes < 0):
        raise ValueError("Target amplitudes must be finite and nonnegative")
    if nominal not in {r["target"] for r in rows}:
        raise ValueError("Nominal target is absent")
    median = float(np.median(amplitudes))
    others = [row for row in rows if row["target"] != nominal]
    central = min(others, key=lambda r: (abs(r["weighted_rms_rise_K"] - median), r["target"]))
    remaining = [r for r in others if r["target"] != central["target"]]
    demanding = min(remaining, key=lambda r: (-r["weighted_rms_rise_K"], r["target"]))
    return [nominal, central["target"], demanding["target"]]


def target_population(problem, count, startup_s):
    count = integer(count, "Target count", 3)
    weights = np.asarray(problem.weights)
    initial = np.tile(problem.initial, problem.slabs)
    if (
        weights.shape != initial.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or not np.isfinite(initial).all()
        or not np.isfinite(problem.temperature_scale)
        or problem.temperature_scale <= 0
    ):
        raise ValueError(
            "Target selection requires positive physical weights and a finite initial state"
        )
    rows = []
    for query in range(count):
        field = desired_temperature(problem, query, count, startup_s)
        if field.shape != weights.shape or not np.isfinite(field).all():
            raise ValueError("Desired field must match all space-time weights")
        amplitude = problem.temperature_scale * np.sqrt(
            np.dot(weights, (field - initial) ** 2) / weights.sum()
        )
        rows.append(
            {
                "target": query,
                "weighted_rms_rise_K": float(amplitude),
                "field_sha256": hashlib.sha256(
                    np.asarray(field, dtype="<f8").tobytes()
                ).hexdigest(),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration-record", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error("Choose a new output directory and positive thread count")
    # Only the input configuration is read. No controls, outcomes or timing enter selection.
    configuration = json.loads(args.configuration_record.read_text())["configuration"]
    if (
        not configuration["transient"]
        or configuration["slabs"] != 64
        or configuration["horizon_s"] != 600
        or configuration["target_startup_s"] != 60
        or configuration["target_count"] != 16
    ):
        raise ValueError("Freeze targets on the declared nominal 64-slab population")
    with threadpool_limits(args.threads):
        problem, baseline = load_problem(
            {**configuration, "baseline_directory": str(args.baseline)}
        )
        rows = target_population(problem, 16, 60)
        selected = choose(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(
        args.output / "selection.json",
        {
            "schema": "coupled-publication-target-selection-v1",
            "environment": environment(),
            "driver_sha256": file_sha256(Path(__file__)),
            "configuration_record_sha256": file_sha256(args.configuration_record),
            "baseline_sha256": baseline["baseline_sha256"],
            "quantity": "Weighted RMS desired-temperature difference from the initial field, in kelvin; positive space-time tracking weights",
            "selection_order": "target 7, closest to the population median excluding target 7, highest remaining amplitude; ties use the lowest index",
            "selected_targets": selected,
            "rows": rows,
            "scope": "Selection uses desired fields alone. No final optimization or forward outcomes enter this rule.",
        },
    )


if __name__ == "__main__":
    main()
