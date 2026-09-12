"""Refine forward time steps while keeping a previously optimized source fixed."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .reporting import environment, write_fields, write_report
from .temporal_resolution import digest, forward_replay, model, weighted_rms
from .validation import integer


def selected_cases(design):
    """Use both declared targets at the coarsest and finest optimization grids."""
    endpoints = {min(design["protocol"]["slabs"]), max(design["protocol"]["slabs"])}
    return [i for i, case in enumerate(design["cases"]) if case["slabs"] in endpoints]


def replay_metrics(state, previous, p, bound, scale, horizon_seconds):
    violation = np.maximum(state - bound, 0)
    location = np.unravel_index(int(violation.argmax()), state.shape)
    result = {
        "maximum_bound_violation_K": float(violation.max() * scale),
        "maximum_temperature_rise_K": float(state.max() * scale),
        "maximum_bound_violation_time_seconds": float(
            (location[0] + 1) * horizon_seconds / len(state)
        )
        if violation.max() > 0
        else None,
        "maximum_bound_violation_coordinates": p.assembly.mesh.nodes[p.free[location[1]]].tolist()
        if violation.max() > 0
        else None,
    }
    if previous is not None:
        if state.shape != (2 * len(previous), previous.shape[1]):
            raise ValueError("Successive replay grids must double")
        difference = state[1::2] - previous
        result.update(
            temperature_change_max_K=float(np.abs(difference).max() * scale),
            temperature_change_rms_K=weighted_rms(difference, p.assembly.mass[p.free]) * scale,
        )
    return result


def run(root, output, case, steps=(256, 512, 1024, 2048, 4096)):
    root, output = Path(root), Path(output)
    design = json.loads((root / "design.json").read_text())
    c = design["protocol"]
    if digest(c) != design["protocol_sha256"]:
        raise ValueError("The original protocol checksum differs")
    integer(case, "Case index")
    if case not in selected_cases(design):
        raise ValueError("Use the preselected endpoint cases")
    if len(steps) < 3:
        raise ValueError("Use at least three replay grids")
    for n in steps:
        integer(n, "Replay slabs", 1)
    if steps[0] < max(c["slabs"]) or steps[0] % max(c["slabs"]):
        raise ValueError("Replay steps must align with every original control jump")
    if any(b != 2 * a for a, b in zip(steps[:-1], steps[1:], strict=True)):
        raise ValueError("Replay grids must double")
    current = environment()
    for name, checksum in design["environment"]["source_sha256"].items():
        if current["source_sha256"].get(name) != checksum:
            raise ValueError(f"Original numerical source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "fixed-source-temporal-refinement-v1",
        "case": case,
        "selected_cases": selected_cases(design),
        "selection": "Both preselected targets at the coarsest and finest optimization grids; all forward grids are evaluated.",
        "optimization_case": design["cases"][case],
        "optimization_source": design["environment"],
        "environment": current,
        "protocol_sha256": design["protocol_sha256"],
        "replay_slabs": list(steps),
        "temperature_change_scale_K": c["temperature_change_K"],
        "status": "running",
        "rows": [],
        "scope": "Unchanged piecewise-constant optimized source. Bound violations include all replay time nodes. Consecutive-grid changes estimate temporal sensitivity, not rigorous error bounds or spatial resolution.",
    }
    write_report(output / "summary.json", report)
    try:
        directory = root / f"case-{case:02d}"
        record = json.loads((directory / "record.json").read_text())
        if (
            record["protocol_sha256"] != design["protocol_sha256"]
            or any(record[k] != v for k, v in design["cases"][case].items())
            or record["environment"]["source_sha256"] != design["environment"]["source_sha256"]
        ):
            raise ValueError("Optimization record differs from the declared case")
        if not record.get("verified", False):
            report["status"] = "optimization_not_verified"
            write_report(output / "summary.json", report)
            return report
        if (
            not record["optimization"]["success"]
            or max(record["optimization"]["kkt"].values()) > c["kkt_tolerance"]
        ):
            raise ValueError("Optimization verification is inconsistent")
        fields_path = directory / "fields.npz"
        checksum = hashlib.sha256(fields_path.read_bytes()).hexdigest()
        if checksum != record["fields_sha256"]:
            raise ValueError("Optimized fields checksum differs")
        report["fields_sha256"] = checksum
        showcase, p = model(c, min(c["slabs"]))
        if showcase.preparation["input_sha256"] != design["input_sha256"]:
            raise ValueError("Thermal input checksum differs")
        with np.load(fields_path, allow_pickle=False) as fields:
            control = fields["control"].reshape(record["slabs"], p.spatial_size)
        if not np.isfinite(control).all():
            raise ValueError("Optimized source must be finite")
        scale = design["physical"].get("temperature_scale_K", 1.0)
        horizon_seconds = c["horizon"] * design["physical"].get("time_scale_s", 1.0)
        report["horizon_seconds"] = horizon_seconds
        previous = None
        for n in steps:
            state, residual = forward_replay(p, control, n)
            finite = bool(np.isfinite(state).all() and np.isfinite(residual))
            row = {
                "slabs": n,
                "maximum_original_residual": residual,
                "status": "verified" if finite and residual <= c["rtol"] else "verification_failed",
            }
            if finite:
                row.update(replay_metrics(state, previous, p, c["bound"], scale, horizon_seconds))
                row["temperature_change_within_scale"] = (
                    row.get("temperature_change_max_K", np.inf) <= c["temperature_change_K"]
                )
                write_fields(
                    output / f"samples-{n}.npz",
                    time_seconds=np.arange(1, n + 1) * horizon_seconds / n,
                    maximum_temperature_rise_K=state.max(axis=1) * scale,
                    maximum_bound_violation_K=np.maximum(state.max(axis=1) - c["bound"], 0) * scale,
                    spatial_peak_temperature_rise_K=state.max(axis=0) * scale,
                    final_temperature_rise_K=state[-1] * scale,
                )
            report["rows"].append(row)
            write_report(output / "summary.json", report)
            if not finite:
                break
            previous = state
        all_verified = len(report["rows"]) == len(steps) and all(
            r["status"] == "verified" for r in report["rows"]
        )
        report["status"] = (
            "within_temperature_change_scale"
            if all_verified
            and all(r["temperature_change_within_scale"] for r in report["rows"][-2:])
            else "resolution_cap"
            if all_verified
            else "verification_failed"
        )
        report["criterion"] = (
            "Both final consecutive-grid maximum changes must be at most the declared scale; all equation checks must pass. This criterion does not require temperature-bound satisfaction."
        )
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__)
        write_report(output / "summary.json", report)
        raise
    write_report(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", required=True, type=int)
    parser.add_argument("--slabs", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    integer(args.threads, "Threads", 1)
    with threadpool_limits(args.threads):
        run(args.root, args.output, args.case, args.slabs)


if __name__ == "__main__":
    main()
