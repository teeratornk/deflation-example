"""Recover an unclipped first trial direction from a verified checkpoint."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_bounds import temperature_bounds
from .coupled_optimize import load_problem
from .coupled_review import read_snapshot
from .reporting import environment, file_sha256, write_arrays, write_report


def reconstruct(meta, arrays, lower, upper):
    """Recover only the first update, with an independently stored displacement.

    This reuses a recorded QP outcome; it neither solves nor verifies a new QP.
    An extrapolation that needs clipping is rejected because the checkpoint
    cannot then identify the original trial direction.
    """
    history = meta["history"]
    if meta["iteration"] != 0 or meta["position"] != 0 or len(history) != 1:
        raise ValueError("Only the first update of the first query is supported")
    attempts = history[0]["attempts"]
    successful = [
        (attempt, trial)
        for attempt in attempts
        for trial in attempt["trials"]
        if trial["status"] in {"decrease", "roundoff_kkt_decrease"}
    ]
    if len(successful) != 1:
        raise ValueError("Exactly one recorded successful trial is required")
    attempt, trial = successful[0]
    if attempt is not attempts[-1] or trial is not attempt["trials"][-1]:
        raise ValueError("The successful trial must end the checkpoint history")
    if attempt["qp_status"] != "converged":
        raise ValueError("The recorded QP must have converged")
    length = float(trial["step"])
    if not np.isfinite(length) or not 0 < length < 1:
        raise ValueError("Reconstruction requires a step strictly between zero and one")
    state = np.asarray(arrays["state"], dtype=float)
    steps = np.asarray(arrays["secant_steps"], dtype=float)
    if state.ndim != 1 or steps.shape != (1, state.size):
        raise ValueError("Exactly one complete secant displacement is required")
    lo, hi = np.broadcast_to(lower, state.shape), np.broadcast_to(upper, state.shape)
    if not np.isfinite([state, steps[0], lo, hi]).all() or np.any(lo >= hi):
        raise ValueError("States, displacements and ordered bounds must be finite")
    initial = np.clip(np.zeros_like(state), lo, hi)
    tolerance = 64 * np.finfo(float).eps * max(1.0, np.max(np.abs(state)))
    defect = float(np.max(np.abs(state - initial - steps[0])))
    if defect > tolerance:
        raise ValueError("The stored displacement does not start at the original zero state")
    direction = steps[0] / length
    full = initial + direction
    violation = float(max(0.0, np.max(lo - full), np.max(full - hi)))
    if violation > tolerance:
        raise ValueError("The extrapolated full step requires clipping")
    if not np.isclose(meta["objective"], trial["objective"], rtol=1e-13, atol=1e-13):
        raise ValueError("The checkpoint objective differs from its final trial")
    return (
        initial,
        direction,
        {
            "policy": "first-accepted-displacement-divided-by-recorded-step-v1",
            "recorded_step": length,
            "recorded_trial": trial,
            "recorded_qp_status": attempt["qp_status"],
            "recorded_qp_kkt": attempt["qp_kkt"],
            "recorded_qp_history": attempt["qp_history"],
            "initial_displacement_defect": defect,
            "extrapolated_bound_violation": violation,
            "roundoff_tolerance": tolerance,
            "new_qp_solved": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record, meta, arrays, manifest = read_snapshot(args.snapshot)
    with threadpool_limits(8):
        cfg = {**record["configuration"], "baseline_directory": str(args.baseline)}
        problem, baseline = load_problem(cfg)
        query = cfg["queries"][meta["position"]]
        bounds = temperature_bounds(cfg, upper_K=query["upper_K"])
        lower = (
            bounds["optimization_lower_K"] - problem.temperature_offset
        ) / problem.temperature_scale
        upper = (
            bounds["optimization_upper_K"] - problem.temperature_offset
        ) / problem.temperature_scale
        state, direction, lineage = reconstruct(meta, arrays, lower, upper)
    args.output.mkdir(parents=True, exist_ok=False)
    write_arrays(args.output / "direction.npz", state=state, direction=direction)
    write_report(
        args.output / "review.json",
        {
            "schema": "coupled-reconstructed-direction-v1",
            "status": "reconstructed",
            "snapshot": manifest,
            "source_environment": environment(),
            "baseline_sha256": baseline["baseline_sha256"],
            "direction_sha256": file_sha256(args.output / "direction.npz"),
            "initial_state_policy": "original_zero_state",
            "qp": {"status": lineage["recorded_qp_status"], "origin": "recorded_checkpoint"},
            "direction_origin": lineage,
            "scope": "Reconstructed recorded direction for forward diagnostics; no new QP or optimizer performance measurement.",
        },
    )


if __name__ == "__main__":
    main()
