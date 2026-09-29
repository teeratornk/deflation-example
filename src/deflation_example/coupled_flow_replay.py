"""Reconstruct retained-state trials and compare momentum solution policies."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_control import FlowEvaluationError
from .coupled_optimize import adjoint_acceptance, equation_acceptance, load_problem
from .coupled_sequence import RestoredEvaluation
from .coupled_trace import read_arrays, read_manifest
from .coupled_trust_check import check
from .reporting import environment, file_sha256, write_report


def reconstructed_trial(base, retained, accepted_step, trial_step, lower, upper, scale, reported):
    """Recover a direction from its retained endpoint, checking the recorded norm.

    This is a floating-point reconstruction, not an archived trial field. It is
    valid only when the original quadratic direction was within the physical box.
    The maximum increment and accepted endpoint are checked before a replay.
    """
    if not 0 < accepted_step <= trial_step <= 1:
        raise ValueError("Use a recorded trial preceding or equal to the accepted step")
    if base.shape != retained.shape or not np.isfinite([base, retained]).all():
        raise ValueError("Retained states must be finite and have matching shapes")
    direction = (retained - base) / accepted_step
    full = base + direction
    tolerance = 256 * np.finfo(float).eps * max(1, np.max(np.abs(full))) / accepted_step
    if np.any(full < lower - tolerance) or np.any(full > upper + tolerance):
        raise ValueError("The reconstructed full direction leaves the physical box")
    restored = np.clip(base + accepted_step * direction, lower, upper)
    if not np.allclose(restored, retained, atol=tolerance, rtol=0):
        raise ValueError("The reconstructed direction does not recover its retained endpoint")
    candidate = np.clip(base + trial_step * direction, lower, upper)
    increment = float(np.max(np.abs(candidate - base))) * scale
    if not np.isclose(increment, reported, atol=scale * tolerance, rtol=1e-10):
        raise ValueError("The reconstructed increment differs from the recorded trial")
    return candidate, {"increment_K": increment, "roundoff_tolerance_K": scale * tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("trial", "derivatives"))
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--quadratic", type=int, required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=7200)
    parser.add_argument("--polish-load", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error("Use a positive finite budget per policy")
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.record.read_text())
    manifest = read_manifest(args.trace)
    metadata = manifest["quadratics"][args.quadratic]
    arrays = read_arrays(args.trace, metadata["file"], metadata["sha256"])
    cfg = {**source["configuration"], "baseline_directory": str(args.baseline.resolve())}
    if any(cfg.get(k) != v for k, v in manifest["configuration"].items()):
        raise ValueError("Trace and optimization configurations differ")
    report = {
        "schema": "coupled-flow-policy-replay-v1",
        "status": "running",
        "environment": environment(),
        "source_record_sha256": file_sha256(args.record),
        "trace_manifest_sha256": file_sha256(args.trace / "manifest.json"),
        "quadratic": metadata,
        "mode": args.mode,
        "polish_load": args.polish_load,
        "cases": [],
        "scope": "Diagnostic evaluation, without optimization or performance-speedup claims.",
    }
    write_report(args.output / "record.json", report)
    with threadpool_limits(cfg["threads"]):
        problem, baseline = load_problem(cfg)
        if baseline["baseline_sha256"] != manifest["baseline_sha256"]:
            raise ValueError("The replay baseline differs from the recorded problem")
        guess = RestoredEvaluation(arrays["state"], arrays["velocity"], arrays["pressure"])
        state = arrays["state"]
        lower = (cfg["lower_K"] - problem.temperature_offset) / problem.temperature_scale
        upper = (
            cfg["queries"][0]["upper_K"] - problem.temperature_offset
        ) / problem.temperature_scale
        if args.mode == "trial":
            following = manifest["quadratics"][args.quadratic + 1]
            if following["iteration"] != metadata["iteration"] + 1:
                raise ValueError("The following quadratic must be the next retained state")
            next_arrays = read_arrays(args.trace, following["file"], following["sha256"])
            history = source["cases"][0]["history"]
            outer = next(r for r in history if r["iteration"] == metadata["iteration"])
            attempt = outer["attempts"][metadata["attempt"]]
            accepted = [r for r in attempt["trials"] if r["accepted"]]
            if len(accepted) != 1:
                raise ValueError("A unique retained endpoint is required")
            trial = attempt["trials"][args.trial]
            state, report["reconstruction"] = reconstructed_trial(
                state,
                next_arrays["state"],
                accepted[0]["step"],
                trial["step"],
                lower,
                upper,
                problem.temperature_scale,
                trial["temperature_step_K"],
            )
            report["original_trial"] = {k: v for k, v in trial.items() if k != "flow_history"}
            report["next_quadratic"] = following
        for continuation in (False, True) if args.mode == "trial" else (False,):
            began = time.perf_counter()
            problem.flow_continuation = continuation
            problem.flow_continuation_polish = args.polish_load
            problem.stop_requested = lambda: time.perf_counter() - began >= args.budget_seconds
            problem.evaluation_callback = lambda row: write_report(
                args.output / "progress.json", row
            )
            row = {"continuation": continuation, "status": "running"}
            report["cases"].append(row)
            write_report(args.output / "record.json", report)
            try:
                if args.mode == "derivatives":
                    checks = check(problem, state, guess, arrays["desired"], bounds=(lower, upper))
                    row["checks"] = checks
                    valid = (
                        checks["derivatives_passed"]
                        and equation_acceptance(checks["equations"], cfg)
                        and adjoint_acceptance(checks["adjoint"])
                    )
                else:
                    evaluation = problem.evaluate(state, initial=guess)
                    checks = problem.verify(evaluation, local_mass=True)
                    row.update(equations=checks, flow_history=[f.history for f in evaluation.flows])
                    valid = equation_acceptance(checks, cfg)
                row["status"] = "verified" if valid else "verification_failed"
            except FlowEvaluationError as failure:
                row.update(
                    status="flow_" + failure.result.status,
                    slab=failure.slab,
                    metrics=failure.metrics,
                    flow_history=failure.result.history,
                )
            except Exception as failure:
                row.update(
                    status="diagnostic_error", error_type=type(failure).__name__, error=str(failure)
                )
            row["seconds"] = time.perf_counter() - began
            write_report(args.output / "record.json", report)
        report["status"] = "complete"
        write_report(args.output / "record.json", report)


if __name__ == "__main__":
    main()
