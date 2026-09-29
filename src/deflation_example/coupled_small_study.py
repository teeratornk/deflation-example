"""Verified, bounded temporal-refinement study of complete coupled optimization."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_initial_state import snapshot_initial_guess
from .coupled_optimize import adjoint_acceptance, equation_acceptance, load_problem
from .coupled_targets import desired_temperature
from .coupled_trust_check import check
from .coupled_trust_run import run
from .reporting import environment, file_sha256, write_report


ALPHAS = (1e-14, 1e-12, 1e-11)
RANKS = (0, 8, 16, 32)
SEQUENCES = {"development": (7,), "nearby": (7, 8, 9), "stress": (7, 15, 14)}


def configuration(original, *, slabs, alpha, rank, sequence, device, continuation):
    """Preserve physical data and accuracy; vary only the declared study axes."""
    if slabs not in (16, 32, 64) or alpha not in ALPHAS or rank not in RANKS:
        raise ValueError("Use the predeclared slabs, alpha and rank grid")
    if sequence not in SEQUENCES or device not in ("cpu", "cuda", "hybrid"):
        raise ValueError("Unknown sequence or execution backend")
    if not isinstance(continuation, bool):
        raise ValueError("Continuation must be Boolean")
    if (
        original.get("transient") is not True
        or original.get("horizon_s") != 600.0
        or original.get("target_startup_s") != 60.0
        or original.get("target_count") != 16
        or original.get("feedback_multiplier", 1.0) != 1.0
        or {q["upper_K"] for q in original["queries"]} != {357.3}
        or original.get("lower_K") != 337.3
    ):
        raise ValueError("The source must declare the full-feedback 600-second physical study")
    cfg = dict(original)
    for key in ("regularization_screen_sha256", "output", "stage"):
        cfg.pop(key, None)
    cfg.update(
        slabs=slabs,
        alpha=alpha,
        rank=rank,
        method="jacobi" if rank == 0 else "reference",
        device=device,
        frozen_layout="block_diagonal" if device == "cuda" else "serial",
        queries=[{"target": target, "upper_K": 357.3} for target in SEQUENCES[sequence]],
        initial_state_time_policy="nested_endpoints",
        initial_state_alpha_policy="shared_temperature",
        flow_continuation=continuation,
        cooperative_flow_deadline=True,
        optimization_only_budget=True,
        capture_trials=False,
        capture_linear_systems=False,
        inner_preconditioner="frozen",
        frozen_sweeps=3,
        qp_solver="projected",
        trial_policy="backtrack",
        trust_accuracy="adaptive_projected",
        inner_tolerance=1e-10,
        qp_tolerance=1e-10,
        nonlinear_tolerance=1e-8,
        flow_tolerance=1e-12,
        equation_acceptance_tolerance=1e-12,
        conservation_tolerance=1e-6,
        inner_cap=50000,
        nonlinear_cap=100,
        reference_krylov_steps=48,
        reference_krylov_seed=20260923,
        reference_krylov_selection="alternating_low_high",
        reference_construction="initial_trajectory_energy_krylov",
        reference_selection="alternating_low_high",
        recycle_window=max(1, 2 * rank),
        warm_start=True,
        study_sequence=sequence,
        study_protocol="coupled-small-to-large-v1",
    )
    return cfg


def verification_identity(cfg):
    # Rank/backend/sequence do not alter the initial physical problem. Derivative
    # verification still binds alpha, discretization, flow policy and all tolerances.
    excluded = {
        "rank",
        "method",
        "device",
        "frozen_layout",
        "recycle_window",
        "queries",
        "study_sequence",
        "repetition",
        "verification_gate_sha256",
    }
    settings = {k: v for k, v in cfg.items() if k not in excluded}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def require_verification(report, cfg, source):
    if (
        report.get("schema") != "coupled-small-study-verification-v1"
        or report.get("status") != "verified"
        or report.get("configuration_identity") != verification_identity(cfg)
        or report.get("environment", {}).get("source_sha256") != source
    ):
        raise ValueError(
            "A verified, source-matched initial problem is required before optimization"
        )


def verify(cfg, output, budget_seconds):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {
        "schema": "coupled-small-study-verification-v1",
        "status": "running",
        "environment": environment(),
        "configuration_identity": verification_identity(cfg),
        "configuration": cfg,
        "scope": "Initial coupled trajectory, derivatives and adjoint on the declared time grid.",
    }
    write_report(output / "record.json", report)
    try:
        with threadpool_limits(cfg["threads"]):
            problem, baseline = load_problem(cfg)
            problem.stop_requested = lambda: time.perf_counter() - start >= budget_seconds
            problem.evaluation_callback = lambda row: write_report(output / "progress.json", row)
            guess, report["initial_state"] = snapshot_initial_guess(cfg, problem, baseline, 0)
            report["baseline_sha256"] = baseline["baseline_sha256"]
            desired = desired_temperature(problem, 7, cfg["target_count"], cfg["target_startup_s"])
            bounds = tuple(
                (v - problem.temperature_offset) / problem.temperature_scale
                for v in (cfg["lower_K"], cfg["queries"][0]["upper_K"])
            )
            result = check(problem, guess.state, guess, desired, bounds=bounds)
            report["checks"] = result
            valid = (
                result["derivatives_passed"]
                and equation_acceptance(result["equations"], cfg)
                and adjoint_acceptance(result["adjoint"])
            )
            report["status"] = "verified" if valid else "verification_failed"
    except Exception as failure:
        report.update(
            status="verification_error", error_type=type(failure).__name__, error=str(failure)
        )
        if hasattr(failure, "result"):
            report.update(
                flow_status=failure.result.status, slab=failure.slab, metrics=failure.metrics
            )
    finally:
        report["seconds"] = time.perf_counter() - start
        write_report(output / "record.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "run"))
    for name in ("source-record", "baseline", "initial-snapshot", "initial-assessment", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--slabs", type=int, choices=(16, 32, 64), default=16)
    parser.add_argument("--alpha", type=float, choices=ALPHAS, default=1e-11)
    parser.add_argument("--rank", type=int, choices=RANKS, default=0)
    parser.add_argument("--sequence", choices=tuple(SEQUENCES), default="development")
    parser.add_argument("--device", choices=("cpu", "cuda", "hybrid"), default="cpu")
    parser.add_argument("--continuation", action="store_true")
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--budget-seconds", type=float, default=14400)
    args = parser.parse_args()
    if not np.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error("Use a positive finite time budget")
    original = json.loads(args.source_record.read_text())["configuration"]
    cfg = configuration(
        original,
        slabs=args.slabs,
        alpha=args.alpha,
        rank=args.rank,
        sequence=args.sequence,
        device=args.device,
        continuation=args.continuation,
    )
    cfg.update(
        baseline_directory=str(args.baseline.resolve()),
        initial_state_snapshot=str(args.initial_snapshot.resolve()),
        initial_state_assessment=str(args.initial_assessment.resolve()),
        study_source_record_sha256=file_sha256(args.source_record),
    )
    if args.mode == "verify":
        result = verify(cfg, args.output, args.budget_seconds)
        success = result["status"] == "verified"
    else:
        if args.verification is None:
            parser.error("Optimization requires --verification")
        gate = json.loads(args.verification.read_text())
        require_verification(gate, cfg, environment()["source_sha256"])
        cfg["verification_gate_sha256"] = file_sha256(args.verification)
        result = run(cfg, args.output, budget_seconds=args.budget_seconds)
        success = result["all_problems_verified"]
    raise SystemExit(0 if success else 2)


if __name__ == "__main__":
    main()
