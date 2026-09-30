"""Bounded trust-radius experiment from a retained unsuccessful trajectory.

The physical equations and final criteria remain fixed. The diagnostic discards
old secants and recycling history equally in every arm; its cost is not a
complete-sequence timing and it never overwrites the source attempt.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_optimize import adjoint_acceptance, equation_acceptance, load_problem
from .coupled_sequence import RestoredEvaluation
from .coupled_targets import desired_temperature
from .coupled_trust import minimize_trust
from .reporting import environment, file_sha256, write_arrays, write_report
from .study_solvers import StudySolver
from .validation import positive_real


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--position", type=int, default=1)
    parser.add_argument("--minimum-radius-K", type=float, choices=(1e-6, 1e-10), required=True)
    parser.add_argument("--budget-seconds", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    positive_real(args.budget_seconds, "Diagnostic time budget")
    source = json.loads(args.record.read_text())
    cfg = source["configuration"]
    if source["status"] == "running" or not 0 <= args.position < len(source["cases"]):
        raise ValueError("Select a retained target from a terminated comparison")
    if source["cases"][args.position]["status"] != "trust_radius_exhausted":
        raise ValueError("This diagnostic addresses trust-radius exhaustion")
    path = args.record.parent / f"target-{args.position:02d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        saved = {k: archive[k].copy() for k in ("state", "velocity", "pressure", "desired")}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-radius-restart-v1",
        "status": "running",
        "verified": False,
        "environment": environment(),
        "source_configuration": cfg,
        "record_sha256": file_sha256(args.record),
        "field_sha256": file_sha256(path),
        "position": args.position,
        "target": source["cases"][args.position]["target"],
        "restart": {
            "initial_radius_K": 1e-6,
            "minimum_radius_K": args.minimum_radius_K,
            "outer_cap": 40,
            "budget_seconds": args.budget_seconds,
            "method": "frozen_preconditioned_CG",
            "device": "cpu",
            "rank": 0,
            "retained_secants": 0,
            "retained_recycling_vectors": 0,
        },
        "scope": "Local optimizer restart from the same failed field. Every arm discards old secants, uses rank zero and the unchanged physical equations and final criteria. This is neither a source-compatible recovery nor a complete-sequence timing.",
    }
    write_report(args.output / "record.json", report)
    start, solver = time.perf_counter(), None
    try:
        with threadpool_limits(cfg["threads"]):
            problem, baseline = load_problem(cfg)
            if baseline["baseline_sha256"] != source["baseline_sha256"]:
                raise ValueError("The physical baseline differs")
            desired = desired_temperature(
                problem, report["target"], cfg["target_count"], cfg["target_startup_s"]
            )
            if not np.array_equal(desired, saved["desired"]):
                raise ValueError("The target differs from the saved field")
            guess = RestoredEvaluation(saved["state"], saved["velocity"], saved["pressure"])
            bounds = tuple(
                (v - problem.temperature_offset) / problem.temperature_scale
                for v in (cfg["lower_K"], cfg["queries"][args.position]["upper_K"])
            )
            solver = StudySolver(
                "jacobi",
                rank=0,
                rtol=cfg["inner_tolerance"],
                maxiter=cfg["inner_cap"],
                cg_factor=0.1,
                refresh=cfg["inner_refresh"],
                residual_policy="refine",
            )
            problem.evaluation_callback = lambda row: write_report(
                args.output / "evaluation-progress.json", row
            )
            optimization_start = time.perf_counter()
            problem.stop_requested = lambda: (
                time.perf_counter() - optimization_start >= args.budget_seconds
            )

            def trial(row, retained, candidate, failure):
                write_report(args.output / "trial-progress.json", row)

            def checkpoint(payload):
                write_report(
                    args.output / "optimization-progress.json",
                    {
                        k: payload[k]
                        for k in (
                            "iteration",
                            "radius_K",
                            "objective",
                            "kkt",
                            "elapsed_seconds",
                            "initial_radius_K",
                            "minimum_radius_K",
                        )
                    },
                )
                write_arrays(
                    args.output / "retained.npz",
                    state=payload["state"],
                    velocity=payload["velocity"],
                    pressure=payload["pressure"],
                )

            result = minimize_trust(
                problem,
                desired,
                *bounds,
                solver,
                initial=saved["state"],
                initial_evaluation=guess,
                tolerance=cfg["nonlinear_tolerance"],
                max_iterations=40,
                qp_tolerance=cfg["qp_tolerance"],
                qp_cap=cfg["qp_cap"],
                secant_memory=cfg["secant_memory"],
                inner_preconditioner=cfg["inner_preconditioner"],
                frozen_sweeps=cfg["frozen_sweeps"],
                accuracy=cfg["trust_accuracy"],
                trial_policy=cfg["trial_policy"],
                qp_solver=cfg["qp_solver"],
                qp_correction_policy=cfg["qp_correction_policy"],
                trial_callback=trial,
                checkpoint=checkpoint,
                budget_seconds=args.budget_seconds,
                initial_radius_K=1e-6,
                minimum_radius_K=args.minimum_radius_K,
            )
            report.update(
                status=result.status,
                objective=result.objective,
                kkt=result.kkt,
                history=result.history,
                optimizer_seconds=result.seconds,
            )
            problem.stop_requested = None
            report["equations"] = problem.verify(result.evaluation, local_mass=True)
            report["adjoint"] = problem.verify_adjoint(result.evaluation, desired)
            report["verified"] = bool(
                result.status == "converged"
                and max(result.kkt.values()) <= cfg["nonlinear_tolerance"]
                and equation_acceptance(report["equations"], cfg)
                and adjoint_acceptance(report["adjoint"])
            )
            final = args.output / "result.npz"
            write_arrays(
                final,
                state=result.evaluation.state,
                control=result.evaluation.control,
                velocity=np.stack([f.velocity for f in result.evaluation.flows]),
                pressure=np.stack([f.pressure for f in result.evaluation.flows]),
                gradient=result.gradient,
                desired=desired,
            )
            report["result_sha256"] = file_sha256(final)
    except Exception as error:
        report.update(status="diagnostic_error", error_type=type(error).__name__, error=str(error))
        if hasattr(error, "result"):
            report.update(flow_status=error.result.status, slab=error.slab, metrics=error.metrics)
        raise
    finally:
        if solver is not None:
            solver.close()
        report["seconds"] = time.perf_counter() - start
        write_report(args.output / "record.json", report)
    if not report["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
