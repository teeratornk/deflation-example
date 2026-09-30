"""Bounded, recoverable development runs of coupled temperature trust regions."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from .coupled_initial_state import snapshot_initial_guess
from .coupled_nominal_krylov import configured_krylov_reference
from .coupled_optimize import (
    adjoint_acceptance,
    equation_acceptance,
    load_problem,
    observe_linear_solves,
)
from .coupled_recovery import RecoveryStore, identity
from .coupled_regularization_complete import checked_settings, complete_configuration
from .coupled_sequence import RestoredEvaluation
from .coupled_targets import desired_temperature
from .coupled_trust import minimize_trust, policy_description
from .coupled_trial_capture import TrialCapture
from .coupled_trace import CoupledTrace
from .memory import ProcessMemory
from .reporting import environment, file_sha256, write_arrays, write_report
from .study_solvers import ArrayReference, StudySolver


def run(
    cfg,
    output,
    *,
    resume_from=None,
    budget_seconds=86400.0,
    prior_attempt_seconds=None,
    problem_loader=load_problem,
):
    """A diagnostic interval, never silently pooled with uninterrupted timings.

    Use a new output directory for each attempt. A resume copies the verified
    immutable reference, reevaluates the retained flow and rebuilds numerical
    factors. Those costs remain in the new interval. Earlier attempts remain
    separate; their final wall intervals must be included when aggregating cost.
    """
    cfg = {
        "flow_continuation": False,
        "linear_heartbeat_seconds": 30.0,
        "trial_policy": "radius_rebuild",
        "capture_trials": False,
        "capture_linear_systems": False,
        "qp_correction_policy": "kkt_decrease",
        "qp_solver": "pdas",
        "device": "cpu",
        "frozen_layout": "serial",
        **cfg,
    }
    if cfg.get("initial_state_policy", "snapshot") not in {"snapshot", "physical_initial"}:
        raise ValueError("Choose snapshot or physical_initial initialization")
    if cfg["trial_policy"] not in {"radius_rebuild", "backtrack"}:
        raise ValueError("Unknown trial policy")
    if cfg["qp_correction_policy"] not in {"kkt_decrease", "allow_partition_change"}:
        raise ValueError("Unknown quadratic correction policy")
    if any(not isinstance(cfg[k], bool) for k in ("capture_trials", "capture_linear_systems")):
        raise ValueError("Diagnostic capture settings must be Boolean")
    transfer = cfg.get("reference_transfer", "full")
    if transfer not in {"full", "sequential"} or (
        transfer == "sequential" and cfg["method"] != "reference"
    ):
        raise ValueError("Sequential transfer applies only to a reference method")
    if cfg.get("optimization_only_budget", False) and resume_from is not None:
        raise ValueError("Optimization-only budgets require an uninterrupted fresh run")
    if cfg.get("reference_transfer", "full") == "sequential" and resume_from is not None:
        raise ValueError("Sequential-transfer comparisons require an uninterrupted fresh run")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = environment()
    fp = identity(cfg, env["source_sha256"])
    store = RecoveryStore(output / "recovery", fp)
    previous_store = (
        None if resume_from is None else RecoveryStore(Path(resume_from) / "recovery", fp)
    )
    saved = None if previous_store is None else previous_store.load()
    if previous_store is not None and saved is None:
        raise ValueError("Resume requires a complete checkpoint")
    if saved is not None and saved.get("finished"):
        raise ValueError("A completed diagnostic cannot be resumed")
    prior_total = 0.0
    if saved is not None:
        previous_record = json.loads((Path(resume_from) / "record.json").read_text())
        interval = previous_record.get("attempt_seconds", prior_attempt_seconds)
        if (
            interval is None
            or not np.isfinite(interval)
            or interval < saved["attempt_elapsed_seconds"]
        ):
            raise ValueError("A killed run needs its full scheduler-measured elapsed seconds")
        prior_total = previous_record.get("prior_attempt_seconds", 0.0) + interval
        if prior_total >= budget_seconds:
            raise ValueError("The cumulative diagnostic time budget is exhausted")
    record = {
        "schema": "coupled-trust-development-v1",
        "policy": {
            **policy_description(cfg["trial_policy"]),
            "qp_correction_policy": cfg["qp_correction_policy"],
        },
        "configuration": cfg,
        "environment": env,
        "status": "running",
        "runtime_policy": "cpu-cg-heartbeat-cooperative-deadline-v1",
        "scope": "Development comparison with checkpoint I/O; separate from frozen timing studies.",
        "resumed": saved is not None,
        "prior_attempt_seconds": prior_total,
        "resume_record_sha256": None
        if resume_from is None
        else file_sha256(Path(resume_from) / "record.json"),
        "timing_boundary": "Assembly, reference processing, optimization, checkpoints, verification and cleanup. Calibration and process preparation are reported separately; interrupted intervals are retained.",
    }
    write_report(output / "record.json", record)
    tick = time.perf_counter()
    sampler = ProcessMemory(
        "cpu" if cfg["device"] == "cpu" else "cuda", cfg.get("memory_interval", 0.1)
    )
    sampler.start()
    record["execution_environment"] = getattr(sampler, "device_metadata", {})
    if cfg["device"] != "cpu":
        import cupy as cp

        record["execution_environment"].update(
            cupy_version=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion()
        )
    record["process_preparation_seconds"] = time.perf_counter() - tick
    start = time.perf_counter()
    solver, reference, problem = None, None, None
    components = {}
    cases = [] if saved is None else list(saved["cases"])
    position = 0 if saved is None else saved["position"]
    try:
        with threadpool_limits(cfg["threads"]):
            tick = time.perf_counter()
            problem, baseline = problem_loader(cfg)
            if cfg.get("cooperative_flow_deadline", False):
                problem.stop_requested = lambda: (
                    prior_total + time.perf_counter() - start >= budget_seconds
                )
            record.update(
                baseline_sha256=baseline["baseline_sha256"],
                calibration_seconds=baseline["seconds"],
                state_dofs_per_problem=problem.size,
            )
            if saved is not None and saved["baseline_sha256"] != baseline["baseline_sha256"]:
                raise ValueError("The recovered baseline differs")
            problem.evaluation_callback = lambda row: write_report(
                output / "evaluation-progress.json", row
            )
            guess = None
            if saved is None:
                if cfg.get("initial_state_policy", "snapshot") == "physical_initial":
                    state = np.tile(problem.initial, problem.slabs)
                    lower = (
                        cfg["lower_K"] - problem.temperature_offset
                    ) / problem.temperature_scale
                    upper = (
                        cfg["queries"][0]["upper_K"] - problem.temperature_offset
                    ) / problem.temperature_scale
                    if np.any(state < lower) or np.any(state > upper):
                        raise ValueError("The physical initial temperature violates the bounds")
                    guess = RestoredEvaluation(
                        state,
                        np.repeat(problem.initial_flow.velocity[None], problem.slabs, axis=0),
                        np.repeat(problem.initial_flow.pressure[None], problem.slabs, axis=0),
                    )
                    record["initial_state"] = {
                        "policy": "physical_initial",
                        "retained_optimizer_history": False,
                    }
                else:
                    guess, record["initial_state"] = snapshot_initial_guess(
                        cfg, problem, baseline, 0
                    )
            components["model_and_initialization"] = time.perf_counter() - tick
            tick = time.perf_counter()
            if cfg["method"] == "reference":
                if saved is None:
                    reference = configured_krylov_reference(
                        problem,
                        {**cfg, "device": "cpu", "reference_transfer": "full"},
                        initial_guess=guess,
                        verification_callback=lambda rows: write_report(
                            output / "reference-equations.json", {"equations": rows}
                        ),
                    )
                else:
                    path = Path(resume_from) / "reference.npz"
                    if file_sha256(path) != saved["reference_sha256"]:
                        raise ValueError("The recovered reference differs")
                    with np.load(path, allow_pickle=False) as data:
                        reference = ArrayReference(
                            data["basis"].copy(), saved["reference_description"]
                        )
                write_arrays(output / "reference.npz", basis=reference.basis)
                if cfg.get("reference_transfer", "full") == "sequential":
                    from .coupled_reference import SequentialReference

                    # Archive the same initial full basis, then release its in-memory
                    # values after the first restriction. Never serialize None as an
                    # object array or restore lost entries during a later solve.
                    reference = SequentialReference(reference, problem.size)
                record.update(
                    reference_sha256=file_sha256(output / "reference.npz"),
                    reference_description=reference.description,
                )
            components["reference_construction_or_restore"] = time.perf_counter() - tick
            if cfg.get("optimization_only_budget", False):
                record["optimization_budget_seconds"] = budget_seconds
                record["budget_preparation_seconds"] = time.perf_counter() - start
                budget_seconds += record["budget_preparation_seconds"]
                record["budget_scope"] = (
                    "Optimization has its own budget; preparation and reference construction "
                    "remain included in attempt_seconds. Preparation has an equal separate cap."
                )
            solver_type, options = StudySolver, {}
            if cfg["device"] == "cuda":
                from .coupled_cuda_solver import CudaCoupledSolver

                solver_type = CudaCoupledSolver
                options["frozen_layout"] = cfg["frozen_layout"]
            elif cfg["device"] == "hybrid":
                from .coupled_hybrid_solver import HybridCoupledSolver

                solver_type = HybridCoupledSolver
                options.update(block_device="cpu", coarse_device="cuda")
            elif cfg["device"] != "cpu":
                raise ValueError("Choose cpu, cuda or hybrid execution")
            record["runtime_policy"] = f"{cfg['device']}-cg-heartbeat-cooperative-deadline-v2"
            solver = solver_type(
                cfg["method"],
                reference=reference,
                rank=cfg["rank"],
                window=cfg["recycle_window"],
                rtol=cfg["inner_tolerance"],
                maxiter=cfg["inner_cap"],
                cg_factor=0.1,
                refresh=cfg["inner_refresh"],
                residual_policy="refine",
                stop_requested=lambda: prior_total + time.perf_counter() - start >= budget_seconds,
                **options,
            )
            if saved is not None:
                solver.import_history(saved["recycling"])
                solver.previous = saved["previous_indices"]
            observe_linear_solves(
                solver,
                output / "linear-progress.json",
                heartbeat_seconds=cfg["linear_heartbeat_seconds"],
            )
            record.update(components_seconds=components)
            write_report(output / "record.json", record)
            optimizer_resume = None if saved is None else saved["optimizer"]
            previous = None
            if (
                saved is not None
                and optimizer_resume is None
                and saved.get("previous_solution") is not None
            ):
                p = saved["previous_solution"]
                previous = RestoredEvaluation(p["state"], p["velocity"], p["pressure"])
            checkpoint_seconds = 0.0

            def save_optimizer(
                payload, *, next_position=None, previous_solution=None, finished=False
            ):
                nonlocal checkpoint_seconds
                began = time.perf_counter()
                store.save(
                    {
                        "position": position if next_position is None else next_position,
                        "optimizer": payload,
                        "cases": list(cases),
                        "finished": finished,
                        "previous_solution": previous_solution,
                        "recycling": solver.export_history(),
                        "previous_indices": solver.previous,
                        "baseline_sha256": baseline["baseline_sha256"],
                        "reference_sha256": record.get("reference_sha256"),
                        "reference_description": record.get("reference_description"),
                        "attempt_elapsed_seconds": time.perf_counter() - start,
                    }
                )
                checkpoint_seconds += time.perf_counter() - began

            for position in range(position, len(cfg["queries"])):
                query = cfg["queries"][position]
                desired = desired_temperature(
                    problem, query["target"], cfg["target_count"], cfg["target_startup_s"]
                )
                lower = (cfg["lower_K"] - problem.temperature_offset) / problem.temperature_scale
                upper = (query["upper_K"] - problem.temperature_offset) / problem.temperature_scale
                began = time.perf_counter()
                capture = (
                    TrialCapture(output / "trial-capture" / f"target-{position:02d}", fp)
                    if cfg["capture_trials"]
                    else None
                )
                trace = (
                    CoupledTrace(
                        output / f"inactive-trace-{position:02d}", cfg, baseline["baseline_sha256"]
                    )
                    if cfg["capture_linear_systems"]
                    else None
                )
                if trace is not None:
                    trace.record["resumed"] = saved is not None
                result = minimize_trust(
                    problem,
                    desired,
                    lower,
                    upper,
                    solver,
                    initial=None
                    if guess is None and previous is None
                    else (guess if previous is None else previous).state,
                    initial_evaluation=guess if previous is None else previous,
                    tolerance=cfg["nonlinear_tolerance"],
                    max_iterations=cfg["nonlinear_cap"],
                    qp_tolerance=cfg["qp_tolerance"],
                    qp_cap=cfg["qp_cap"],
                    secant_memory=cfg["secant_memory"],
                    inner_preconditioner=cfg["inner_preconditioner"],
                    frozen_sweeps=cfg["frozen_sweeps"],
                    accuracy=cfg["trust_accuracy"],
                    trial_policy=cfg["trial_policy"],
                    qp_correction_policy=cfg["qp_correction_policy"],
                    qp_solver=cfg["qp_solver"],
                    trial_callback=capture,
                    observer=trace,
                    checkpoint=save_optimizer,
                    resume=optimizer_resume,
                    initial_radius_K=cfg.get("initial_radius_K"),
                    minimum_radius_K=cfg.get("minimum_radius_K"),
                    budget_seconds=max(
                        1e-6, budget_seconds - prior_total - (time.perf_counter() - start)
                    )
                    + (0.0 if optimizer_resume is None else optimizer_resume["elapsed_seconds"]),
                    callback=lambda row, ev: write_report(
                        output / "optimization-progress.json", {"position": position, **row}
                    ),
                )
                checks = problem.verify(
                    result.evaluation, local_mass=cfg.get("local_mass_diagnostics", False)
                )
                adjoint = problem.verify_adjoint(result.evaluation, desired)
                verified = (
                    result.status == "converged"
                    and equation_acceptance(checks, cfg)
                    and adjoint_acceptance(adjoint)
                )
                if trace is not None:
                    trace.record.update(
                        optimizer_status=result.status, optimization_verified=verified
                    )
                    trace.finish()
                case = {
                    "position": position,
                    "target": query["target"],
                    "status": result.status,
                    "verified": verified,
                    "kkt": result.kkt,
                    "equations": checks,
                    "adjoint": adjoint,
                    "objective": result.objective * problem.objective_scale,
                    "history": result.history,
                    "seconds_this_attempt": time.perf_counter() - began,
                    "trial_capture_seconds": 0.0 if capture is None else capture.seconds,
                    "inner_iterations": sum(
                        step.get("linear_iterations", 0)
                        for outer in result.history
                        for attempt in outer["attempts"]
                        for step in attempt["qp_history"]
                    ),
                }
                cases.append(case)
                fields = {
                    "state": result.evaluation.state,
                    "control": result.evaluation.control,
                    "desired": desired,
                    "velocity": np.stack([f.velocity for f in result.evaluation.flows]),
                    "pressure": np.stack([f.pressure for f in result.evaluation.flows]),
                }
                write_arrays(output / f"target-{position:02d}.npz", **fields)
                if not verified:
                    record["status"] = (
                        result.status if result.status != "converged" else "verification_failed"
                    )
                    for later in range(position + 1, len(cfg["queries"])):
                        cases.append(
                            {
                                "position": later,
                                "target": cfg["queries"][later]["target"],
                                "status": "not_run_after_failure",
                                "verified": False,
                                "history": [],
                                "inner_iterations": 0,
                                "seconds_this_attempt": 0.0,
                            }
                        )
                    break
                previous = result.evaluation
                guess, optimizer_resume = None, None
                save_optimizer(
                    None,
                    next_position=position + 1,
                    previous_solution=fields,
                    finished=position + 1 == len(cfg["queries"]),
                )
            else:
                record["status"] = "complete"
            record["checkpoint_seconds"] = checkpoint_seconds
    except Exception as failure:
        record.update(
            status="diagnostic_error", error_type=type(failure).__name__, error=str(failure)
        )
        raise
    finally:
        tick = time.perf_counter()
        if solver is not None:
            solver.close()
        solver = reference = None
        problem = result = previous = guess = None
        gc.collect()
        components["cleanup"] = time.perf_counter() - tick
        record.update(
            cases=cases,
            verified_problems=sum(c["verified"] for c in cases),
            all_problems_verified=record["status"] == "complete"
            and len(cases) == len(cfg["queries"])
            and all(c["verified"] for c in cases),
            attempt_seconds=time.perf_counter() - start,
            memory=sampler.finish(),
        )
        components["optimization_verification_and_other"] = record["attempt_seconds"] - sum(
            components.values()
        )
        record["components_seconds"] = components
        record["cumulative_attempt_seconds"] = prior_total + record["attempt_seconds"]
        write_report(output / "record.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "screen",
        "settings",
        "optimization",
        "baseline",
        "initial-snapshot",
        "initial-assessment",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("jacobi", "frozen", "recycling", "reference"), default="frozen"
    )
    parser.add_argument(
        "--accuracy", choices=("strict", "adaptive", "adaptive_projected"), default="strict"
    )
    parser.add_argument("--qp-solver", choices=("pdas", "projected"), default="pdas")
    parser.add_argument("--device", choices=("cpu", "cuda", "hybrid"), default="cpu")
    parser.add_argument("--frozen-layout", choices=("serial", "block_diagonal"), default="serial")
    parser.add_argument("--budget-seconds", type=float, default=86400)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--continuation", action="store_true")
    parser.add_argument(
        "--trial-policy", choices=("radius_rebuild", "backtrack"), default="radius_rebuild"
    )
    parser.add_argument("--capture-trials", action="store_true")
    parser.add_argument("--capture-linear-systems", action="store_true")
    parser.add_argument(
        "--qp-correction-policy",
        choices=("kkt_decrease", "allow_partition_change"),
        default="kkt_decrease",
    )
    parser.add_argument("--complete-sequence", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--prior-attempt-seconds", type=float)
    parser.add_argument("--repetition", type=int, default=0)
    args = parser.parse_args()
    settings = checked_settings(args.screen, args.settings)
    cfg = complete_configuration(
        json.loads((args.optimization / "record.json").read_text()),
        settings,
        args.arm,
        args.repetition,
    )
    if not args.complete_sequence:
        cfg["queries"] = cfg["queries"][:1]
    cfg.update(
        baseline_directory=str(args.baseline),
        initial_state_snapshot=str(args.initial_snapshot),
        initial_state_assessment=str(args.initial_assessment),
        trust_accuracy=args.accuracy,
        flow_continuation=args.continuation,
        optimizer_policy=policy_description(args.trial_policy)["identifier"],
        trial_policy=args.trial_policy,
        capture_trials=args.capture_trials,
        capture_linear_systems=args.capture_linear_systems,
        qp_correction_policy=args.qp_correction_policy,
        qp_solver=args.qp_solver,
        device=args.device,
        frozen_layout=args.frozen_layout,
    )
    if args.rank is not None:
        from .validation import integer

        cfg["rank"] = integer(args.rank, "Reference rank", 1)
    # Resolve through the same configuration type used by the other runners.
    cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    result = run(
        cfg,
        args.output,
        resume_from=args.resume_from,
        prior_attempt_seconds=args.prior_attempt_seconds,
        budget_seconds=args.budget_seconds,
    )
    if not result["all_problems_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
