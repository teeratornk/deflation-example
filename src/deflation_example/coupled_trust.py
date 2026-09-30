"""Temperature-step trust regions for prescribed-temperature coupled optimization.

Temporary step bounds constrain quadratic models only. Final KKT conditions use
the physical bounds. This policy is separate from the archived line search.
"""

import time

import numpy as np

from .coupled_control import FlowEvaluationError
from .coupled_derivatives import GaussNewtonOperator, StabilizationBranchError
from .coupled_optimizer import CoupledOptimum, NUMERICAL_POLICY, box_kkt, box_quadratic
from .coupled_sequence import RestoredEvaluation
from .validation import integer, positive_real


POLICY = {
    "identifier": "coupled-temperature-trust-v1",
    "initial_radius_K": 0.25,
    "minimum_radius_K": 1e-6,
    "maximum_radius_K": 2.0,
    "acceptance_ratio": 0.1,
    "expansion_ratio": 0.75,
    "boundary_fraction": 0.9,
    "strict_switch_kkt": 1e-4,
}


def policy_description(trial_policy):
    if trial_policy not in {"radius_rebuild", "backtrack"}:
        raise ValueError("Choose radius_rebuild or backtrack trial policy")
    return {
        **POLICY,
        "identifier": POLICY["identifier"]
        if trial_policy == "radius_rebuild"
        else "coupled-temperature-backtrack-v1",
        "trial_policy": trial_policy,
        "maximum_step_halvings": 6 if trial_policy == "backtrack" else 0,
    }


def optimality(problem, evaluation, desired, gradient, lower, upper):
    scaled = gradient / problem.weights
    scale = max(
        1.0,
        float(np.max(np.abs(evaluation.state - desired))),
        float(np.max(np.abs(scaled - (evaluation.state - desired)))),
    )
    return box_kkt(evaluation.state, scaled, lower, upper, scale), scale


def radius_update(radius, ratio, boundary, *, flow_failed=False):
    if flow_failed or not np.isfinite(ratio) or ratio < POLICY["acceptance_ratio"]:
        return radius * 0.5, False
    if ratio > POLICY["expansion_ratio"] and boundary:
        radius = min(POLICY["maximum_radius_K"], 2 * radius)
    return radius, True


def intermediate_targets(kkt, mode, qp_tolerance, linear_tolerance):
    if mode not in {"strict", "adaptive", "adaptive_projected"}:
        raise ValueError("Choose strict or adaptive intermediate accuracy")
    if mode == "strict" or kkt <= POLICY["strict_switch_kkt"]:
        return qp_tolerance, linear_tolerance, True
    # This target uses the same weight-normalized equations as outer optimality.
    target = max(qp_tolerance, min(1e-2, 0.1 * kkt**1.5))
    if mode == "adaptive_projected":
        return target, max(linear_tolerance, min(1e-2, target)), False
    return target, max(linear_tolerance, min(1e-4, 0.1 * target)), False


def minimize_trust(
    problem,
    desired,
    lower,
    upper,
    solver,
    *,
    initial=None,
    initial_evaluation=None,
    tolerance=1e-8,
    max_iterations=100,
    qp_tolerance=1e-10,
    qp_cap=100,
    secant_memory=10,
    inner_preconditioner="frozen",
    frozen_sweeps=3,
    accuracy="strict",
    trial_policy="radius_rebuild",
    qp_correction_policy="kkt_decrease",
    qp_solver="pdas",
    trial_callback=None,
    observer=None,
    callback=None,
    checkpoint=None,
    resume=None,
    budget_seconds=86400.0,
    initial_radius_K=None,
    minimum_radius_K=None,
):
    start = time.perf_counter()
    tolerance = positive_real(tolerance, "Nonlinear tolerance")
    budget_seconds = positive_real(budget_seconds, "Optimization time budget")
    initial_radius = positive_real(
        POLICY["initial_radius_K"] if initial_radius_K is None else initial_radius_K,
        "Initial trust radius",
    )
    minimum_radius = positive_real(
        POLICY["minimum_radius_K"] if minimum_radius_K is None else minimum_radius_K,
        "Minimum trust radius",
    )
    if not minimum_radius <= initial_radius <= POLICY["maximum_radius_K"]:
        raise ValueError("Trust radii must satisfy minimum <= initial <= maximum")
    if solver.device not in {"cpu", "cuda"}:
        raise ValueError("The coupled trust-region runtime requires a supported vector kernel")
    if solver.device == "cuda":
        from .coupled_cuda_solver import CudaCoupledSolver

        if not isinstance(solver, CudaCoupledSolver):
            raise ValueError("Coupled optimization requires CudaCoupledSolver or the CPU kernel")
    max_iterations = integer(max_iterations, "Nonlinear cap", 1)
    qp_cap = integer(qp_cap, "Quadratic cap", 1)
    secant_memory = integer(secant_memory, "Secant memory", 0)
    if inner_preconditioner not in {"jacobi", "frozen"}:
        raise ValueError("Choose Jacobi or frozen preconditioning")
    lower = np.broadcast_to(lower, (problem.size,)).astype(float).copy()
    upper = np.broadcast_to(upper, (problem.size,)).astype(float).copy()
    if not np.isfinite([lower, upper]).all() or np.any(lower >= upper):
        raise ValueError("Finite strictly ordered physical bounds are required")
    final_rtol = solver.rtol
    intermediate_targets(1.0, accuracy, qp_tolerance, final_rtol)
    if qp_solver not in {"pdas", "projected"}:
        raise ValueError("Choose pdas or projected quadratic solver")
    if accuracy == "adaptive_projected" and qp_solver != "projected":
        raise ValueError("Adaptive projected accuracy requires the projected quadratic solver")
    if trial_policy not in {"radius_rebuild", "backtrack"}:
        raise ValueError("Choose radius_rebuild or backtrack trial policy")
    if qp_correction_policy not in {"kkt_decrease", "allow_partition_change"}:
        raise ValueError("Unknown quadratic correction policy")
    if trial_callback is not None and not callable(trial_callback):
        raise ValueError("The trial callback must be callable")
    radius, history, secants = initial_radius, [], []
    iteration, attempts, qp_resume = 0, [], None
    prior_seconds = 0.0
    pending_qp = None
    if resume is not None:
        for name, value in (
            ("initial_radius_K", initial_radius),
            ("minimum_radius_K", minimum_radius),
        ):
            if resume.get(name, POLICY[name]) != value:
                raise ValueError("Checkpoint trust-radius policy differs")
        if (
            resume.get("qp_solver", "pdas") != qp_solver
            or resume.get("accuracy", accuracy) != accuracy
        ):
            raise ValueError("Checkpoint quadratic solver or accuracy policy differs")
        if resume.get("trial_policy", "radius_rebuild") != trial_policy:
            raise ValueError("Checkpoint trial policy differs")
        if resume.get("qp_correction_policy", "kkt_decrease") != qp_correction_policy:
            raise ValueError("Checkpoint quadratic correction policy differs")
        initial = resume["state"]
        initial_evaluation = RestoredEvaluation(initial, resume["velocity"], resume["pressure"])
        radius, history, secants = resume["radius_K"], resume["history"], resume["secants"]
        iteration, attempts, qp_resume = resume["iteration"], resume["attempts"], resume["qp"]
        prior_seconds = resume["elapsed_seconds"]
        if not minimum_radius <= radius <= POLICY["maximum_radius_K"]:
            raise ValueError("Invalid checkpoint trust radius")
    y = np.zeros(problem.size) if initial is None else np.asarray(initial, dtype=float).copy()
    if y.shape != (problem.size,) or not np.isfinite(y).all():
        raise ValueError("Initial state must be finite and match the trajectory")
    evaluation = problem.evaluate(np.clip(y, lower, upper), initial=initial_evaluation)
    objective, gradient = problem.objective_gradient(evaluation, desired)
    last_strict = (
        resume.get("last_strict", accuracy == "strict") if resume else accuracy == "strict"
    )

    def elapsed():
        return prior_seconds + time.perf_counter() - start

    previous_stop = solver.stop_requested

    def stop_requested():
        return elapsed() >= budget_seconds or (previous_stop is not None and previous_stop())

    def save(qp=None):
        if checkpoint is not None:
            retained_kkt, retained_scale = optimality(
                problem, evaluation, desired, gradient, lower, upper
            )
            checkpoint(
                {
                    "state": evaluation.state.copy(),
                    "velocity": np.stack([f.velocity for f in evaluation.flows]),
                    "pressure": np.stack([f.pressure for f in evaluation.flows]),
                    "objective": objective,
                    "kkt": retained_kkt,
                    "stationarity_scale": retained_scale,
                    "stationarity_numerator": retained_kkt["stationarity"] * retained_scale,
                    "radius_K": radius,
                    "initial_radius_K": initial_radius,
                    "minimum_radius_K": minimum_radius,
                    "iteration": iteration,
                    "history": list(history),
                    "secants": list(secants),
                    "attempts": list(attempts),
                    "qp": qp,
                    "elapsed_seconds": elapsed(),
                    "last_strict": last_strict,
                    "trial_policy": trial_policy,
                    "qp_correction_policy": qp_correction_policy,
                    "qp_solver": qp_solver,
                    "accuracy": accuracy,
                }
            )

    class BudgetReached(Exception):
        pass

    def qp_checkpoint(payload):
        nonlocal pending_qp
        pending_qp = payload
        save(payload)
        if elapsed() >= budget_seconds:
            raise BudgetReached

    status = "nonlinear_iteration_cap"
    solver.stop_requested = stop_requested
    try:
        save(qp_resume)
        while iteration < max_iterations:
            kkt, scale = optimality(problem, evaluation, desired, gradient, lower, upper)
            error = max(kkt.values())
            if error <= tolerance and last_strict:
                status = "converged"
                break
            if elapsed() >= budget_seconds:
                status = "budget_exhausted"
                break
            qtol, ltol, strict = intermediate_targets(error, accuracy, qp_tolerance, final_rtol)
            # A loose intermediate step cannot be the last numerical solve.
            force_strict = error <= tolerance and not last_strict
            if force_strict:
                qtol = min(qtol, max(np.finfo(float).eps, error * 0.1))
            H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
            if secants:
                from .coupled_secant import SecantGaussNewton

                H = SecantGaussNewton(H, secants)
            diagonal = problem.preconditioning_diagonal(evaluation, 0.0)
            delta = radius / problem.temperature_scale
            lo = np.maximum(lower - evaluation.state, -delta)
            hi = np.minimum(upper - evaluation.state, delta)
            if observer is not None:
                observer.begin_quadratic(
                    iteration,
                    len(attempts),
                    evaluation,
                    desired,
                    gradient,
                    diagonal,
                    0.0,
                    secants,
                    evaluation.state + lo,
                    evaluation.state + hi,
                )
                observer.record["quadratics"][-1].update(
                    trust_radius_K=radius,
                    linear_tolerance=ltol,
                    qp_tolerance=qtol,
                    strict_accuracy=strict,
                )
                observer.save()
            factory = None
            if inner_preconditioner == "frozen":
                from .coupled_frozen_preconditioner import frozen_preconditioner_factory

                factory = frozen_preconditioner_factory(problem, evaluation, sweeps=frozen_sweeps)
            solver.rtol = ltol
            pending_qp = None
            if qp_solver == "projected":
                from .box_projected_cg import box_projected_cg

                solve_quadratic, extra = box_projected_cg, {}
            else:
                solve_quadratic, extra = box_quadratic, {"correction_policy": qp_correction_policy}
            qp = solve_quadratic(
                H,
                gradient,
                diagonal,
                lo,
                hi,
                solver,
                tolerance=qtol,
                max_steps=qp_cap,
                preconditioner_factory=factory,
                kkt_evaluator=lambda x, g: box_kkt(x, g / problem.weights, lo, hi, scale),
                checkpoint=qp_checkpoint,
                resume=qp_resume,
                observer=observer,
                **extra,
            )
            solver.rtol = final_rtol
            qp_resume = None
            attempt = {
                "radius_K": radius,
                "qp_status": qp.status,
                "qp_kkt": qp.kkt,
                "qp_history": qp.history,
                "qp_tolerance": qtol,
                "linear_tolerance": ltol,
                "strict_accuracy": strict,
                "qp_solver": qp_solver,
                "trials": [],
            }
            attempts.append(attempt)
            if qp.status != "converged":
                status = (
                    "budget_exhausted"
                    if qp.status == "linear_budget_exhausted"
                    else "quadratic_" + qp.status
                )
                save()
                break
            attempt["trial_policy"] = trial_policy
            trial = None
            accepted = False
            strict_zero = force_strict and not np.any(qp.x)
            if strict_zero:
                # Strict zero increment independently verifies stationarity.
                last_strict = True
                save()
                continue
            exhausted = False
            trial_count = 7 if trial_policy == "backtrack" else 1
            for backtrack in range(trial_count):
                if stop_requested():
                    status, exhausted = "budget_exhausted", True
                    break
                step = 2.0 ** (-backtrack)
                candidate = np.clip(evaluation.state + step * qp.x, lower, upper)
                applied_step = candidate - evaluation.state
                # Evaluate the actual increment, including floating-point projection.
                predicted = -float(
                    gradient @ applied_step + 0.5 * applied_step @ (H @ applied_step)
                )
                physical_step = float(np.max(np.abs(applied_step))) * problem.temperature_scale
                attempt.update(predicted_reduction=predicted, temperature_step_K=physical_step)
                event = {
                    "iteration": iteration,
                    "attempt": len(attempts) - 1,
                    "trial": backtrack,
                    "step": step,
                    "radius_K": radius,
                    "predicted_reduction": predicted,
                    "temperature_step_K": physical_step,
                }
                if trial_callback is not None:
                    trial_callback({**event, "phase": "started"}, evaluation, candidate, None)
                failed_flow = None
                began = time.perf_counter()
                if not np.isfinite(predicted) or predicted <= 0:
                    outcome = {"status": "nonpositive_predicted_reduction"}
                else:
                    try:
                        trial = problem.evaluate(candidate, initial=evaluation)
                        value, derivative = problem.objective_gradient(trial, desired)
                        change = problem.objective_difference(trial, evaluation, desired)
                        finite = (
                            np.isfinite([value, change]).all() and np.isfinite(derivative).all()
                        )
                        ratio = -change / predicted if finite else float("nan")
                        trial_kkt, _ = optimality(problem, trial, desired, derivative, lower, upper)
                        new_radius, accepted = radius_update(
                            radius, ratio, physical_step >= POLICY["boundary_fraction"] * radius
                        )
                        allowance = (
                            NUMERICAL_POLICY["roundoff_epsilon_factor"]
                            * np.finfo(float).eps
                            * max(1.0, abs(objective))
                        )
                        roundoff = (
                            finite
                            and not accepted
                            and error <= NUMERICAL_POLICY["roundoff_kkt_threshold"]
                            and predicted <= allowance
                            and change <= allowance
                            and max(trial_kkt.values())
                            <= NUMERICAL_POLICY["roundoff_kkt_contraction"] * error
                        )
                        accepted = accepted or roundoff
                        outcome = {
                            "status": "roundoff_kkt_decrease"
                            if roundoff
                            else "decrease"
                            if accepted
                            else "insufficient_decrease",
                            "objective": value,
                            "objective_change": change,
                            "reduction_ratio": ratio,
                            "maximum_kkt": max(trial_kkt.values()),
                            "evaluation_seconds": trial.seconds,
                        }
                        if accepted and backtrack:
                            radius = max(minimum_radius, min(radius, 2 * physical_step))
                        elif accepted and not roundoff:
                            radius = new_radius
                    except FlowEvaluationError as failure:
                        failed_flow = failure.result
                        if failed_flow.status == "budget_exhausted":
                            status, exhausted = "budget_exhausted", True
                        outcome = {
                            "status": "flow_" + failure.result.status,
                            "slab": failure.slab,
                            "metrics": failure.metrics,
                            "flow_history": failure.result.history,
                        }
                    except StabilizationBranchError:
                        outcome = {"status": "stabilization_branch_switch"}
                trial_row = {
                    **event,
                    **outcome,
                    "accepted": accepted,
                    "trial_seconds": time.perf_counter() - began,
                }
                attempt["trials"].append(trial_row)
                if trial_callback is not None:
                    trial_callback(
                        {**trial_row, "phase": "finished"}, evaluation, candidate, failed_flow
                    )
                if accepted or exhausted:
                    break
            if exhausted:
                save()
                break
            if not accepted:
                radius *= 0.5
            if accepted:
                if secant_memory:
                    secants.append((trial.state - evaluation.state, derivative - gradient))
                    secants = secants[-secant_memory:]
                row = {
                    "iteration": iteration,
                    "objective": objective,
                    "kkt": kkt,
                    "evaluation_seconds": evaluation.seconds,
                    "attempts": list(attempts),
                }
                evaluation, objective, gradient = trial, value, derivative
                last_strict = strict
                retained_kkt, retained_scale = optimality(
                    problem, evaluation, desired, gradient, lower, upper
                )
                row["retained"] = {
                    "objective": objective,
                    "kkt": retained_kkt,
                    "stationarity_scale": retained_scale,
                    "stationarity_numerator": retained_kkt["stationarity"] * retained_scale,
                }
                history.append(row)
                if callback is not None:
                    callback(row, evaluation)
                iteration += 1
                attempts = []
            if radius < minimum_radius:
                status = "trust_radius_exhausted"
                break
            save()
    except BudgetReached:
        status = "budget_exhausted"
        attempts.append(
            {
                "radius_K": radius,
                "qp_status": status,
                "qp_kkt": pending_qp["kkt"],
                "qp_history": pending_qp["history"],
                "qp_tolerance": qtol,
                "linear_tolerance": ltol,
                "strict_accuracy": strict,
                "trials": [],
            }
        )
    finally:
        solver.rtol = final_rtol
        solver.stop_requested = previous_stop
    if attempts:
        history.append(
            {
                "iteration": iteration,
                "objective": objective,
                "kkt": optimality(problem, evaluation, desired, gradient, lower, upper)[0],
                "evaluation_seconds": evaluation.seconds,
                "attempts": attempts,
            }
        )
    kkt, _ = optimality(problem, evaluation, desired, gradient, lower, upper)
    if max(kkt.values()) <= tolerance and last_strict and status == "nonlinear_iteration_cap":
        status = "converged"
    return CoupledOptimum(evaluation, objective, gradient, kkt, status, history, elapsed())
