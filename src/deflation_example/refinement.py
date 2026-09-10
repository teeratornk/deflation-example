"""Bounded residual correction with an unchanged original-system acceptance test."""

import time

import numpy as np

from .solvers import LinearResult, independent_residual
from .timing import PHASES
from .validation import integer, positive_real, real_array


REFINEMENT_POLICY = "initial-guard-original-residual-error-equations-v2"


def verified_refinement(solve, A, b, initial, tolerance, maxiter, max_corrections=4):
    """Solve error equations after a rejected candidate, within one iteration cap.

    ``solve`` receives (rhs, initial, relative_target, remaining_iterations).
    Each error equation requests a factor-ten residual reduction. Only the
    independently recomputed residual of the original system accepts the result.
    Nested kernel timers, restriction costs, and additional CPU verification
    partition the returned complete callback time.
    """
    start = time.perf_counter()
    tolerance = positive_real(tolerance, "Final relative tolerance")
    maxiter = integer(maxiter, "Shared iteration cap", 1)
    max_corrections = integer(max_corrections, "Correction cap")
    b = real_array(b, "Right-hand side")
    current = np.zeros_like(b) if initial is None else real_array(initial, "Initial state").copy()
    if b.ndim != 1 or current.shape != b.shape or not np.isfinite([b, current]).all():
        raise ValueError("Finite matching right-hand side and initial state vectors are required")
    if A.shape != (len(b), len(b)):
        raise ValueError("Operator dimensions must match the right-hand side")
    tick = time.perf_counter()
    initial_residual = independent_residual(A, current, b)
    verification_seconds = time.perf_counter() - tick
    if initial_residual <= tolerance:
        elapsed = time.perf_counter() - start
        components = dict.fromkeys(PHASES, 0.0)
        components["verification"] = verification_seconds
        components["host_bookkeeping"] = elapsed - verification_seconds
        return LinearResult(current, 0, initial_residual, "converged"), {
            "components_seconds": components,
            "total_seconds": elapsed,
            "callback_seconds": elapsed,
            "restriction_and_transfer_seconds": 0.0,
            "iteration_rtol": None,
            "acceptance_rtol": tolerance,
            "refinement_attempts": [],
            "refinement_verification_seconds": verification_seconds,
            "refinement_policy": REFINEMENT_POLICY,
            "initial_residual": initial_residual,
            "initial_guess_accepted": True,
            "best_candidate_attempt": None,
            "newly_inactive": None,
            "newly_active": None,
            "input_basis_columns": 0,
            "restricted_basis_bytes": 0,
            "cached_operator_product_bytes": 0,
        }
    rhs = b.copy()
    guess = None if initial is None else current.copy()
    local_tolerance = tolerance
    iterations = 0
    records, metrics = [], []
    previous_residual = initial_residual
    best_result, best_attempt = None, None
    final_status = "residual_failed"
    for attempt in range(max_corrections + 1):
        result, timing = solve(rhs, guess, local_tolerance, maxiter - iterations)
        iterations += result.iterations
        metrics.append(timing)
        tick = time.perf_counter()
        candidate = result.x if attempt == 0 else current + result.x
        residual = independent_residual(A, candidate, b)
        records.append(
            {
                "attempt": attempt,
                "kind": "state" if attempt == 0 else "error_correction",
                "iterations": result.iterations,
                "status": result.status,
                "local_relative_target": local_tolerance,
                "local_residual": result.residual,
                "original_residual": residual,
                "rank": result.rank,
                "coarse_condition": result.coarse_condition,
                "termination_test": timing.get("termination_test"),
                "iteration_rtol": timing.get("iteration_rtol"),
                "native_absolute_tolerance": timing.get("native_absolute_tolerance"),
                "components_seconds": timing["components_seconds"],
                "kernel_seconds": timing["total_seconds"],
            }
        )
        improved = np.isfinite(residual) and residual < previous_residual
        if improved:
            current, previous_residual = np.array(candidate, copy=True), residual
            best_result, best_attempt = result, attempt
        if residual <= tolerance:
            final_status = "converged"
        elif not improved:
            final_status = "residual_stagnation"
        elif iterations >= maxiter:
            final_status = "maxiter"
        elif result.status == "breakdown" and timing.get("termination_test") != "nonpositive_rz":
            final_status = "breakdown"
        else:
            final_status = "residual_failed"
        if final_status != "residual_failed" or attempt == max_corrections:
            verification_seconds += time.perf_counter() - tick
            break
        rhs = b - A @ current
        guess, local_tolerance = None, 0.1
        verification_seconds += time.perf_counter() - tick

    combined = dict(metrics[-1])
    components = {key: sum(m["components_seconds"][key] for m in metrics) for key in PHASES}
    components["verification"] += verification_seconds
    elapsed = time.perf_counter() - start
    bookkeeping = elapsed - sum(m["callback_seconds"] for m in metrics) - verification_seconds
    components["host_bookkeeping"] += max(0.0, bookkeeping)
    combined.update(
        components_seconds=components,
        total_seconds=sum(components.values()),
        restriction_and_transfer_seconds=sum(
            m["restriction_and_transfer_seconds"] for m in metrics
        ),
        callback_seconds=elapsed,
        iteration_rtol=metrics[0]["iteration_rtol"],
        acceptance_rtol=tolerance,
        refinement_attempts=records,
        refinement_verification_seconds=verification_seconds,
        refinement_policy=REFINEMENT_POLICY,
        initial_residual=initial_residual,
        initial_guess_accepted=False,
        best_candidate_attempt=best_attempt,
        last_kernel_rank=result.rank,
        newly_inactive=metrics[0].get("newly_inactive"),
        newly_active=metrics[0].get("newly_active"),
    )
    for key in ("setup_seconds", "solve_seconds", "return_and_verify_seconds"):
        if key in combined:
            combined[key] = sum(m.get(key, 0.0) for m in metrics)
    for key in ("peak_torch_bytes", "restricted_basis_bytes", "cached_operator_product_bytes"):
        if key in combined:
            combined[key] = max(m.get(key, 0) for m in metrics)
    if "return_and_verify_seconds" in combined:
        combined["return_and_verify_seconds"] = (
            combined["total_seconds"] - combined["setup_seconds"] - combined["solve_seconds"]
        )
    return LinearResult(
        current,
        iterations,
        previous_residual,
        final_status,
        best_result.rank if best_result is not None else 0,
        best_result.coarse_condition if best_result is not None else 1.0,
        best_result.fallback_reason if best_result is not None else None,
    ), combined
