"""Bounded residual correction with an unchanged original-system acceptance test."""

import time

import numpy as np

from .solvers import LinearResult, independent_residual
from .timing import PHASES


def verified_refinement(solve, A, b, initial, tolerance, maxiter, max_corrections=4):
    """Solve error equations after a rejected candidate, within one iteration cap.

    ``solve`` receives (rhs, initial, relative_target, remaining_iterations).
    Each error equation requests a factor-ten residual reduction. Only the
    independently recomputed residual of the original system accepts the result.
    Nested kernel timers, restriction costs, and additional CPU verification
    partition the returned complete callback time.
    """
    start = time.perf_counter()
    current = np.zeros_like(b) if initial is None else np.array(initial, copy=True)
    rhs, guess, local_tolerance = b, initial, tolerance
    iterations, verification_seconds = 0, 0.0
    records, metrics = [], []
    previous_residual = float("inf")
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
            }
        )
        improved = np.isfinite(residual) and residual < previous_residual
        if improved:
            current, previous_residual = candidate, residual
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
        refinement_policy="candidate-original-residual-error-equations-v1",
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
        result.rank,
        result.coarse_condition,
        result.fallback_reason,
    ), combined
