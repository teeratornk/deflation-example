"""Auxiliary step-size continuation ending at the unchanged physical equation."""

from copy import copy
import time

from .coupled_forward import CoupledResult
from .coupled_newton_replay import newton_step, step_equations, criteria_met, equation_merit
from .coupled_resolution import forward_model
from .validation import integer, positive_real


def continued_step(
    problem,
    source,
    previous_state,
    previous_flow,
    slab,
    *,
    initial_state=None,
    initial_flow=None,
    tolerance=1e-12,
    max_iterations=100,
    stage_cap=64,
    minimum_increment=1 / 1024,
    callback=None,
):
    """Vary only an auxiliary storage coefficient, holding physical history fixed.

    Auxiliary roots supply initial guesses for the original step. They are not
    additional physical time levels. Every return contains a state, flow and
    residual evaluated with the original time step and unchanged source.
    """
    start = time.perf_counter()
    stage_cap = integer(stage_cap, "Continuation cap", 1)
    max_iterations = integer(max_iterations, "Newton cap", 0)
    minimum_increment = positive_real(minimum_increment, "Minimum continuation increment")
    if minimum_increment > 1 / 16:
        raise ValueError("The minimum increment must not exceed the first continuation stage")
    model = forward_model(problem)
    initial = newton_step(
        problem,
        source,
        previous_state,
        previous_flow,
        slab,
        initial_state=initial_state,
        initial_flow=initial_flow,
        tolerance=tolerance,
        max_iterations=0,
    )
    state, flow = initial.state, initial.flow
    _, metrics = step_equations(
        problem, model, state, flow, source, previous_state, previous_flow, slab
    )
    best_state, best_flow, best_metrics = state.copy(), flow, metrics
    fraction, increment, histories = 0.0, 1 / 16, []
    reason = "continuation_stage_cap"
    for attempt in range(stage_cap):
        if criteria_met(best_metrics, tolerance):
            reason = "converged"
            break
        next_fraction = min(1.0, fraction + increment)
        auxiliary = copy(problem)
        auxiliary.physical_steps = problem.physical_steps.copy()
        auxiliary.physical_steps[slab] *= next_fraction
        auxiliary.steps = auxiliary.physical_steps / problem.time_scale
        result = newton_step(
            auxiliary,
            source,
            previous_state,
            previous_flow,
            slab,
            initial_state=state,
            initial_flow=flow,
            tolerance=tolerance,
            max_iterations=max_iterations,
            line_search="fixed_scaled",
            backtrack_cap=40,
        )
        _, original_checks = step_equations(
            problem, model, result.state, result.flow, source, previous_state, previous_flow, slab
        )
        if equation_merit(original_checks) < equation_merit(best_metrics):
            best_state, best_flow, best_metrics = result.state.copy(), result.flow, original_checks
        row = {
            "attempt": attempt,
            "auxiliary_step_fraction": next_fraction,
            "auxiliary_status": result.status,
            "seconds": result.seconds,
            "original_equations": original_checks,
            "history": result.history,
        }
        histories.append(row)
        if callback is not None:
            callback(histories.copy())
        if criteria_met(original_checks, tolerance):
            best_state, best_flow, best_metrics = result.state.copy(), result.flow, original_checks
            reason = "converged"
            break
        if result.status == "converged":
            fraction, state, flow = next_fraction, result.state, result.flow
            increment = min(2 * increment, 0.25)
        else:
            increment /= 2
            if increment < minimum_increment:
                reason = "continuation_minimum_increment"
                break
    # Never return an auxiliary residual as an original-system check.
    _, best_metrics = step_equations(
        problem, model, best_state, best_flow, source, previous_state, previous_flow, slab
    )
    status = "converged" if criteria_met(best_metrics, tolerance) else reason
    if status == "converged" and not criteria_met(best_metrics, tolerance):
        status = "continuation_verification_failed"
    return CoupledResult(
        best_state,
        best_flow,
        status,
        [
            {"procedure": "auxiliary_step_continuation", "stages": histories},
            {"procedure": "returned_state_verification", **best_metrics},
        ],
        time.perf_counter() - start,
    )
