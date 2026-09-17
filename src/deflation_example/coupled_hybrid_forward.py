"""Fixed-source forward Newton with an explicitly charged Anderson fallback."""

import time

from .coupled_fixed_point import solve_forward
from .coupled_forward import CoupledResult
from .coupled_newton_replay import criteria_met, equation_merit, newton_step, step_equations
from .coupled_resolution import forward_model
from .validation import integer, positive_real


def hybrid_step(
    problem,
    source,
    previous_state,
    previous_flow,
    slab,
    *,
    tolerance=1e-12,
    max_iterations=100,
    initial_state=None,
    initial_flow=None,
    line_search="fixed_scaled",
    backtrack_cap=21,
    trust_region=None,
    anderson_cap=300,
    flow_cap=100,
    attempt_callback=None,
):
    """Verify every returned candidate against the unchanged coupled equations.

    Newton retains its final fields at a branch switch, stagnation, or exhausted
    budget. The fallback starts from those fields and changes only the forward
    solution procedure. Each attempt's time and history are retained, including
    unsuccessful attempts. Source and prior physical fields remain unchanged.
    """
    tolerance = positive_real(tolerance, "Coupled tolerance")
    anderson_cap = integer(anderson_cap, "Anderson iteration cap", 1)
    flow_cap = integer(flow_cap, "Momentum iteration cap", 1)
    start = time.perf_counter()
    model = forward_model(problem)
    first = newton_step(
        problem,
        source,
        previous_state,
        previous_flow,
        slab,
        tolerance=tolerance,
        max_iterations=max_iterations,
        initial_state=initial_state,
        initial_flow=initial_flow,
        line_search=line_search,
        backtrack_cap=backtrack_cap,
        trust_region=trust_region,
    )
    attempts = []

    def verified(candidate, procedure):
        _, metrics = step_equations(
            problem,
            model,
            candidate.state,
            candidate.flow,
            source,
            previous_state,
            previous_flow,
            slab,
        )
        attempts.append(
            {
                "procedure": procedure,
                "status": candidate.status,
                "seconds": candidate.seconds,
                "history": candidate.history,
                "independent_equations": metrics,
            }
        )
        if attempt_callback is not None:
            # Observational persistence; preserve a returned Newton state even
            # if the subsequent fallback is interrupted by a scheduler limit.
            attempt_callback(candidate, attempts[-1])
        return metrics

    first_metrics = verified(
        first, "fixed_scaled_newton" if line_search == "fixed_scaled" else "newton"
    )
    selected, metrics, name = first, first_metrics, "newton"
    if not criteria_met(first_metrics, tolerance):
        second = solve_forward(
            model,
            source,
            first.state,
            first.flow,
            previous_state=previous_state,
            previous_velocity=previous_flow.velocity,
            time_step=float(problem.physical_steps[slab]),
            policy="anderson",
            mixing="coupled",
            depth=5,
            relaxation=0.5,
            tolerance=tolerance,
            max_iterations=anderson_cap,
            flow_cap=flow_cap,
        )
        second_metrics = verified(second, "complete_field_anderson")
        if criteria_met(second_metrics, tolerance) or equation_merit(
            second_metrics
        ) < equation_merit(first_metrics):
            selected, metrics, name = second, second_metrics, "anderson"
    status = "converged" if criteria_met(metrics, tolerance) else "hybrid_no_convergence"
    history = [
        {"procedure": "forward_attempts", "attempts": attempts},
        {"procedure": "returned_state_verification", "selected": name, **metrics},
    ]
    return CoupledResult(
        selected.state, selected.flow, status, history, time.perf_counter() - start
    )
