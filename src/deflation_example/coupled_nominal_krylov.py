"""A fixed reference constructed from the initial coupled trajectory.

The construction uses the same temperature supplied to every optimizer. It
reevaluates the complete flow and control Jacobian and imports no optimization
history. Construction and its extra flow evaluation belong to sequence cost.
"""

import time

import numpy as np

from .coupled_bounds import temperature_bounds
from .coupled_derivatives import GaussNewtonOperator
from .coupled_frozen_preconditioner import frozen_preconditioner_factory
from .coupled_krylov_reference import krylov_reference
from .coupled_optimize import equation_acceptance
from .validation import integer


def configured_krylov_reference(
    problem, cfg, initial_guess=None, initial_state=None, *, verification_callback=None
):
    """Construct a full-domain energy-metric reference once, before optimization."""
    transfer = cfg.get("reference_transfer", "full")
    if transfer not in {"full", "sequential"} or not cfg["queries"]:
        raise ValueError("Krylov construction requires a declared transfer and targets")
    if len({q["upper_K"] for q in cfg["queries"]}) != 1:
        raise ValueError("The reference sequence holds the physical bound fixed")
    if initial_guess is not None and initial_state is not None:
        raise ValueError("Supply only one initial temperature")
    if cfg["device"] != "cpu" or cfg.get("inner_preconditioner") != "frozen":
        raise ValueError("This construction uses the CPU velocity-frozen comparison")
    if cfg.get("frozen_sweeps", 3) != 3:
        raise ValueError("The declared construction uses three fixed preconditioner sweeps")
    rank = integer(cfg["rank"], "Reference rank", 1)
    steps = integer(cfg.get("reference_krylov_steps", 48), "Krylov steps", 1)
    if rank > steps:
        raise ValueError("The requested rank must not exceed the Krylov step count")
    start = time.perf_counter()
    bounds = temperature_bounds(cfg, upper_K=cfg["queries"][0]["upper_K"])
    lower = (
        bounds["optimization_lower_K"] - problem.temperature_offset
    ) / problem.temperature_scale
    upper = (
        bounds["optimization_upper_K"] - problem.temperature_offset
    ) / problem.temperature_scale
    state = np.asarray(
        initial_guess.state
        if initial_guess is not None
        else initial_state
        if initial_state is not None
        else np.clip(np.zeros(problem.size), lower, upper),
        dtype=float,
    )
    if (
        state.shape != (problem.size,)
        or not np.isfinite(state).all()
        or np.any(state < lower)
        or np.any(state > upper)
    ):
        raise ValueError("The nominal temperature must be finite, feasible and full-domain")
    evaluation = problem.evaluate(state.copy(), initial=initial_guess)
    diagnostics = problem.verify(evaluation)
    if verification_callback is not None:
        verification_callback(diagnostics)
    if not equation_acceptance(diagnostics, cfg):
        raise ValueError("The nominal coupled equations fail independent verification")
    operator = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
    diagonal = problem.preconditioning_diagonal(evaluation, 0.0)
    evaluation_seconds = time.perf_counter() - start
    tick = time.perf_counter()
    inverse = frozen_preconditioner_factory(problem, evaluation, sweeps=3)(np.arange(problem.size))
    inverse_seconds = time.perf_counter() - tick
    reference = krylov_reference(
        operator,
        inverse,
        diagonal,
        rank,
        steps=steps,
        seed=cfg.get("reference_krylov_seed", 20260923),
        selection=cfg.get("reference_krylov_selection", "alternating_low_high"),
    )
    reference.description.update(
        nominal_evaluation_seconds=evaluation_seconds,
        nominal_preconditioner_seconds=inverse_seconds,
        total_reference_construction_seconds=time.perf_counter() - start,
        nominal_equations=diagnostics,
        nominal_alpha=problem.alpha,
        nominal_policy="Initial temperature; full coupled Gauss-Newton; zero damping and no secants",
        lifetime="Fixed full-domain space across every target and nonlinear/PDAS update",
    )
    if cfg.get("physics") == "prescribed_flow":
        reference.description.update(
            nominal_policy="Exact fixed-flow space-time quadratic operator; zero damping and no secants",
            lifetime="Fixed full-domain space across every target and PDAS update",
        )
    if transfer == "sequential":
        from .coupled_reference import SequentialReference

        sequential = SequentialReference(reference, problem.size)
        sequential.description["lifetime"] = (
            "Previous restricted space, transferred by zero extension at every inactive solve; no learned or replacement directions"
        )
        return sequential
    return reference
