"""Two-sided active-set Gauss--Newton optimization with exact-gradient checks."""

from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .coupled_control import FlowEvaluationError
from .coupled_derivatives import GaussNewtonOperator, StabilizationBranchError
from .validation import integer, positive_real


def box_kkt(state, gradient, lower, upper, scale=1.0):
    """Five box-constrained KKT components; primal feasibility is absolute.

    Nonnegative multipliers are recovered only at their corresponding bounds.
    Stationarity, dual feasibility and complementarity use the stated gradient
    scale. The bound identification tolerance is at floating-point precision.
    """
    state, gradient, lower, upper = np.broadcast_arrays(state, gradient, lower, upper)
    if not np.isfinite([state, gradient, lower, upper]).all() or not np.isfinite(scale):
        return dict.fromkeys(
            (
                "primal_absolute",
                "stationarity",
                "dual_feasibility",
                "lower_complementarity",
                "upper_complementarity",
            ),
            float("inf"),
        )
    scale = max(1.0, float(scale))
    epsilon = 32 * np.finfo(float).eps * np.maximum(1, np.abs(state))
    lo = np.where(state <= lower + epsilon, np.maximum(gradient, 0), 0)
    hi = np.where(state >= upper - epsilon, np.maximum(-gradient, 0), 0)
    return {
        "primal_absolute": float(np.max(np.maximum(lower - state, state - upper), initial=0)),
        "stationarity": float(np.max(np.abs(gradient + hi - lo), initial=0) / scale),
        "dual_feasibility": float(np.max(np.maximum(-lo, -hi), initial=0) / scale),
        "lower_complementarity": float(np.max(np.abs(lo * (state - lower)), initial=0) / scale),
        "upper_complementarity": float(np.max(np.abs(hi * (upper - state)), initial=0) / scale),
    }


@dataclass
class BoxQPResult:
    x: np.ndarray
    status: str
    kkt: dict
    history: list


def box_quadratic(H, gradient, diagonal, lower, upper, solver, *, tolerance=1e-9, max_steps=100):
    """Solve min 0.5*x.T H x + gradient.T x under two-sided finite bounds.

    The gradient determines activation and release at each bound. Inactive
    operators zero-extend into the full space before applying H, retaining all
    rows of the source Jacobian. The supplied diagonal is a positive approximate
    preconditioner and active-set scale, not a replacement for H.
    """
    tolerance = positive_real(tolerance, "Quadratic KKT tolerance")
    max_steps = integer(max_steps, "Active-set iteration cap", 1)
    g = np.asarray(gradient, dtype=float)
    d = np.asarray(diagonal, dtype=float)
    lo, hi = np.broadcast_arrays(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), g)[
        :2
    ]
    if (
        g.ndim != 1
        or H.shape != (len(g),) * 2
        or d.shape != g.shape
        or not np.isfinite([g, d, lo, hi]).all()
        or np.any(d <= 0)
        or np.any(lo >= hi)
    ):
        raise ValueError(
            "Finite compatible quadratic data, positive diagonal and strict bounds are required"
        )
    x = np.clip(np.zeros_like(g), lo, hi)
    history = []
    scale = max(1.0, np.linalg.norm(g, np.inf))
    status = "active_set_cap"
    for step in range(max_steps):
        current_gradient = H @ x + g
        kkt = box_kkt(x, current_gradient, lo, hi, scale)
        if max(kkt.values()) <= tolerance:
            return BoxQPResult(x, "converged", kkt, history)
        projected = x - current_gradient / d
        low_active, high_active = projected <= lo, projected >= hi
        inactive = np.flatnonzero(~(low_active | high_active))
        fixed = np.zeros_like(x)
        fixed[low_active], fixed[high_active] = lo[low_active], hi[high_active]
        row = {
            "step": step,
            "lower_active": int(low_active.sum()),
            "upper_active": int(high_active.sum()),
            "inactive": len(inactive),
        }
        if len(inactive):
            if hasattr(H, "restrict"):
                B = H.restrict(inactive)
            else:

                def action(z):
                    full = np.zeros_like(x)
                    full[inactive] = np.asarray(z).ravel()
                    return (H @ full)[inactive]

                B = LinearOperator((len(inactive),) * 2, matvec=action, rmatvec=action, dtype=float)
            B.diagonal = lambda: d[inactive].copy()
            rhs = -(g + H @ fixed)[inactive]
            result, timing = solver.solve(B, rhs, inactive, initial=x[inactive])
            row.update(
                linear_status=result.status,
                linear_residual=result.residual,
                linear_iterations=result.iterations,
                deployed_rank=result.rank,
                coarse_condition=result.coarse_condition,
                fallback=result.fallback_reason,
                timing=timing,
            )
            if result.status != "converged":
                history.append(row)
                status = "linear_" + result.status
                break
            fixed[inactive] = result.x
        else:
            row.update(linear_status="empty", linear_iterations=0, deployed_rank=0)
        x = fixed
        history.append(row)
    kkt = box_kkt(x, H @ x + g, lo, hi, scale)
    if status == "active_set_cap" and max(kkt.values()) <= tolerance:
        status = "converged"
    return BoxQPResult(x, status, kkt, history)


@dataclass
class CoupledOptimum:
    evaluation: object
    objective: float
    gradient: np.ndarray
    kkt: dict
    status: str
    history: list
    seconds: float


def minimize_coupled(
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
    initial_damping=0.0,
    max_regularizations=8,
    max_backtracks=24,
    backtracking="halving",
    secant_memory=0,
    callback=None,
):
    """Damped Gauss--Newton with feasible Armijo steps and exact nonlinear KKT.

    Failed flow evaluations cause backtracking and remain in the history.
    All methods share globalization, QP tolerances and the original-residual
    inner check supplied by StudySolver. No global optimality claim is implied.
    """
    start = time.perf_counter()
    tolerance = positive_real(tolerance, "Nonlinear KKT tolerance")
    max_iterations = integer(max_iterations, "Nonlinear iteration cap", 1)
    max_regularizations = integer(max_regularizations, "Regularization attempts", 1)
    max_backtracks = integer(max_backtracks, "Backtracking cap", 1)
    secant_memory = integer(secant_memory, "Secant memory", 0)
    if backtracking not in {"halving", "quadratic"}:
        raise ValueError("Choose halving or safeguarded quadratic backtracking")
    if not np.isfinite(initial_damping) or initial_damping < 0:
        raise ValueError("Initial damping must be finite and nonnegative")
    lower = np.broadcast_to(lower, (problem.size,)).astype(float, copy=True)
    upper = np.broadcast_to(upper, (problem.size,)).astype(float, copy=True)
    if not np.isfinite([lower, upper]).all() or np.any(lower >= upper):
        raise ValueError("Temperature bounds must be finite and strictly ordered")
    y = np.zeros(problem.size) if initial is None else np.asarray(initial, dtype=float).copy()
    if y.shape != (problem.size,) or not np.isfinite(y).all():
        raise ValueError("Initial temperature must match the complete trajectory")
    y = np.clip(y, lower, upper)
    evaluation = problem.evaluate(y, initial=initial_evaluation)
    objective, gradient = problem.objective_gradient(evaluation, desired)
    history, status = [], "nonlinear_iteration_cap"
    damping = float(initial_damping)
    secants = []
    for iteration in range(max_iterations + 1):
        # Normalize each weighted optimality equation by its positive tracking weight.
        scaled_gradient = gradient / problem.weights
        scale = max(
            1.0,
            np.max(np.abs(evaluation.state - desired)),
            np.max(np.abs(scaled_gradient - (evaluation.state - desired))),
        )
        kkt = box_kkt(evaluation.state, scaled_gradient, lower, upper, scale)
        if max(kkt.values()) <= tolerance:
            status = "converged"
            break
        if iteration == max_iterations:
            break
        row = {
            "iteration": iteration,
            "objective": objective,
            "kkt": kkt,
            "evaluation_seconds": evaluation.seconds,
            "attempts": [],
        }
        moved = False
        for regularization in range(max_regularizations):
            H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha, damping)
            if secants:
                from .coupled_secant import SecantGaussNewton

                H = SecantGaussNewton(H, secants)
            qp = box_quadratic(
                H,
                gradient,
                problem.preconditioning_diagonal(evaluation, damping),
                lower - evaluation.state,
                upper - evaluation.state,
                solver,
                tolerance=qp_tolerance,
                max_steps=qp_cap,
            )
            attempt = {
                "damping": damping,
                "qp_status": qp.status,
                "qp_kkt": qp.kkt,
                "qp_history": qp.history,
                "trials": [],
                "secants": getattr(H, "secant_diagnostics", []),
            }
            row["attempts"].append(attempt)
            slope = float(gradient @ qp.x)
            attempt["directional_derivative"] = slope
            if qp.status == "converged" and np.isfinite(slope) and slope < 0:
                length = 1.0
                for _ in range(max_backtracks):
                    candidate = np.clip(evaluation.state + length * qp.x, lower, upper)
                    try:
                        trial = problem.evaluate(candidate, initial=evaluation)
                        value, derivative = problem.objective_gradient(trial, desired)
                    except FlowEvaluationError as failure:
                        attempt["trials"].append(
                            {
                                "step": length,
                                "status": "flow_" + failure.result.status,
                                "slab": failure.slab,
                                "metrics": failure.metrics,
                                "flow_history": failure.result.history,
                            }
                        )
                        length *= 0.5
                        continue
                    except StabilizationBranchError:
                        attempt["trials"].append(
                            {"step": length, "status": "stabilization_branch_switch"}
                        )
                        length *= 0.5
                        continue
                    finite = np.isfinite(value) and np.isfinite(derivative).all()
                    accepted = finite and value <= objective + 1e-4 * length * slope
                    attempt["trials"].append(
                        {
                            "step": length,
                            "objective": value,
                            "status": "nonfinite_trial"
                            if not finite
                            else "decrease"
                            if accepted
                            else "insufficient_decrease",
                            "evaluation_seconds": trial.seconds,
                            "directional_derivative": float(derivative @ qp.x),
                        }
                    )
                    if accepted:
                        if secant_memory:
                            secants.append((trial.state - evaluation.state, derivative - gradient))
                            secants = secants[-secant_memory:]
                        evaluation, objective, gradient = trial, value, derivative
                        moved = True
                        damping *= 0.25
                        break
                    if backtracking == "quadratic" and np.isfinite(value):
                        # Minimize the quadratic matching f(0), f'(0), and f(length).
                        denominator = 2 * (value - objective - length * slope)
                        proposal = (
                            -slope * length**2 / denominator if denominator > 0 else 0.5 * length
                        )
                        length = float(np.clip(proposal, 0.1 * length, 0.5 * length))
                    else:
                        length *= 0.5
            if moved:
                break
            damping = max(1e-4, 10 * damping)
        history.append(row)
        if callback is not None:
            callback(row, evaluation)
        if not moved:
            status = "globalization_failed"
            break
    # The returned residuals and gradient are always evaluated at the retained state.
    scaled_gradient = gradient / problem.weights
    scale = max(
        1.0,
        np.max(np.abs(evaluation.state - desired)),
        np.max(np.abs(scaled_gradient - (evaluation.state - desired))),
    )
    kkt = box_kkt(evaluation.state, scaled_gradient, lower, upper, scale)
    return CoupledOptimum(
        evaluation, objective, gradient, kkt, status, history, time.perf_counter() - start
    )
