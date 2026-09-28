"""Gradient projection and reduced CG for a fixed convex box quadratic.

Separate diagnostic procedure following the projected searches described by
Benson, McInnes and More, ANL/MCS-P768-0799, Section 3. The reduced CG solve uses
the package's independent residual acceptance rather than that paper's
objective-decrease stopping rule. Existing PDAS and nonlinear runners are unchanged.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .coupled_optimizer import BoxQPResult, box_kkt
from .solvers import independent_residual
from .validation import integer, positive_real


def projected_search(H, x, gradient, direction, lower, upper, step=1.0, cap=40):
    """Use a quadratic difference, avoiding subtraction of large objective values."""
    step = positive_real(step, "Initial projected-search step")
    for halving in range(integer(cap, "Projected-search cap", 1)):
        candidate = np.clip(x + step * direction, lower, upper)
        change = candidate - x
        slope = float(gradient @ change)
        if np.any(change) and np.isfinite(slope) and slope < 0:
            product = H @ change
            decrease = slope + 0.5 * float(change @ product)
            if np.isfinite(decrease) and decrease <= 1e-4 * slope:
                return candidate, {
                    "step_length": step,
                    "halvings": halving,
                    "quadratic_change": decrease,
                }
        step *= 0.5
    return None, {"status": "projected_search_failed"}


def box_projected_cg(
    H,
    gradient,
    diagonal,
    lower,
    upper,
    solver,
    *,
    initial=None,
    tolerance=1e-9,
    max_steps=100,
    projected_steps=8,
    preconditioner_factory=None,
    kkt_evaluator=None,
    checkpoint=None,
):
    """Maintain feasible iterates and independently verify every returned state.

    Gradient projection identifies a face; CG computes a correction within it.
    Both updates use a projected Armijo search. The supplied positive diagonal
    defines the gradient-projection metric. All unsuccessful solves and searches
    retain an explicit status. This diagnostic has no nonlinear acceptance role.
    """
    tolerance = positive_real(tolerance, "Quadratic KKT tolerance")
    max_steps = integer(max_steps, "Outer iteration cap", 1)
    projected_steps = integer(projected_steps, "Gradient-projection cap", 1)
    g, d = np.asarray(gradient, dtype=float), np.asarray(diagonal, dtype=float)
    lo, hi = np.broadcast_arrays(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), g)[
        :2
    ]
    if (
        g.ndim != 1
        or H.shape != (len(g), len(g))
        or d.shape != g.shape
        or not np.isfinite([g, d, lo, hi]).all()
        or np.any(d <= 0)
        or np.any(lo >= hi)
    ):
        raise ValueError(
            "Finite compatible quadratic data, positive diagonal and strict bounds required"
        )
    x = np.zeros_like(g) if initial is None else np.asarray(initial, dtype=float).copy()
    if x.shape != g.shape or not np.isfinite(x).all():
        raise ValueError("Initial state must be finite and dimensionally matched")
    x = np.clip(x, lo, hi)
    scale = max(1.0, np.linalg.norm(g, np.inf))
    assess = kkt_evaluator or (lambda state, derivative: box_kkt(state, derivative, lo, hi, scale))
    history, cached_indices, cached_preconditioner = [], None, None

    def finish(status):
        kkt = assess(x, H @ x + g)
        if max(kkt.values()) <= tolerance:
            status = "converged"
        return BoxQPResult(x.copy(), status, kkt, history)

    for iteration in range(max_steps):
        if solver.stop_requested is not None and solver.stop_requested():
            return finish("budget_exhausted")
        derivative = H @ x + g
        if max(assess(x, derivative).values()) <= tolerance:
            return finish("converged")
        row = {"step": iteration, "projected_steps": []}
        history.append(row)
        best_decrease = 0.0
        for _ in range(projected_steps):
            active = (x == lo) | (x == hi)
            binding = ((x == lo) & (derivative >= 0)) | ((x == hi) & (derivative <= 0))
            direction = np.where(binding, 0.0, -derivative / d)
            curvature = float(direction @ (H @ direction))
            slope = float(derivative @ direction)
            if not np.isfinite([curvature, slope]).all() or curvature <= 0 or slope >= 0:
                return finish("projected_curvature_or_descent_failed")
            candidate, event = projected_search(
                H, x, derivative, direction, lo, hi, -slope / curvature
            )
            row["projected_steps"].append(event)
            if candidate is None:
                return finish("projected_search_failed")
            x = candidate
            derivative = H @ x + g
            if max(assess(x, derivative).values()) <= tolerance:
                return finish("converged")
            decrease = -event["quadratic_change"]
            unchanged = np.array_equal(active, (x == lo) | (x == hi))
            if unchanged or decrease <= 0.1 * best_decrease:
                break
            best_decrease = max(best_decrease, decrease)
        indices = np.flatnonzero((x > lo) & (x < hi))
        row["inactive"] = len(indices)
        if not len(indices):
            continue

        def action(z):
            full = np.zeros_like(g)
            full[indices] = np.asarray(z).ravel()
            return (H @ full)[indices]

        B = LinearOperator((len(indices),) * 2, matvec=action, rmatvec=action, dtype=float)
        B.diagonal = lambda: d[indices].copy()
        if preconditioner_factory is not None:
            if cached_indices is None or not np.array_equal(cached_indices, indices):
                cached_preconditioner = preconditioner_factory(indices)
                cached_indices = indices.copy()
            B.preconditioner = cached_preconditioner
        rhs = -derivative[indices]
        result, timing = solver.solve(B, rhs, indices, initial=np.zeros_like(rhs))
        residual = independent_residual(B, result.x, rhs)
        row.update(
            linear_status=result.status,
            linear_iterations=result.iterations,
            linear_residual=residual,
            timing=timing,
        )
        if result.status != "converged":
            return finish("linear_" + result.status)
        if not np.isfinite(residual) or residual > solver.rtol:
            return finish("linear_original_residual_failed")
        direction = np.zeros_like(g)
        direction[indices] = result.x
        candidate, event = projected_search(H, x, derivative, direction, lo, hi)
        row["reduced_search"] = event
        if candidate is None:
            return finish("reduced_search_failed")
        x = candidate
        row["kkt"] = assess(x, H @ x + g)
        if checkpoint is not None:
            checkpoint({"x": x.copy(), "history": list(history), "kkt": row["kkt"]})
    return finish("iteration_cap")
