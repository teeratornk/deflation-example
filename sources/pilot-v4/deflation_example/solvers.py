"""PDAS and exact-coarse deflation for small, auditable control examples.

Adapted from the verified manuscript implementation (see PROVENANCE.md).
All calculations are float64; an iteration cap is a failure, never a
convergence indication.
"""

from dataclasses import dataclass
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, spsolve
from .validation import integer, matrix, positive_real, real_array
from .timing import PhaseTimer


@dataclass
class LinearResult:
    x: np.ndarray
    iterations: int
    residual: float
    status: str
    rank: int = 0
    coarse_condition: float = 1.0
    fallback_reason: str | None = None


def relative_norm(residual, rhs):
    """Scale before taking norms; zero loads use an absolute norm.

    Nonfinite arithmetic is a failed check, never a zero residual. Scaling
    avoids squaring tiny/large loads in the convergence test.
    """
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(rhs)):
        return float("inf")
    scale = float(np.max(np.abs(rhs), initial=0))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value = (
            linalg.norm(residual / scale, check_finite=False)
            / linalg.norm(rhs / scale, check_finite=False)
            if scale
            else linalg.norm(residual, check_finite=False)
        )
    return float(value) if np.isfinite(value) else float("inf")


def independent_residual(A, x, b):
    with np.errstate(over="ignore", invalid="ignore"):
        return relative_norm(np.asarray(b) - A @ x, np.asarray(b))


def orthonormalize(Z, tolerance=1e-12):
    """Rank revealing SVD, including empty and dependent input spaces."""
    Z = real_array(Z, "Basis")
    if Z.ndim != 2 or not np.all(np.isfinite(Z)):
        raise ValueError("Basis must be a finite two-dimensional array")
    positive_real(tolerance, "Rank tolerance")
    if 0 in Z.shape:
        return Z[:, :0].copy()
    U, s, _ = linalg.svd(Z, full_matrices=False, check_finite=True)
    return U[:, s > tolerance * s[0]] if s[0] else U[:, :0]


def deflated_cg(
    A,
    b,
    basis=None,
    diagonal=None,
    x0=None,
    rtol=1e-10,
    maxiter=20000,
    refresh=1000,
    condition_limit=1e10,
    direction_callback=None,
    cache_operator_product=False,
):
    """Coarse-corrected, projected-direction CG (exact-coarse A-DEF2).

    A is symmetric positive definite. The original residual is recomputed before
    accepting a solve. Residual replacement also coarse-corrects and restarts CG.
    Rank loss and an ill-conditioned coarse matrix lead to explicit truncation or
    ordinary Jacobi-CG, not a pseudoinverse of a singular coarse matrix.
    """
    A = matrix(A)
    if direction_callback is not None and not callable(direction_callback):
        raise ValueError("Direction callback must be callable")
    if not isinstance(cache_operator_product, bool):
        raise ValueError("Operator-product caching must be Boolean")
    b, x, d = validate_linear_inputs(
        A, b, basis, diagonal, x0, rtol, maxiter, refresh, condition_limit
    )
    n = len(b)
    Z = np.empty((n, 0)) if basis is None else orthonormalize(basis)
    condition = 1.0
    fallback = None
    factor = None
    if Z.shape[1]:
        AZ = A @ Z
        E = Z.T @ AZ
        E = (E + E.T) / 2
        condition = float(np.linalg.cond(E)) if np.all(np.isfinite(E)) else float("inf")
        if np.isfinite(condition) and condition <= condition_limit:
            try:
                factor = linalg.cho_factor(E, lower=True)
            except linalg.LinAlgError:
                return LinearResult(
                    x, 0, independent_residual(A, x, b), "breakdown", Z.shape[1], condition
                )
        else:
            Z = Z[:, :0]
            fallback = "coarse_condition_limit"
    rank = Z.shape[1]

    def Q(v):
        return Z @ linalg.cho_solve(factor, Z.T @ v) if rank else np.zeros_like(v)

    def precondition(r):
        z = r / d
        if rank and cache_operator_product:
            return z - Z @ linalg.cho_solve(factor, AZ.T @ z)
        return z - Q(A @ z) if rank else z

    x += Q(b - A @ x)
    r = b - A @ x
    if relative_norm(r, b) <= rtol:
        return LinearResult(
            x, 0, independent_residual(A, x, b), "converged", rank, condition, fallback
        )
    z = precondition(r)
    p, rz = z.copy(), float(r @ z)
    status, iterations = "maxiter", 0
    for k in range(maxiter):
        if direction_callback is not None:
            direction_callback(p.copy())
        Ap = A @ p
        curvature = float(p @ Ap)
        if curvature <= 0 or rz <= 0 or not np.isfinite(curvature + rz):
            status = "breakdown"
            break
        step = rz / curvature
        x += step * p
        r -= step * Ap
        iterations = k + 1
        restart = iterations % refresh == 0 or relative_norm(r, b) <= rtol
        if restart:
            r = b - A @ x
            if relative_norm(r, b) <= rtol:
                status = "converged"
                break
            x += Q(r)
            r = b - A @ x
            if relative_norm(r, b) <= rtol:
                status = "converged"
                break
        z = precondition(r)
        rz_new = float(r @ z)
        p = z.copy() if restart else z + (rz_new / rz) * p
        rz = rz_new
    residual = independent_residual(A, x, b)
    if residual <= rtol:
        status = "converged"
    elif status == "converged":
        status = "residual_failed"
    return LinearResult(x, iterations, residual, status, rank, condition, fallback)


def validate_linear_inputs(A, b, basis, diagonal, x0, rtol, maxiter, refresh, condition_limit):
    """Shared CPU/GPU input contract; SPD remains a caller requirement."""
    b = real_array(b, "Right-hand side")
    if b.ndim != 1 or b.size == 0 or not np.all(np.isfinite(b)) or A.shape != (b.size, b.size):
        raise ValueError(
            "Matrix and finite one-dimensional right-hand side must have compatible shapes"
        )
    positive_real(rtol, "Relative tolerance")
    integer(maxiter, "Iteration cap")
    integer(refresh, "Residual refresh interval", 1)
    if positive_real(condition_limit, "Coarse condition limit") < 1:
        raise ValueError("Coarse condition limit must be at least one")
    x = np.zeros(b.size) if x0 is None else real_array(x0, "Initial state").copy()
    d = np.ones(b.size) if diagonal is None else real_array(diagonal, "Jacobi diagonal")
    if x.shape != b.shape or not np.all(np.isfinite(x)):
        raise ValueError(
            "Initial state must be finite and have the same shape as the right-hand side"
        )
    if d.shape != b.shape or np.any(d <= 0) or not np.all(np.isfinite(d)):
        raise ValueError("Jacobi diagonal must be positive, finite and have the correct shape")
    if basis is not None:
        Z = real_array(basis, "Basis")
        if Z.ndim != 2 or Z.shape[0] != b.size or not np.all(np.isfinite(Z)):
            raise ValueError("Basis must be finite with one row per unknown")
    return b, x, d


def kkt_metrics(H, f, bound, y, multiplier):
    """Upper bound convention: H y - f + multiplier = 0."""
    g = H @ y - f
    scale = max(1.0, np.linalg.norm(f, np.inf))
    slack = np.asarray(bound) - y
    return {
        "stationarity": float(np.linalg.norm(g + multiplier, np.inf) / scale),
        "primal": float(np.maximum(-slack, 0).max(initial=0)),
        "dual": float(np.maximum(-multiplier, 0).max(initial=0) / scale),
        "complementarity": float(np.max(np.abs(multiplier * slack), initial=0) / scale),
        "projected_gradient": float(np.linalg.norm(y - np.minimum(bound, y - g), np.inf) / scale),
    }


def pdas(H, f, bound, initial_active=None, tolerance=1e-9, maxiter=100, linear_solver=None):
    """Primal-dual active-set solve of min .5 y'H y - f'y, y <= bound.

    The positive scaling c_i=H_ii is fixed during the solve. Cycles and
    unsuccessful inner solves are reported rather than silently accepted.
    """
    timer = PhaseTimer(
        phases=(
            "preparation",
            "restriction",
            "inner_solve",
            "inner_verification",
            "kkt_and_update",
            "host_bookkeeping",
        )
    )
    H = matrix(H)
    matrix_free = isinstance(H, LinearOperator)
    if matrix_free:
        if not callable(getattr(H, "restrict", None)) or not callable(getattr(H, "diagonal", None)):
            raise ValueError("Matrix-free PDAS needs restriction and diagonal operations")
        if linear_solver is None:
            raise ValueError("Matrix-free PDAS requires an explicit inner solver")
    else:
        H = sparse.csr_matrix(H, dtype=float)
    f = real_array(f, "PDAS load")
    if f.ndim != 1 or f.size == 0 or not np.all(np.isfinite(f)) or H.shape != (f.size, f.size):
        raise ValueError("PDAS needs a nonempty finite load and a compatible matrix")
    if (not matrix_free and not np.all(np.isfinite(H.data))) or (
        not np.all(np.isfinite(H.diagonal())) or np.any(H.diagonal() <= 0)
    ):
        raise ValueError("PDAS requires a finite SPD matrix")
    positive_real(tolerance, "PDAS tolerance")
    integer(maxiter, "PDAS iteration cap", 1)
    bound = np.broadcast_to(real_array(bound, "Bounds"), f.shape).copy()
    if not np.all(np.isfinite(bound)):
        raise ValueError("Bounds must be finite")
    raw_mask = (
        np.zeros(len(f), dtype=bool)
        if initial_active is None
        else real_array(initial_active, "Initial mask")
    )
    if not np.all(np.isin(raw_mask, [0, 1])):
        raise ValueError("Initial mask must contain only Boolean or zero/one values")
    active = np.array(raw_mask, dtype=bool)
    if active.shape != f.shape:
        raise ValueError("Initial mask must have one entry per unknown")
    solved_active = active.copy()
    seen, history = set(), []
    y, multiplier = bound.copy(), np.zeros(len(f))
    status = "maxiter"
    timer.mark("preparation")
    for iteration in range(maxiter):
        timer.mark("host_bookkeeping")
        key = active.tobytes()
        if key in seen:
            status = "cycle"
            break
        seen.add(key)
        solved_active = active.copy()
        I, J = np.flatnonzero(~active), np.flatnonzero(active)
        y[J] = bound[J]
        inner_residual, inner_iterations, inner_status = 0.0, 0, "converged"
        if len(I):
            if matrix_free:
                HII = H.restrict(I)
                boundary_state = np.zeros(len(f))
                boundary_state[J] = bound[J]
                rhs = f[I] - (H @ boundary_state)[I]
            else:
                HII = H[I][:, I].tocsr()
                rhs = f[I] - H[I][:, J] @ bound[J]
            timer.mark("restriction")
            if linear_solver is None:
                y[I] = spsolve(HII, rhs)
                timer.mark("inner_solve")
                inner_residual = independent_residual(HII, y[I], rhs)
            else:
                result = linear_solver(HII, rhs, I)
                timer.mark("inner_solve")
                values = real_array(result.x, "Inner solution")
                if values.shape != rhs.shape:
                    raise ValueError("Inner solution must have one entry per inactive unknown")
                y[I] = values
                inner_iterations = integer(result.iterations, "Inner iteration count")
                inner_residual = independent_residual(HII, y[I], rhs)
                inner_status = result.status
            if not np.all(np.isfinite(y[I])) or not np.isfinite(inner_residual):
                inner_status = "nonfinite"
            timer.mark("inner_verification")
        else:
            timer.mark("restriction")
        # Inactive multipliers are imposed as zero, so stationarity is not a
        # tautology that could conceal a failed inactive linear solve.
        multiplier[:] = 0
        multiplier[J] = (f - H @ y)[J]
        metrics = kkt_metrics(H, f, bound, y, multiplier)
        new_active = multiplier + H.diagonal() * (y - bound) > 0
        history.append(
            {
                "active": int(active.sum()),
                "entered": int((new_active & ~active).sum()),
                "left": int((active & ~new_active).sum()),
                "linear_residual": inner_residual,
                "linear_iterations": inner_iterations,
                "linear_status": inner_status,
                **metrics,
            }
        )
        timer.mark("kkt_and_update")
        if inner_status != "converged":
            status = "inner_" + inner_status
            break
        if all(np.isfinite(v) and v <= tolerance for v in metrics.values()):
            status = "converged"
            break
        active = new_active
    output = {
        "y": y,
        "multiplier": multiplier,
        "active": solved_active,
        "status": status,
        "iterations": len(history),
        "history": history,
        "kkt": kkt_metrics(H, f, bound, y, multiplier),
        "objective": float(0.5 * y @ (H @ y) - f @ y),
    }
    output["timing"] = timer.finish()
    return output


def restricted_normal_operator(A, alpha, inactive):
    """R_I (I + alpha A.T A) R_I.T; never substitute A_II.T A_II."""
    A = sparse.csr_matrix(matrix(A), dtype=float)
    positive_real(alpha, "Regularization")
    indices = np.asarray(inactive)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise ValueError("Inactive indices must be a one-dimensional integer array")
    indices = indices.astype(int)
    if (
        np.any(indices < 0)
        or np.any(indices >= A.shape[1])
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("Inactive indices must be distinct and within the column range")
    AI = A[:, indices]

    def apply(v):
        return v + alpha * (AI.T @ (AI @ v))

    return LinearOperator((len(indices), len(indices)), matvec=apply, matmat=apply, dtype=float)


def calibrate_bound(H, target, fraction, steps=12):
    """Calibrate against achieved PDAS activity, not an unconstrained percentile.

    For the positive-target validation cases, bound=0 has full activity and the
    unconstrained maximum has zero activity. Return the best attained fraction;
    discrete/symmetric masks need not realize an arbitrary exact percentage.
    """
    if positive_real(fraction, "Activity target") >= 1:
        raise ValueError(
            "Activity target must lie in (0,1), with a positive number of calibration steps"
        )
    integer(steps, "Calibration steps", 1)
    hi = max(float(np.max(spsolve(H, target))), np.finfo(float).eps)
    lo = 0.0
    best = None
    for _ in range(steps):
        bound = (lo + hi) / 2
        result = pdas(H, target, bound, tolerance=1e-8)
        if result["status"] != "converged":
            raise RuntimeError("Bound calibration failed: " + result["status"])
        actual = float(result["active"].mean())
        if best is None or abs(actual - fraction) < best[0]:
            best = (abs(actual - fraction), bound, actual)
        if actual > fraction:
            lo = bound
        else:
            hi = bound
    return best[1], best[2]
