"""Optional float64 CUDA linear kernels. Importing this module does not load Torch."""

import time
import numpy as np
from scipy import sparse
from scipy.linalg import norm
from .solvers import LinearResult, independent_residual, orthonormalize, validate_linear_inputs
from .validation import matrix


def require_cuda():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "CUDA requested: install a CUDA-enabled PyTorch build and the [gpu] extra"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but no CUDA device is available; use device=cpu or check PyTorch/driver installation"
        )
    return torch


def gpu_deflated_cg(
    A,
    b,
    basis=None,
    diagonal=None,
    x0=None,
    rtol=1e-10,
    maxiter=20000,
    refresh=1000,
    condition_limit=1e10,
):
    """Return (LinearResult, timing/memory metrics), verified with the CPU matrix.

    The GPU context should be warmed by the caller. Setup includes CPU SVD,
    transfer and coarse construction. Total includes transfer back and the fresh
    CPU residual. It excludes problem assembly, reference construction and PDAS.
    """
    torch = require_cuda()
    A = sparse.csr_matrix(matrix(A), dtype=float)
    b, initial, d = validate_linear_inputs(
        A, b, basis, diagonal, x0, rtol, maxiter, refresh, condition_limit
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    baseline = torch.cuda.memory_allocated()
    H = torch.sparse_csr_tensor(
        torch.tensor(A.indptr, device="cuda"),
        torch.tensor(A.indices, device="cuda"),
        torch.tensor(A.data, device="cuda"),
        size=A.shape,
        dtype=torch.float64,
    )
    rhs = torch.tensor(b, device="cuda", dtype=torch.float64)
    diag = torch.tensor(d, device="cuda", dtype=torch.float64)
    x = torch.tensor(initial, device="cuda", dtype=torch.float64)
    Z = np.empty((b.size, 0)) if basis is None else orthonormalize(basis)
    rank = Z.shape[1]
    V = torch.tensor(Z, device="cuda", dtype=torch.float64) if rank else None
    condition, fallback, coarse_failed = 1.0, None, False

    def apply(v):
        return torch.mv(H, v)

    if rank:
        E = V.T @ torch.mm(H, V)
        E = (E + E.T) / 2
        condition = float(torch.linalg.cond(E)) if bool(torch.isfinite(E).all()) else float("inf")
        if not np.isfinite(condition) or condition > condition_limit:
            rank, V, fallback = 0, None, "coarse_condition_limit"
        else:
            chol, info = torch.linalg.cholesky_ex(E)
            coarse_failed = bool(info)

    def Q(v):
        return (
            V @ torch.cholesky_solve((V.T @ v)[:, None], chol)[:, 0]
            if rank
            else torch.zeros_like(v)
        )

    def precondition(r):
        z = r / diag
        return z - Q(apply(z)) if rank else z

    rhs_scale = float(np.max(np.abs(b))) or 1.0
    rhs_norm = float(norm(b / rhs_scale)) or 1.0

    def small_residual(residual):
        value = float(torch.linalg.vector_norm(residual / rhs_scale)) / rhs_norm
        return np.isfinite(value) and value <= rtol

    status, iterations = "breakdown" if coarse_failed else "maxiter", 0
    if not coarse_failed:
        x += Q(rhs - apply(x))
    r = rhs - apply(x)
    torch.cuda.synchronize()
    setup = time.perf_counter() - start
    solve_start = time.perf_counter()
    if not coarse_failed:
        if small_residual(r):
            status = "converged"
        else:
            z = precondition(r)
            p, rz = z.clone(), torch.dot(r, z)
            for k in range(maxiter):
                Ap = apply(p)
                curvature = torch.dot(p, Ap)
                if (
                    float(curvature) <= 0
                    or float(rz) <= 0
                    or not bool(torch.isfinite(curvature + rz))
                ):
                    status = "breakdown"
                    break
                step = rz / curvature
                x += step * p
                r -= step * Ap
                iterations = k + 1
                restart = iterations % refresh == 0 or small_residual(r)
                if restart:
                    r = rhs - apply(x)
                    if small_residual(r):
                        status = "converged"
                        break
                    x += Q(r)
                    r = rhs - apply(x)
                    if small_residual(r):
                        status = "converged"
                        break
                z = precondition(r)
                new_rz = torch.dot(r, z)
                p = z.clone() if restart else z + (new_rz / rz) * p
                rz = new_rz
    torch.cuda.synchronize()
    solve = time.perf_counter() - solve_start
    x_cpu = x.cpu().numpy()
    residual = independent_residual(A, x_cpu, b)
    if residual <= rtol and not coarse_failed:
        status = "converged"
    elif status == "converged":
        status = "residual_failed"
    total = time.perf_counter() - start
    return LinearResult(x_cpu, iterations, residual, status, rank, condition, fallback), {
        "setup_seconds": setup,
        "solve_seconds": solve,
        "return_and_verify_seconds": total - setup - solve,
        "total_seconds": total,
        "peak_torch_bytes": torch.cuda.max_memory_allocated(),
        "baseline_torch_bytes": baseline,
    }
