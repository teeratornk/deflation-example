"""Optional float64 CUDA linear kernels. Importing this module does not load Torch."""

import time
import numpy as np
from scipy import sparse
from scipy.linalg import norm
from .solvers import LinearResult, independent_residual, orthonormalize, validate_linear_inputs
from .validation import matrix, positive_real
from .timing import PhaseTimer


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


def orthonormalize_gpu(basis, torch, tolerance=1e-12):
    """Thin QR and SVD of its small factor, with the CPU singular-value cutoff."""
    if 0 in basis.shape:
        return basis[:, :0].clone()
    Q, R = torch.linalg.qr(basis, mode="reduced")
    U, values, _ = torch.linalg.svd(R, full_matrices=False, driver="gesvd" if R.is_cuda else None)
    keep = values > tolerance * values[0]
    return Q if bool(keep.all()) else Q @ U[:, keep]


def _gpu_deflated_cg(
    A,
    b,
    basis=None,
    diagonal=None,
    x0=None,
    rtol=1e-10,
    maxiter=20000,
    refresh=1000,
    condition_limit=1e10,
    *,
    basis_backend,
    timer,
    torch,
    acceptance_rtol,
):
    """Return (LinearResult, timing/memory metrics), verified with the CPU matrix.

    The caller initializes the GPU context. Setup includes the selected CPU-SVD
    or GPU-QR basis processing, transfer and coarse construction. The public
    wrapper additionally times return transfer, fresh CPU residual and cleanup.
    Problem assembly, reference construction and PDAS are outside this kernel.
    """
    timer.synchronize(torch.cuda.synchronize)
    A = sparse.csr_matrix(matrix(A), dtype=float)
    b, initial, d = validate_linear_inputs(
        A, b, basis, diagonal, x0, rtol, maxiter, refresh, condition_limit
    )
    torch.cuda.reset_peak_memory_stats()
    start = timer.start
    baseline = torch.cuda.memory_allocated()
    timer.mark("conversion")
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
    timer.mark("upload")
    timer.synchronize(torch.cuda.synchronize)
    if basis is None:
        V = None
    elif basis_backend == "cpu_svd":
        Z = orthonormalize(basis)
        timer.mark("basis_processing")
        V = torch.tensor(Z, device="cuda", dtype=torch.float64)
        timer.mark("upload")
        timer.synchronize(torch.cuda.synchronize)
    else:
        V = torch.tensor(basis, device="cuda", dtype=torch.float64)
        timer.mark("upload")
        timer.synchronize(torch.cuda.synchronize)
        V = orthonormalize_gpu(V, torch)
        timer.mark("basis_processing")
        timer.synchronize(torch.cuda.synchronize)
    rank = 0 if V is None else V.shape[1]
    requested_rank = 0 if basis is None else basis.shape[1]
    orthogonalized_rank = rank
    condition, fallback, coarse_failed = 1.0, None, False

    def apply(v):
        return torch.mv(H, v)

    if rank:
        E = V.T @ torch.mm(H, V)
        E = (E + E.T) / 2
        eigenvalues = torch.linalg.eigvalsh(E)
        condition = (
            float(eigenvalues[-1] / eigenvalues[0])
            if bool(torch.isfinite(eigenvalues).all()) and float(eigenvalues[0]) > 0
            else float("inf")
        )
        if not np.isfinite(condition) or condition > condition_limit:
            rank, V, fallback = 0, None, "coarse_condition_limit"
        else:
            chol, info = torch.linalg.cholesky_ex(E)
            coarse_failed = bool(info)
    timer.mark("coarse_or_hierarchy_setup")
    timer.synchronize(torch.cuda.synchronize)

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
    timer.mark("initialization")
    timer.synchronize(torch.cuda.synchronize)
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
    timer.mark("iteration")
    timer.synchronize(torch.cuda.synchronize)
    solve = time.perf_counter() - solve_start
    x_cpu = x.cpu().numpy()
    timer.mark("download")
    timer.synchronize(torch.cuda.synchronize)
    residual = independent_residual(A, x_cpu, b)
    if residual <= acceptance_rtol and not coarse_failed:
        status = "converged"
    elif status == "converged":
        status = "residual_failed"
    timer.mark("verification")
    total = time.perf_counter() - start
    return LinearResult(x_cpu, iterations, residual, status, rank, condition, fallback), {
        "setup_seconds": setup,
        "solve_seconds": solve,
        "return_and_verify_seconds": total - setup - solve,
        "total_seconds": total,
        "peak_torch_bytes": torch.cuda.max_memory_allocated(),
        "baseline_torch_bytes": baseline,
        "requested_rank": requested_rank,
        "orthogonalized_rank": orthogonalized_rank,
        "basis_backend": basis_backend,
        "iteration_rtol": rtol,
        "acceptance_rtol": acceptance_rtol,
    }


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
    *,
    basis_backend="cpu_svd",
    acceptance_rtol=None,
):
    """Verified CUDA solve, including conversion, transfers and temporary cleanup.

    Runtime initialization is the caller's responsibility. Components partition
    the complete call after runtime availability is checked. Explicit barriers
    are separate from asynchronous stage times. PyTorch's allocator remains
    cached; cleanup releases this solve's tensors, not the shared runtime.
    """
    if basis_backend not in {"cpu_svd", "gpu_qr"}:
        raise ValueError("Basis backend must be cpu_svd or gpu_qr")
    acceptance_rtol = (
        rtol if acceptance_rtol is None else positive_real(acceptance_rtol, "Acceptance tolerance")
    )
    if acceptance_rtol < rtol:
        raise ValueError("Acceptance tolerance must be at least the iteration tolerance")
    torch = require_cuda()
    timer = PhaseTimer()
    result, metrics = _gpu_deflated_cg(
        A,
        b,
        basis,
        diagonal,
        x0,
        rtol,
        maxiter,
        refresh,
        condition_limit,
        basis_backend=basis_backend,
        timer=timer,
        torch=torch,
        acceptance_rtol=acceptance_rtol,
    )
    # Returning from the helper releases its temporary GPU tensors. The
    # returned solution owns CPU storage only. No empty_cache() is charged.
    timer.mark("cleanup")
    timer.synchronize(torch.cuda.synchronize)
    metrics.update(timer.finish())
    metrics["return_and_verify_seconds"] = (
        metrics["total_seconds"] - metrics["setup_seconds"] - metrics["solve_seconds"]
    )
    return result, metrics
