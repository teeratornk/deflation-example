"""Optional float64 CUDA linear kernels. Importing this module does not load Torch."""

import time
import numpy as np
from scipy import sparse
from scipy.linalg import norm
from .solvers import LinearResult, independent_residual, orthonormalize, validate_linear_inputs
from .validation import matrix, positive_real, real_array
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
    direction_callback,
    completion_callback,
    cache_operator_product,
    device_basis=None,
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
    basis = None if basis is None else real_array(basis, "Basis")
    if device_basis is not None:
        if (basis is not None or basis_backend != "gpu_qr" or not torch.is_tensor(device_basis)
            or device_basis.device != torch.device("cuda", torch.cuda.current_device())
            or device_basis.dtype != torch.float64 or device_basis.ndim != 2
            or device_basis.shape[0] != len(b) or not bool(torch.isfinite(device_basis).all())):
            raise ValueError("Device basis must be finite float64 on the current CUDA device, with GPU QR and no CPU basis")
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
    if device_basis is not None:
        V = orthonormalize_gpu(device_basis, torch)
        timer.mark("basis_processing")
        timer.synchronize(torch.cuda.synchronize)
    elif basis is None:
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
    requested_rank = device_basis.shape[1] if device_basis is not None else (0 if basis is None else basis.shape[1])
    orthogonalized_rank = rank
    condition, fallback, coarse_failed = 1.0, None, False

    def apply(v):
        return torch.mv(H, v)

    AV = None
    if rank:
        AV = torch.mm(H, V)
        E = V.T @ AV
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
        if not cache_operator_product or not rank:
            AV = None
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
        if rank and AV is not None:
            return z - V @ torch.cholesky_solve((AV.T @ z)[:, None], chol)[:, 0]
        return z - Q(apply(z)) if rank else z

    rhs_scale = float(np.max(np.abs(b))) or 1.0
    rhs_norm = float(norm(b / rhs_scale)) or 1.0

    def small_residual(residual):
        value = float(torch.linalg.vector_norm(residual / rhs_scale)) / rhs_norm
        return np.isfinite(value) and value <= rtol

    status, iterations = "breakdown" if coarse_failed else "maxiter", 0
    if rank and not coarse_failed:
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
                if direction_callback is not None:
                    direction_callback(p)
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
                    if rank:
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
    completion_metrics = None
    if completion_callback is not None:
        completion_metrics = completion_callback(H, diag, V, status)
        timer.mark("basis_processing")
        timer.synchronize(torch.cuda.synchronize)
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
        "completion": completion_metrics,
        "operator_product_cached": AV is not None,
        "cached_operator_product_bytes": 0 if AV is None else AV.numel() * AV.element_size(),
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
    direction_callback=None,
    completion_callback=None,
    cache_operator_product=False,
    device_basis=None,
):
    """Verified CUDA solve, including conversion, transfers and temporary cleanup.

    Runtime initialization is the caller's responsibility. Components partition
    the complete call after runtime availability is checked. Explicit barriers
    are separate from asynchronous stage times. PyTorch's allocator remains
    cached; cleanup releases this solve's tensors, not the shared runtime.

    Optional callbacks support bounded recycling. The direction callback receives
    a read-only view of each executed search direction and must copy directions
    it retains. After independent verification, the completion callback receives
    the GPU matrix, diagonal, deployed coarse basis and final status. Its work
    belongs to basis_processing and total time; it must preserve solver inputs.
    """
    if basis_backend not in {"cpu_svd", "gpu_qr"}:
        raise ValueError("Basis backend must be cpu_svd or gpu_qr")
    if not isinstance(cache_operator_product, bool):
        raise ValueError("Operator-product caching must be Boolean")
    for callback in (direction_callback, completion_callback):
        if callback is not None and not callable(callback):
            raise ValueError("Solver callbacks must be callable")
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
        direction_callback=direction_callback,
        completion_callback=completion_callback,
        cache_operator_product=cache_operator_product,
        device_basis=device_basis,
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
