"""Small interchangeable kernel adapters; CUDA is imported only when requested."""

from typing import Protocol
import time
import numpy as np
from scipy.sparse import csr_matrix
from .solvers import LinearResult, deflated_cg


class LinearKernel(Protocol):
    """Solve an SPD system without mutating inputs; report the complete kernel cost."""

    def __call__(
        self,
        matrix: csr_matrix,
        rhs: np.ndarray,
        basis: np.ndarray | None,
        diagonal: np.ndarray,
        *,
        rtol: float,
        maxiter: int,
    ) -> tuple[LinearResult, dict[str, float | int]]: ...


def cpu_kernel(matrix, rhs, basis, diagonal, *, rtol, maxiter):
    start = time.perf_counter()
    result = deflated_cg(matrix, rhs, basis, diagonal, rtol=rtol, maxiter=maxiter)
    return result, {"total_seconds": time.perf_counter() - start}


def cuda_kernel(matrix, rhs, basis, diagonal, *, rtol, maxiter):
    from .gpu import gpu_deflated_cg

    result, metrics = gpu_deflated_cg(matrix, rhs, basis, diagonal, rtol=rtol, maxiter=maxiter)
    # The demo kernel contract accepts scalar timing and memory measurements.
    # Detailed benchmark metadata and nested phase timings remain available
    # from gpu_deflated_cg and the dedicated benchmark commands.
    fields = (
        "total_seconds",
        "setup_seconds",
        "solve_seconds",
        "return_and_verify_seconds",
        "peak_torch_bytes",
        "baseline_torch_bytes",
    )
    return result, {key: metrics[key] for key in fields}


def default_kernels(device) -> dict[str, LinearKernel]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    kernels = {"cpu": cpu_kernel}
    if device == "cuda":
        kernels["cuda"] = cuda_kernel
    return kernels
