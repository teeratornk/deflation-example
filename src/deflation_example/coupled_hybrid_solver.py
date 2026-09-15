"""CPU projected CG with CUDA block products for coupled coarse-space processing.

Vector iterations and final original-residual checks use the CPU operator.
Reference and recycling bases share the same optional block acceleration.
All transfers, lazy factor uploads and block products remain inside the caller's
solve interval. This backend does not change the nonlinear equations or targets.
"""

import time

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .coupled_cuda import CudaControlJacobian, CudaGaussNewton
from .recycling import inactive_indices
from .study_solvers import StudySolver
from .validation import integer


class HybridCoupledSolver(StudySolver):
    """Retain CPU vectors and accelerate blocks of at least twenty columns."""

    def __init__(self, method, *, block_min_columns=20, coarse_device="cpu", **kwargs):
        if method not in {"jacobi", "reference", "recycling"}:
            raise ValueError("Hybrid coupled policies are jacobi, reference and recycling")
        if coarse_device not in {"cpu", "cuda"}:
            raise ValueError("The hybrid coarse-space correction runs on cpu or cuda")
        super().__init__(method, device="cpu", **kwargs)
        self.block_min_columns = integer(block_min_columns, "CUDA block threshold", 2)
        self.coarse_device = coarse_device
        self.cpu_jacobian = self.device_jacobian = None
        self.block_records = []

    def close(self):
        if self.device_jacobian is not None:
            self.device_jacobian.close()
            self.device_jacobian.cp.cuda.get_current_stream().synchronize()
        self.cpu_jacobian = self.device_jacobian = None
        super().close()

    def solve(self, B, b, indices, initial=None):
        # Scope this log to the whole refinement call, including correction solves.
        self.block_records = []
        result, timing = super().solve(B, b, indices, initial)
        timing["hybrid_block_processing"] = {
            "minimum_columns": self.block_min_columns,
            "calls": list(self.block_records),
            "seconds": sum(row["seconds"] for row in self.block_records),
            "timing_scope": "Included in the CPU kernel and recycling-selection intervals; do not add again to total_seconds.",
            "vector_and_verification_device": "cpu",
            "block_device": "cuda",
        }
        return result, timing

    def _solve_once(
        self, B, b, indices, initial=None, *, target=None, cap=None, verify_candidates=False
    ):
        indices = inactive_indices(indices)
        parent = getattr(B, "coupled_parent", None)
        if parent is None or not np.array_equal(indices, B.inactive_indices):
            raise ValueError("A matching restricted coupled Gauss--Newton operator is required")
        normal = None

        def block(vectors):
            nonlocal normal
            if vectors.shape[1] < self.block_min_columns:
                return B @ vectors
            start = time.perf_counter()
            uploaded = False
            upload_seconds = 0.0
            if self.cpu_jacobian is not parent.jacobian:
                if self.device_jacobian is not None:
                    self.device_jacobian.close()
                self.cpu_jacobian = self.device_jacobian = None
                self.device_jacobian = CudaControlJacobian(parent.jacobian)
                self.cpu_jacobian = parent.jacobian
                uploaded = True
                # Factor upload and triangular analyses recur once per nonlinear
                # iterate; keep them visible rather than folded into block time.
                self.device_jacobian.cp.cuda.get_current_stream().synchronize()
                upload_seconds = time.perf_counter() - start
            if normal is None:
                normal = CudaGaussNewton(
                    self.device_jacobian,
                    parent.weights,
                    parent.alpha,
                    parent.damping,
                    corrections=getattr(parent, "corrections", ()),
                ).restrict(indices)
            cp = self.device_jacobian.cp
            answer = cp.asnumpy(normal(vectors))
            cp.cuda.get_current_stream().synchronize()
            self.block_records.append(
                {
                    "kind": "operator_block",
                    "columns": vectors.shape[1],
                    "factor_upload": uploaded,
                    "upload_and_analysis_seconds": upload_seconds,
                    "seconds": time.perf_counter() - start,
                }
            )
            return answer

        wrapped = LinearOperator(
            B.shape,
            matvec=lambda x: B @ x,
            rmatvec=lambda x: B @ x,
            matmat=block,
            rmatmat=block,
            dtype=np.dtype(float),
        )
        wrapped.diagonal = B.diagonal
        return super()._solve_once(
            wrapped,
            b,
            indices,
            initial,
            target=target,
            cap=cap,
            verify_candidates=verify_candidates,
        )
