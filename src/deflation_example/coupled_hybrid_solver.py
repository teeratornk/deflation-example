"""CPU projected CG with CUDA block products for coupled coarse-space processing.

Vector iterations and final original-residual checks use the CPU operator.
Reference and recycling bases share the same optional block acceleration, and
their exact coarse correction may also reside on the device. All transfers,
lazy factor uploads, block products and device corrections remain inside the
caller's solve interval. This backend does not change the nonlinear equations
or targets, and the rank-zero control never performs device work.
"""

import time

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .coupled_cuda import CudaControlJacobian, CudaGaussNewton
from .recycling import inactive_indices
from .solvers import CpuCoarseSpace
from .study_solvers import StudySolver
from .validation import integer


def chunk_bounds(columns, width):
    """Nearly equal column chunks of at most ``width``; one chunk when it fits."""
    columns, width = integer(columns, "Columns", 0), integer(width, "Chunk width", 1)
    if columns <= width:
        return [(0, columns)]
    count = -(-columns // width)
    edges = [round(columns * k / count) for k in range(count + 1)]
    return [(edges[k], edges[k + 1]) for k in range(count)]


class HybridCoupledSolver(StudySolver):
    """Retain CPU vectors and accelerate blocks of at least twenty columns."""

    def __init__(
        self,
        method,
        *,
        block_min_columns=20,
        block_max_columns=100,
        coarse_device="cpu",
        **kwargs,
    ):
        if method not in {"jacobi", "reference", "recycling"}:
            raise ValueError("Hybrid coupled policies are jacobi, reference and recycling")
        if coarse_device not in {"cpu", "cuda"}:
            raise ValueError("The hybrid coarse-space correction runs on cpu or cuda")
        super().__init__(method, device="cpu", **kwargs)
        self.block_min_columns = integer(block_min_columns, "CUDA block threshold", 2)
        # Device block products are applied in bounded column chunks: every
        # per-slab triangular plan holds buffers proportional to the chunk width.
        self.block_max_columns = integer(block_max_columns, "CUDA block chunk width", 1)
        self.coarse_device = coarse_device
        self.cpu_jacobian = self.device_jacobian = None
        self.block_records = []
        self.coarse_records = []

    def close(self):
        if self.device_jacobian is not None:
            self.device_jacobian.close()
            self.device_jacobian.cp.cuda.get_current_stream().synchronize()
        self.cpu_jacobian = self.device_jacobian = None
        super().close()

    def solve(self, B, b, indices, initial=None):
        # Scope these logs to the whole refinement call, including correction solves.
        self.block_records, self.coarse_records = [], []
        result, timing = super().solve(B, b, indices, initial)
        timing["hybrid_block_processing"] = {
            "minimum_columns": self.block_min_columns,
            "calls": list(self.block_records),
            "seconds": sum(
                row["seconds"]
                for row in self.block_records
                if not row.get("enclosed_by_operator_block", False)
            ),
            "timing_scope": "Included in the CPU kernel and recycling-selection intervals; do not add again to total_seconds.",
            "vector_and_verification_device": "cpu",
            "block_device": "cuda",
        }
        if self.coarse_device == "cuda":
            spaces = [space.report() for space in self.coarse_records]
            timing["hybrid_coarse_correction"] = {
                "device": "cuda",
                "spaces": spaces,
                "applications": sum(row["applications"] for row in spaces),
                "seconds": sum(
                    row["upload_seconds"]
                    + row["orthonormalization_seconds"]
                    + row["setup_seconds"]
                    + row["correction_seconds"]
                    for row in spaces
                ),
                "resident_basis_bytes": max(
                    (row["resident_basis_bytes"] for row in spaces), default=0
                ),
                "resident_operator_product_bytes": max(
                    (row["resident_operator_product_bytes"] for row in spaces), default=0
                ),
                "timing_scope": "Included in components_seconds.iteration; do not add again to total_seconds.",
            }
            if spaces:
                # The operator-basis product lives on the device; no host copy is held.
                timing["cached_operator_product_bytes"] = 0
        return result, timing

    def _solve_once(
        self, B, b, indices, initial=None, *, target=None, cap=None, verify_candidates=False
    ):
        indices = inactive_indices(indices)
        parent = getattr(B, "coupled_parent", None)
        if parent is None or not np.array_equal(indices, B.inactive_indices):
            raise ValueError("A matching restricted coupled Gauss--Newton operator is required")
        normal = None

        def device_operator():
            nonlocal normal
            if self.cpu_jacobian is not parent.jacobian:
                tick = time.perf_counter()
                if self.device_jacobian is not None:
                    self.device_jacobian.close()
                self.cpu_jacobian = self.device_jacobian = None
                self.device_jacobian = CudaControlJacobian(parent.jacobian)
                self.cpu_jacobian = parent.jacobian
                self.device_jacobian.cp.cuda.get_current_stream().synchronize()
                # Factor upload recurs once per nonlinear iterate; the triangular
                # analyses follow in the first block product of each width.
                self.block_records.append(
                    {
                        "kind": "factor_upload",
                        "columns": 0,
                        "factor_upload": True,
                        "seconds": time.perf_counter() - tick,
                    }
                )
            if normal is None:
                normal = CudaGaussNewton(
                    self.device_jacobian,
                    parent.weights,
                    parent.alpha,
                    parent.damping,
                    corrections=getattr(parent, "corrections", ()),
                ).restrict(indices)
            return normal

        def block(vectors):
            if vectors.shape[1] < self.block_min_columns:
                return B @ vectors
            start = time.perf_counter()
            log_start = len(self.block_records)
            apply = device_operator()
            for row in self.block_records[log_start:]:
                row["enclosed_by_operator_block"] = True
            cp = self.device_jacobian.cp
            bounds = chunk_bounds(vectors.shape[1], self.block_max_columns)
            answer = np.concatenate([cp.asnumpy(apply(vectors[:, a:b])) for a, b in bounds], axis=1)
            cp.cuda.get_current_stream().synchronize()
            self.block_records.append(
                {
                    "kind": "operator_block",
                    "includes_factor_upload": True,
                    "columns": vectors.shape[1],
                    "chunks": len(bounds),
                    "factor_upload": False,
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
        self.coarse_factory = None
        if self.coarse_device == "cuda" and self.method != "jacobi":

            def factory(A, basis, condition_limit):
                if basis is None or basis.shape[1] == 0:
                    return CpuCoarseSpace(A, basis, condition_limit)
                from .coupled_hybrid_coarse import CudaCoarseSpace

                apply = device_operator()
                space = CudaCoarseSpace(
                    self.device_jacobian.cp,
                    apply,
                    A.shape[0],
                    basis,
                    condition_limit,
                    chunk=self.block_max_columns,
                )
                self.coarse_records.append(space)
                return space

            self.coarse_factory = factory
        try:
            return super()._solve_once(
                wrapped,
                b,
                indices,
                initial,
                target=target,
                cap=cap,
                verify_candidates=verify_candidates,
            )
        finally:
            self.coarse_factory = None
