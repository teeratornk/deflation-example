"""Device-resident exact coarse correction for the hybrid coupled solver.

The host CG recurrence is unchanged. The restricted basis is orthonormalized on
the device, the operator-basis product is one device block product, and each
coarse correction is two device matrix-vector products with one vector
transfer in each direction. Rank, condition and breakdown semantics follow the
host coarse space, so the projected recurrence sees the same coarse policy.
"""

import time

import numpy as np

from .coupled_cuda_solver import _orthogonalize


class CudaCoarseSpace:
    """Coarse basis, operator product and Cholesky factor kept on the GPU."""

    device = "cuda"

    def __init__(self, cp, apply, size, basis, condition_limit, chunk=100):
        from cupyx.scipy.linalg import solve_triangular

        from .coupled_hybrid_solver import chunk_bounds

        self.cp, self.solve_triangular = cp, solve_triangular
        self.chunk = chunk
        start = time.perf_counter()
        empty = basis is None or basis.shape[1] == 0
        Z = cp.empty((size, 0)) if empty else cp.asarray(basis, dtype=cp.float64)
        cp.cuda.get_current_stream().synchronize()
        tick = time.perf_counter()
        self.upload_seconds = tick - start
        Z = _orthogonalize(cp, Z)
        cp.cuda.get_current_stream().synchronize()
        self.orthonormalization_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        self.requested_rank = Z.shape[1]
        self.condition, self.fallback, self.breakdown = 1.0, None, False
        self.factor, self.AZ = None, None
        if Z.shape[1]:
            # Bounded chunks keep every per-slab triangular plan within memory.
            self.AZ = cp.concatenate(
                [apply(Z[:, a:b]) for a, b in chunk_bounds(Z.shape[1], chunk)], axis=1
            )
            E = Z.T @ self.AZ
            E = (E + E.T) / 2
            eigenvalues = cp.linalg.eigvalsh(E)
            smallest = float(eigenvalues[0])
            self.condition = float(eigenvalues[-1]) / smallest if smallest > 0 else float("inf")
            if np.isfinite(self.condition) and self.condition <= condition_limit:
                try:
                    self.factor = cp.linalg.cholesky(E)
                except cp.linalg.LinAlgError:
                    self.breakdown = True
            else:
                Z, self.AZ = Z[:, :0], None
                self.fallback = "coarse_condition_limit"
        cp.cuda.get_current_stream().synchronize()
        self.setup_seconds = time.perf_counter() - tick
        self.Z = Z
        self.rank = Z.shape[1]
        self.applications, self.seconds = 0, 0.0

    def _coefficients(self, y):
        lower = self.solve_triangular(self.factor, y, lower=True)
        return self.solve_triangular(self.factor.T, lower, lower=False)

    def _apply(self, product, v):
        tick = time.perf_counter()
        device = self.cp.asarray(v, dtype=self.cp.float64)
        answer = self.cp.asnumpy(self.Z @ self._coefficients(product.T @ device))
        self.cp.cuda.get_current_stream().synchronize()
        self.seconds += time.perf_counter() - tick
        self.applications += 1
        return answer

    def correct(self, v):
        return self._apply(self.Z, v)

    def precondition_correction(self, z):
        return self._apply(self.AZ, z)

    def storage(self):
        return {
            "device": self.device,
            "resident_basis_bytes": int(self.Z.nbytes),
            "resident_operator_product_bytes": 0 if self.AZ is None else int(self.AZ.nbytes),
        }

    def report(self):
        return {
            **self.storage(),
            "requested_rank": self.requested_rank,
            "rank": self.rank,
            "coarse_condition": self.condition,
            "fallback_reason": self.fallback,
            "breakdown": self.breakdown,
            "upload_seconds": self.upload_seconds,
            "orthonormalization_seconds": self.orthonormalization_seconds,
            "setup_seconds": self.setup_seconds,
            "applications": self.applications,
            "correction_seconds": self.seconds,
        }
