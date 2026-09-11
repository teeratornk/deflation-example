"""Optional CUDA applications of already factored coupled derivatives.

Momentum Jacobian factorization remains on the CPU. CuPy copies those exact
factors and performs tangent and transpose triangular solves on the GPU.
This module imports CUDA libraries only when its adapter is constructed.
"""

import numpy as np


class CudaControlJacobian:
    """Keep the complete thermal and momentum tangent trajectory on one GPU."""

    def __init__(self, jacobian):
        try:
            import cupy as cp
            from cupyx.scipy import sparse
            from .coupled_triangular import PersistentSuperLU
        except ImportError as error:
            raise RuntimeError(
                "Coupled CUDA actions require cupy-cuda12x and a CUDA device"
            ) from error
        self.cp = cp
        self.shape = jacobian.shape
        self.slabs, self.spatial_size = jacobian.slabs, jacobian.spatial_size
        self.thermal = sparse.csr_matrix(jacobian.thermal)
        self.buoyancy = sparse.csr_matrix(jacobian.buoyancy)
        self.velocity_actions = tuple(sparse.csr_matrix(A) for A in jacobian.velocity_actions)
        self.history = tuple(sparse.csr_matrix(A) for A in jacobian.history)
        self.factors = tuple(PersistentSuperLU(factor) for factor in jacobian.factors)
        cp.cuda.get_current_stream().synchronize()

    def apply(self, vectors, *, transpose=False):
        cp = self.cp
        x = cp.asarray(vectors, dtype=cp.float64)
        if x.ndim not in (1, 2) or x.shape[0] != self.shape[0]:
            raise ValueError("CUDA derivative directions must match the complete trajectory")
        vector = x.ndim == 1
        x = x.reshape(self.shape[0], -1)
        columns = x.shape[1]
        if not columns:
            return cp.empty_like(x)
        blocks = x.reshape(self.slabs, self.spatial_size, columns)
        result = ((self.thermal.T if transpose else self.thermal) @ x).reshape(blocks.shape)
        adjacent = cp.zeros((self.buoyancy.shape[0], columns))
        if transpose:
            for n in range(self.slabs - 1, -1, -1):
                rhs = self.velocity_actions[n].T @ blocks[n]
                if n + 1 < self.slabs:
                    rhs += self.history[n + 1].T @ adjacent
                adjacent = self.factors[n].solve(rhs, trans="T")
                result[n] += self.buoyancy.T @ adjacent
        else:
            for n in range(self.slabs):
                rhs = self.buoyancy @ blocks[n] + self.history[n] @ adjacent
                adjacent = self.factors[n].solve(rhs)
                result[n] += self.velocity_actions[n] @ adjacent
        return result.ravel() if vector else result.reshape(self.shape[0], columns)

    def storage_bytes(self):
        matrices = [self.thermal, self.buoyancy, *self.velocity_actions, *self.history]
        matrices += [A for factor in self.factors for A in (factor.L, factor.U)]
        total = sum(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes for A in matrices)
        total += sum(factor.perm_r.nbytes + factor.perm_c.nbytes for factor in self.factors)
        total += sum(factor.storage_bytes() for factor in self.factors)
        return int(total)

    def close(self):
        for factor in self.factors:
            factor.close()


class CudaGaussNewton:
    """Full-space normal products and zero-extension inactive restrictions."""

    def __init__(self, jacobian, weights, alpha, damping=0.0, corrections=()):
        self.jacobian, self.cp = jacobian, jacobian.cp
        self.shape = jacobian.shape
        self.weights = self.cp.asarray(weights, dtype=self.cp.float64)
        self.alpha, self.damping = float(alpha), float(damping)
        self.corrections = tuple((self.cp.asarray(q), self.cp.asarray(y)) for q, y in corrections)
        if any(q.shape != (self.shape[0],) or y.shape != q.shape for q, y in self.corrections):
            raise ValueError("Curvature corrections must match the complete state")
        if (
            self.weights.shape != (self.shape[0],)
            or not np.isfinite([alpha, damping]).all()
            or alpha <= 0
            or damping < 0
            or not bool(self.cp.all(self.cp.isfinite(self.weights) & (self.weights > 0)))
        ):
            raise ValueError("CUDA Gauss--Newton requires positive weights and regularization")

    def restrict(self, indices):
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or len(np.unique(indices)) != len(indices)
            or (len(indices) and (indices.min() < 0 or indices.max() >= self.shape[0]))
        ):
            raise ValueError("Inactive indices must be distinct integers in range")
        device_indices = self.cp.asarray(indices, dtype=self.cp.int64)
        return lambda vectors: self._apply(vectors, device_indices)

    def apply(self, vectors, indices=None):
        return self._apply(vectors) if indices is None else self.restrict(indices)(vectors)

    def _apply(self, vectors, indices=None):
        cp = self.cp
        x = cp.asarray(vectors, dtype=cp.float64)
        vector = x.ndim == 1
        if x.ndim not in (1, 2):
            raise ValueError("A vector or column block is required")
        x = x[:, None] if vector else x
        if indices is not None:
            if len(indices) != len(x):
                raise ValueError("Directions must match the inactive indices")
            full = cp.zeros((self.shape[0], x.shape[1]))
            full[indices] = x
        else:
            if x.shape[0] != self.shape[0]:
                raise ValueError("Directions must match the full state space")
            full = x
        result = (1 + self.damping) * self.weights[
            :, None
        ] * full + self.alpha * self.jacobian.apply(
            self.weights[:, None] * self.jacobian.apply(full), transpose=True
        )
        for q, y in self.corrections:
            result -= q[:, None] * (q @ full)
            result += y[:, None] * (y @ full)
        if indices is not None:
            result = result[indices]
        return result.ravel() if vector else result
