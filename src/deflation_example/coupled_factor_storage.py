"""Recompute sparse momentum factors from immutable matrix checkpoints.

Each checkpoint retains the exact sparse Jacobian at one linearization. LU
factors exist only during a vector or block solve. The resulting extra work is
part of the caller's solve interval. This CPU policy trades factorization work
for storage and leaves the discretization and tangent equations unchanged.
"""

import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


class RecomputedLU:
    """The SuperLU solve interface backed by a fixed sparse matrix checkpoint."""

    def __init__(self, matrix):
        if np.iscomplexobj(matrix):
            raise ValueError("Momentum matrix checkpoints must be real")
        value = sparse.csc_matrix(matrix, dtype=float, copy=True)
        value.sum_duplicates()
        value.sort_indices()
        if (
            value.shape[0] != value.shape[1]
            or not value.shape[0]
            or not np.isfinite(value.data).all()
        ):
            raise ValueError("A momentum checkpoint requires a finite nonempty square matrix")
        for array in (value.data, value.indices, value.indptr):
            array.flags.writeable = False
        self._matrix = value
        self.shape = value.shape
        self.factorizations = 0
        self.factor_seconds = 0.0
        self.solve_seconds = 0.0

    @property
    def checkpoint_bytes(self):
        return sum(a.nbytes for a in (self._matrix.data, self._matrix.indices, self._matrix.indptr))

    def solve(self, rhs, trans="N"):
        if trans not in {"N", "T", "H"}:
            raise ValueError("Choose N, T or H for the momentum solve")
        values = np.asarray(rhs)
        if (
            values.ndim not in {1, 2}
            or values.shape[0] != self.shape[0]
            or np.iscomplexobj(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Momentum right-hand sides must be finite real vectors or blocks")
        values = np.asarray(values, dtype=float)
        if values.ndim == 2 and values.shape[1] == 0:
            return np.empty_like(values)
        start = time.perf_counter()
        # SuperLU may canonicalize its input; the private stored matrix stays
        # immutable even when a SciPy version requests writable buffers.
        factor = splu(self._matrix.copy())
        self.factorizations += 1
        self.factor_seconds += time.perf_counter() - start
        start = time.perf_counter()
        try:
            result = factor.solve(values, trans=trans)
            if not np.isfinite(result).all():
                raise np.linalg.LinAlgError(
                    "The momentum checkpoint solve returned nonfinite values"
                )
            return result
        finally:
            self.solve_seconds += time.perf_counter() - start
            del factor
