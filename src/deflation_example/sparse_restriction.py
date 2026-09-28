"""Inactive-set restrictions as slices of the once-assembled reduced Hessian.

The weighted reduced operator ``H = W + alpha A^T W A`` retains every state-equation
row on restriction, so its restriction to the inactive indices ``I`` is the principal
submatrix ``H[I, I] = W_I + alpha A[:, I]^T W A[:, I]``. Forming it by a sparse
product at every active-set update costs a product over all retained rows; slicing
the matrix assembled once per sequence costs a pass over the selected rows. The
entries are the same sums of the same products, so the two agree to round-off.
"""

import numpy as np
from scipy import sparse

from .mesh_control import WeightedReducedOperator


def principal_submatrix(matrix, indices):
    """``matrix[I][:, I]`` for a CSR matrix, gathered in one vectorized pass.

    Each selected row keeps the entries whose column is selected, renumbered by
    its position in ``indices``; with ascending indices and sorted rows the result
    keeps sorted column indices.
    """
    I = np.asarray(indices, dtype=np.int64)
    position = np.full(matrix.shape[1], -1, dtype=np.int64)
    position[I] = np.arange(len(I))
    starts = matrix.indptr[I].astype(np.int64)
    lengths = matrix.indptr[I + 1].astype(np.int64) - starts
    first = np.cumsum(lengths) - lengths
    entries = np.repeat(starts - first, lengths) + np.arange(int(lengths.sum()))
    columns = position[matrix.indices[entries]]
    keep = columns >= 0
    rows = np.repeat(np.arange(len(I)), lengths)[keep]
    counts = np.bincount(rows, minlength=len(I))
    indptr = np.concatenate(([0], np.cumsum(counts)))
    result = sparse.csr_matrix(
        (matrix.data[entries[keep]], columns[keep], indptr), shape=(len(I), len(I))
    )
    if not np.all(np.diff(I) > 0):
        result.sort_indices()
    return result


def scipy_submatrix(matrix, indices):
    """``matrix[I][:, I]`` through scipy's compiled row and column indexing."""
    result = matrix[indices][:, indices].tocsr()
    result.sort_indices()
    return result


SLICERS = {"scipy": scipy_submatrix, "gather": principal_submatrix}


class PreassembledRestriction(WeightedReducedOperator):
    """A weighted reduced operator whose restrictions slice its assembled matrix."""

    def __init__(self, base, slicer="scipy"):
        if slicer not in SLICERS:
            raise ValueError("Choose the scipy or gather slicer")
        self._slice = SLICERS[slicer]
        if not isinstance(base, WeightedReducedOperator):
            raise TypeError("A weighted reduced operator is required")
        if not base.assembled_restriction:
            raise ValueError("Slicing replaces assembled restrictions only")
        super().__init__(base.A, base.weights, base.alpha, assembled_restriction=True)
        self._matrix = super().assembled()
        self._matrix.sort_indices()

    def assembled(self):
        return self._matrix.copy()

    def restrict(self, indices):
        I = np.asarray(indices)
        if I.ndim != 1 or I.dtype.kind not in "iu" or len(np.unique(I)) != len(I):
            raise ValueError("Inactive indices must be distinct integers")
        if len(I) and (I.min() < 0 or I.max() >= self.shape[0]):
            raise ValueError("Inactive indices are out of range")
        return self._slice(self._matrix, I)
