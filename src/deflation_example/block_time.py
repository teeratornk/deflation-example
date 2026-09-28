"""Block-in-time preconditioning of the prescribed-flow trajectory Hessian.

With prescribed flow the reduced Hessian ``H = W + alpha A^T W A`` of an
all-at-once backward-Euler trajectory is block tridiagonal in time, and its
slab-diagonal blocks are the exact slab operators. Block-Jacobi sweeps over the
blocks of each inactive restriction give a symmetric positive definite
preconditioner for every sweep count (the eigenvalues of ``D^-1 H_II`` lie in
(0, 2) for a block-tridiagonal SPD matrix). The slab blocks are factored when the
restriction is formed, so their cost falls in the restriction phase of every
active-set update, and the sweeps run inside the host CG recurrence.
"""

import numpy as np
from scipy import sparse

from .coupled_frozen_preconditioner import FrozenSweepPreconditioner
from .mesh_control import WeightedReducedOperator
from .validation import integer


class BlockTimeRestriction(WeightedReducedOperator):
    """A reduced Hessian whose restrictions carry a block-in-time preconditioner."""

    def __init__(self, base, spatial_size, sweeps=3):
        if not isinstance(base, WeightedReducedOperator) or not base.assembled_restriction:
            raise ValueError("Block-in-time preconditioning needs assembled restrictions")
        super().__init__(base.A, base.weights, base.alpha, assembled_restriction=True)
        self.base = base
        self.spatial_size = integer(spatial_size, "Spatial size", 1)
        if base.shape[0] % self.spatial_size:
            raise ValueError("The trajectory size must be a whole number of slabs")
        self.sweeps = integer(sweeps, "Block sweeps", 1)
        self.factorizations = 0

    def assembled(self):
        return self.base.assembled()

    def restrict(self, indices):
        I = np.asarray(indices)
        restricted = sparse.csr_matrix(self.base.restrict(I))
        restricted.preconditioner = FrozenSweepPreconditioner(
            restricted, I // self.spatial_size, sweeps=self.sweeps
        )
        self.factorizations += 1
        return restricted
