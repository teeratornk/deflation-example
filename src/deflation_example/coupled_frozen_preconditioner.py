"""Velocity-frozen space-time preconditioner for the coupled Gauss--Newton system.

The coupled operator ``W + alpha J^T W J`` applies the tangent twice, and each
application solves every slab's momentum equations, so it costs seconds. The
velocity-frozen operator ``P = W + alpha T^T W T`` keeps the thermal blocks and
drops that coupling, so it costs milliseconds and is already assembled for the
preconditioning diagonal. Measured on the corrected model at 64 slabs, one exact
application takes 2.116 s against 0.038 s frozen, and the frozen operator is close
enough that lagging the velocity coupling contracts the error by about 3.5 per
sweep over the whole horizon.

``P`` is inverted approximately by block-Jacobi sweeps over its slab-diagonal
blocks. Within a sweep the slabs are independent, which is the time-parallel form,
and an optional coarse correction between sweeps is available for windows long
enough to need one. The preconditioner never defines the direction: the outer
iteration applies the exact operator and accepts on its residual, so this changes
what a solve costs and not what it computes.

Without an additional coarse correction, the SPD block-tridiagonal temporal
operator has block-Jacobi eigenvalues in (0, 2). Exact block solves therefore give
an SPD sweep preconditioner for every positive sweep count. This property applies
to the declared temporal operator; an arbitrary supplied matrix or additional
correction needs its own justification. The default remains three sweeps.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .mesh_control import WeightedReducedOperator
from .validation import integer, real_array


class FrozenSweepPreconditioner:
    """Approximate inverse of the restricted frozen normal operator.

    ``operator`` is the assembled restricted ``P``; ``slabs`` assigns each of its
    rows to a time slab, which is what makes the block solves independent.
    """

    def __init__(self, operator, slabs, sweeps=3, correction=None):
        self.operator = sparse.csr_matrix(operator)
        if self.operator.shape[0] != self.operator.shape[1]:
            raise ValueError("The frozen preconditioner requires a square operator")
        slabs = np.asarray(slabs)
        if slabs.shape != (self.operator.shape[0],) or slabs.dtype.kind not in "iu":
            raise ValueError("Every row needs an integer slab index")
        self.sweeps = integer(sweeps, "Frozen preconditioner sweeps", 1)
        if correction is not None and not callable(correction):
            raise ValueError("A coarse correction must be callable")
        self.correction = correction
        self.positions = [np.flatnonzero(slabs == slab) for slab in np.unique(slabs)]
        self.factors = []
        for rows in self.positions:
            block = self.operator[rows][:, rows].tocsc()
            self.factors.append(splu(block))
        self.applications = 0

    def storage(self):
        """Nonzeros retained, for the record that reports what the run held."""
        return {
            "operator_nonzeros": int(self.operator.nnz),
            "slabs": len(self.positions),
            "sweeps": self.sweeps,
            "coarse_corrected": self.correction is not None,
        }

    def __call__(self, residual):
        r = real_array(residual, "Preconditioner residual")
        x = np.zeros_like(r)
        for _ in range(self.sweeps):
            defect = r - self.operator @ x
            for rows, factor in zip(self.positions, self.factors, strict=True):
                x[rows] += factor.solve(defect[rows])
            if self.correction is not None:
                x = x + self.correction(r - self.operator @ x)
        self.applications += 1
        return x


def frozen_normal_operator(problem, evaluation, damping=0.0):
    """``W + alpha T^T W T`` plus the damping the quadratic subproblem carries.

    This is the operator whose diagonal ``preconditioning_diagonal`` already
    returns; the whole of it is kept here instead of only that diagonal.
    """
    return WeightedReducedOperator(evaluation.frozen_operator, problem.weights, problem.alpha), (
        float(damping) * np.asarray(problem.weights, dtype=float)
    )


def frozen_preconditioner_factory(
    problem, evaluation, damping=0.0, sweeps=3, correction=None, *, restriction="product"
):
    """Build the per-active-set factory the quadratic subproblem attaches.

    The restriction is assembled once per active set, which the frozen operator
    can afford: it is sparse and thermal, unlike the exact operator, which is
    matrix-free precisely because assembling it would mean momentum solves.
    ``restriction="submatrix"`` assembles the full frozen normal matrix once per
    factory and slices it thereafter. Its construction and storage must be charged
    to the comparison. The default retains the measured sparse-product path.
    """
    if restriction not in {"product", "submatrix"}:
        raise ValueError("Choose product or submatrix frozen restriction")
    operator, damped = frozen_normal_operator(problem, evaluation, damping)
    if restriction == "submatrix":
        from .sparse_restriction import PreassembledRestriction

        operator = PreassembledRestriction(operator)
    spatial_size = problem.spatial_size

    def factory(indices):
        indices = np.asarray(indices)
        restricted = sparse.csr_matrix(operator.restrict(indices))
        if damping:
            restricted = (restricted + sparse.diags(damped[indices])).tocsr()
        return FrozenSweepPreconditioner(
            restricted, indices // spatial_size, sweeps=sweeps, correction=correction
        )

    return factory
