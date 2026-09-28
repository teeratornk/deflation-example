"""CUDA application of the CPU-factorized frozen temporal sweep preconditioner.

Sparse assembly and factorization remain on the CPU. The same factors and
restricted matrix are copied once per inactive set; every sweep then runs on
the device. No host residual or block solution is transferred inside a sweep.
"""

from .coupled_frozen_preconditioner import FrozenSweepPreconditioner


class CudaFrozenSweepPreconditioner:
    """Retain the declared block-Jacobi polynomial and its CPU sparse factors."""

    def __init__(self, host, *, layout="serial"):
        if not isinstance(host, FrozenSweepPreconditioner) or host.correction is not None:
            raise ValueError("CUDA sweeps require an unmodified frozen sweep preconditioner")
        import cupy as cp
        from cupyx.scipy import sparse
        from .coupled_triangular import PersistentSuperLU

        self.cp, self.sweeps = cp, host.sweeps
        if layout not in {"serial", "block_diagonal"}:
            raise ValueError("Choose serial or block_diagonal independent slab solves")
        self.layout = layout
        self.shape = host.operator.shape
        self.closed = False
        self.operator, self.positions, self.factors = None, [], []
        self.applications = 0
        try:
            self.operator = sparse.csr_matrix(host.operator)
            if layout == "block_diagonal" and host.factors:
                self.positions = [cp.concatenate([cp.asarray(rows) for rows in host.positions])]
                self.factors = [PersistentSuperLU.block_diagonal(host.factors)]
            else:
                self.positions = [cp.asarray(rows) for rows in host.positions]
                for factor in host.factors:
                    self.factors.append(PersistentSuperLU(factor))
            cp.cuda.get_current_stream().synchronize()
        except BaseException:
            self.close()
            raise

    def __call__(self, residual):
        if self.closed:
            raise RuntimeError("The CUDA frozen preconditioner is closed")
        cp = self.cp
        if cp.iscomplexobj(residual):
            raise ValueError("Use real frozen-preconditioner residuals")
        r = cp.asarray(residual, dtype=cp.float64)
        if r.ndim not in (1, 2) or r.shape[0] != self.shape[0]:
            raise ValueError("Residuals must match the restricted frozen operator")
        x = cp.zeros_like(r)
        for _ in range(self.sweeps):
            defect = r - self.operator @ x
            for rows, factor in zip(self.positions, self.factors, strict=True):
                x[rows] += factor.solve(defect[rows])
        self.applications += 1
        return x

    def storage_bytes(self):
        if self.closed:
            return 0
        matrices = [self.operator, *(A for factor in self.factors for A in (factor.L, factor.U))]
        arrays = [*self.positions]
        arrays.extend(
            a
            for f in self.factors
            for a in (f.perm_r, f.perm_c, f.factor._perm_r_rev, f.factor._perm_c_rev)
        )
        return int(
            sum(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes for A in matrices)
            + sum(a.nbytes for a in arrays)
            + sum(f.storage_bytes() for f in self.factors)
        )

    def close(self):
        for factor in self.factors:
            factor.close()
        self.factors, self.positions, self.operator = [], [], None
        self.closed = True
