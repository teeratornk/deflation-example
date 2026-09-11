"""Matched CUDA CG policies for matrix-free coupled Gauss--Newton systems.

All policies use the same derivative factors, positive diagonal, projected
recurrence, precision, and independently evaluated CPU residual. The inherited
residual-correction wrapper retains the initial guess and shares its iteration
budget across at most four correction solves.
"""

from collections import deque
import time

import numpy as np

from .coupled_cuda import CudaControlJacobian, CudaGaussNewton
from .recycling import inactive_indices
from .solvers import LinearResult, independent_residual, validate_linear_inputs
from .study_solvers import StudySolver
from .timing import PhaseTimer


def _orthogonalize(cp, basis, tolerance=1e-12):
    if 0 in basis.shape:
        return basis[:, :0]
    Q, R = cp.linalg.qr(basis, mode="reduced")
    U, singular, _ = cp.linalg.svd(R, full_matrices=False)
    keep = singular > tolerance * singular[0]
    return Q if bool(cp.all(keep)) else Q @ U[:, keep]


class CudaRecycleSpace:
    """Retain old coarse vectors and new directions in preconditioned coordinates."""

    def __init__(self, cp, rank, window):
        self.cp, self.rank, self.window = cp, rank, window
        self.clear()

    def clear(self):
        self.indices = np.empty(0, dtype=np.int64)
        self.basis = self.cp.empty((0, 0), dtype=self.cp.float64)
        self.directions = deque(maxlen=self.window)
        self.current = None
        self.observed = 0

    def begin(self, indices):
        self.current = indices.copy()
        self.directions.clear()
        self.observed = 0
        basis = self.cp.zeros((len(indices), self.basis.shape[1]))
        if basis.shape[1]:
            _, previous, current = np.intersect1d(
                self.indices, indices, assume_unique=True, return_indices=True
            )
            basis[self.cp.asarray(current)] = self.basis[self.cp.asarray(previous)]
        return basis

    def capture(self, direction):
        self.observed += 1
        if self.rank:
            self.directions.append(direction.copy())

    def finish(self, apply, diagonal, deployed, status):
        if status != "converged":
            self.clear()
            return {"status": "discarded_after_failed_solve", "selected_rank": 0}
        cp = self.cp
        blocks = ([deployed] if deployed.shape[1] else []) + list(self.directions)
        candidates = cp.column_stack(blocks) if blocks else cp.empty((len(diagonal), 0))
        sqrt_d = cp.sqrt(diagonal)
        scaled = sqrt_d[:, None] * candidates
        if scaled.shape[1]:
            norms = cp.linalg.norm(scaled, axis=0)
            nonzero = norms > 0
            scaled = scaled[:, nonzero] / norms[nonzero]
        Q = _orthogonalize(cp, scaled)
        physical = Q / sqrt_d[:, None]
        if physical.shape[1] and self.rank:
            projected = physical.T @ apply(physical)
            _, directions = cp.linalg.eigh((projected + projected.T) / 2)
            selected = physical @ directions[:, : self.rank]
        else:
            selected = physical[:, :0]
        metrics = {
            "status": "selected",
            "policy": "existing-coarse-plus-new-directions-scaled-Ritz",
            "existing_columns": deployed.shape[1],
            "candidate_columns": candidates.shape[1],
            "candidate_rank": Q.shape[1],
            "selected_rank": selected.shape[1],
            "new_directions_observed": self.observed,
            "new_directions_retained": len(self.directions),
            "candidate_bytes": candidates.nbytes,
            "history_bytes_after": selected.nbytes,
        }
        self.indices, self.basis = self.current, selected
        self.directions.clear()
        self.current = None
        return metrics


class CudaCoupledSolver(StudySolver):
    """Keep coupled factors on the GPU across all masks of one nonlinear iterate."""

    def __init__(self, method, **kwargs):
        if method not in {"jacobi", "reference", "recycling"}:
            raise ValueError("Coupled CUDA policies are jacobi, reference and recycling")
        super().__init__(method, device="cpu", **kwargs)
        try:
            import cupy as cp
        except ImportError as error:
            raise RuntimeError("Coupled CUDA solves require the CuPy CUDA extra") from error
        self.cp = cp
        self.device_jacobian, self.cpu_jacobian = None, None
        if self.history is not None:
            self.history = CudaRecycleSpace(cp, self.rank, kwargs.get("window", 20))

    def close(self):
        if self.device_jacobian is not None:
            self.device_jacobian.close()
        self.cp.cuda.get_current_stream().synchronize()
        self.device_jacobian, self.cpu_jacobian = None, None
        super().close()

    def _solve_once(
        self, B, b, indices, initial=None, *, target=None, cap=None, verify_candidates=False
    ):
        cp = self.cp
        timer = PhaseTimer()
        start = timer.start
        target = self.rtol if target is None else target
        cap = self.maxiter if cap is None else cap
        indices = inactive_indices(indices)
        parent = getattr(B, "coupled_parent", None)
        if parent is None or not np.array_equal(indices, B.inactive_indices):
            raise ValueError("A matching restricted coupled Gauss--Newton operator is required")
        b, guess, diagonal = validate_linear_inputs(
            B, b, None, B.diagonal(), initial, target, cap, self.refresh, 1e10
        )
        timer.mark("conversion")

        # Each phase includes its terminal CUDA barrier. No barrier is counted twice.
        def mark(phase):
            cp.cuda.get_current_stream().synchronize()
            timer.mark(phase)

        if self.cpu_jacobian is not parent.jacobian:
            if self.device_jacobian is not None:
                self.device_jacobian.close()
            self.device_jacobian, self.cpu_jacobian = None, None
            self.device_jacobian = CudaControlJacobian(parent.jacobian)
            self.cpu_jacobian = parent.jacobian
        normal = CudaGaussNewton(
            self.device_jacobian,
            parent.weights,
            parent.alpha,
            parent.damping,
            corrections=getattr(parent, "corrections", ()),
        )
        apply = normal.restrict(indices)
        rhs, d, x = cp.asarray(b), cp.asarray(diagonal), cp.asarray(guess)
        Z = (
            cp.asarray(self.reference.restrict(indices), dtype=cp.float64)
            if self.method == "reference"
            else self.history.begin(indices)
            if self.history is not None
            else cp.empty((len(b), 0))
        )
        requested_rank = Z.shape[1]
        mark("upload")
        Z = _orthogonalize(cp, Z)
        mark("basis_processing")
        rank, condition, fallback = Z.shape[1], 1.0, None
        AZ, factor = None, None
        if rank:
            AZ = apply(Z)
            E = Z.T @ AZ
            E = (E + E.T) / 2
            eigenvalues = cp.linalg.eigvalsh(E)
            condition = (
                float(eigenvalues[-1] / eigenvalues[0])
                if float(eigenvalues[0]) > 0
                else float("inf")
            )
            if not np.isfinite(condition) or condition > 1e10:
                Z, AZ, rank, fallback = Z[:, :0], None, 0, "coarse_condition_limit"
            else:
                try:
                    factor = cp.linalg.cholesky(E)
                except cp.linalg.LinAlgError:
                    Z, AZ, rank, fallback = Z[:, :0], None, 0, "coarse_cholesky_failed"
        mark("coarse_or_hierarchy_setup")
        from cupyx.scipy.linalg import solve_triangular

        def coarse_coefficients(z):
            return solve_triangular(factor.T, solve_triangular(factor, z, lower=True), lower=False)

        def correct(z):
            return Z @ coarse_coefficients(Z.T @ z) if rank else cp.zeros_like(z)

        def precondition(r):
            z = r / d
            return z - Z @ coarse_coefficients(AZ.T @ z) if rank else z

        rhs_scale = float(cp.max(cp.abs(rhs)))
        denominator = float(cp.linalg.norm(rhs / rhs_scale)) if rhs_scale else 1.0

        def relative(r):
            return float(cp.linalg.norm(r / rhs_scale if rhs_scale else r)) / denominator

        x += correct(rhs - apply(x))
        r = rhs - apply(x)
        mark("initialization")
        iterations, status, termination = 0, "maxiter", None
        internal_target = target * self.cg_factor
        if relative(r) <= internal_target:
            status = "converged"
        else:
            z = precondition(r)
            p, rz = z.copy(), float(r @ z)
            for k in range(cap):
                if self.history is not None:
                    self.history.capture(p)
                Ap = apply(p)
                curvature = float(p @ Ap)
                if curvature <= 0 or rz <= 0 or not np.isfinite(curvature + rz):
                    status = "breakdown"
                    termination = (
                        "nonfinite_scalar"
                        if not np.isfinite(curvature + rz)
                        else "nonpositive_curvature"
                        if curvature <= 0
                        else "nonpositive_rz"
                    )
                    break
                step = rz / curvature
                x += step * p
                r -= step * Ap
                iterations = k + 1
                restart = iterations % self.refresh == 0 or relative(r) <= internal_target
                if restart:
                    r = rhs - apply(x)
                    if relative(r) <= internal_target:
                        status = "converged"
                        break
                    x += correct(r)
                    r = rhs - apply(x)
                    if relative(r) <= internal_target:
                        status = "converged"
                        break
                z = precondition(r)
                new_rz = float(r @ z)
                p = z.copy() if restart else z + (new_rz / rz) * p
                rz = new_rz
        mark("iteration")
        host = cp.asnumpy(x)
        mark("download")
        residual = independent_residual(B, host, b)
        status = (
            "converged"
            if residual <= target
            else "residual_failed"
            if status == "converged"
            else status
        )
        mark("verification")
        selection = self.history.finish(apply, d, Z, status) if self.history is not None else None
        mark("basis_processing")
        previous = self.previous
        self.previous = indices.copy()
        metrics = timer.finish()
        metrics.update(
            callback_seconds=time.perf_counter() - start,
            restriction_and_transfer_seconds=0.0,
            iteration_rtol=internal_target,
            acceptance_rtol=target,
            termination_test=termination,
            completion=selection,
            input_basis_columns=requested_rank,
            restricted_basis_bytes=Z.nbytes,
            cached_operator_product_bytes=0 if AZ is None else AZ.nbytes,
            derivative_device_bytes=self.device_jacobian.storage_bytes(),
            cuda_pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
            cuda_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
            newly_inactive=None if previous is None else len(np.setdiff1d(indices, previous)),
            newly_active=None if previous is None else len(np.setdiff1d(previous, indices)),
            phase_policy="nonoverlapping wall intervals including terminal CUDA barriers",
        )
        return LinearResult(host, iterations, residual, status, rank, condition, fallback), metrics
