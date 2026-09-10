"""Mass-weighted state elimination for steady and transient mesh problems."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, spsolve

from .validation import positive_real, real_array


class WeightedReducedOperator(LinearOperator):
    """W + alpha*A.T*W*A, with all state-equation rows retained on restriction."""

    def __init__(self, A, weights, alpha, assembled_restriction=True):
        self.A = sparse.csr_matrix(A, dtype=float)
        self.weights = np.asarray(weights, dtype=float).copy()
        self.alpha = positive_real(alpha, "Regularization")
        if self.A.shape[0] != self.A.shape[1] or self.weights.shape != (self.A.shape[0],):
            raise ValueError("A square state operator and matching weights are required")
        if not np.isfinite(self.A.data).all() or not np.isfinite(self.weights).all():
            raise ValueError("State operator and weights must be finite")
        if np.any(self.weights <= 0):
            raise ValueError("Objective weights must be positive")
        self.assembled_restriction = bool(assembled_restriction)
        self._diagonal = self.weights + alpha * np.asarray(
            self.A.power(2).T @ self.weights
        ).ravel()
        super().__init__(dtype=np.dtype(float), shape=self.A.shape)

    def _matvec(self, x):
        return self.weights * x + self.alpha * (self.A.T @ (self.weights * (self.A @ x)))

    def _rmatvec(self, x):
        return self._matvec(x)

    def _matmat(self, x):
        return self.weights[:, None] * x + self.alpha * (
            self.A.T @ (self.weights[:, None] * (self.A @ x))
        )

    def diagonal(self):
        return self._diagonal.copy()

    def assembled(self):
        W = sparse.diags(self.weights)
        return (W + self.alpha * self.A.T @ W @ self.A).tocsr()

    def restrict(self, indices):
        I = np.asarray(indices)
        if I.ndim != 1 or I.dtype.kind not in "iu" or len(np.unique(I)) != len(I):
            raise ValueError("Inactive indices must be distinct integers")
        if len(I) and (I.min() < 0 or I.max() >= self.shape[0]):
            raise ValueError("Inactive indices are out of range")
        AI = self.A[:, I]
        if self.assembled_restriction:
            return (sparse.diags(self.weights[I])
                    + self.alpha * AI.T @ sparse.diags(self.weights) @ AI).tocsr()

        def action(x):
            return (self.weights[I] * x + self.alpha * (AI.T @ (self.weights * (AI @ x))))

        def block(x):
            return (self.weights[I, None] * x
                    + self.alpha * (AI.T @ (self.weights[:, None] * (AI @ x))))

        op = LinearOperator((len(I), len(I)), matvec=action, rmatvec=action,
                            matmat=block, dtype=float)
        op.diagonal = lambda: self._diagonal[I].copy()
        return op


@dataclass
class MeshControlProblem:
    assembly: object
    A: sparse.csr_matrix
    weights: np.ndarray
    forcing: np.ndarray
    alpha: float
    steps: np.ndarray
    initial: np.ndarray
    spatial_A: sparse.csr_matrix
    spatial_forcing: np.ndarray
    nodal_capacity: np.ndarray
    objective_scale: float
    H: WeightedReducedOperator

    @property
    def free(self):
        return self.assembly.mesh.free

    @property
    def spatial_size(self):
        return len(self.free)

    @property
    def size(self):
        return self.A.shape[0]

    def load(self, desired):
        desired = real_array(desired, "Desired temperature")
        if desired.shape != (self.size,) or not np.isfinite(desired).all():
            raise ValueError("Desired temperature must match the complete problem")
        return self.weights * desired + self.alpha * (
            self.A.T @ (self.weights * self.forcing)
        )

    def recover(self, state):
        control = self.A @ state - self.forcing
        adjoint = self.alpha * self.weights * control
        return control, adjoint

    def objective(self, state, desired):
        control, _ = self.recover(state)
        return float(0.5 * self.objective_scale * np.sum(
            self.weights * ((state - desired)**2 + self.alpha * control**2)
        ))

    def forward(self, control):
        """Independent spatial solve or sequential forward substitution."""
        u = np.asarray(control).reshape(-1, self.spatial_size)
        if not len(self.steps):
            return spsolve(self.spatial_A, u[0] + self.spatial_forcing)
        previous = self.initial.copy()
        states = []
        for step, un in zip(self.steps, u, strict=True):
            C = self.nodal_capacity / step
            previous = spsolve(self.spatial_A + sparse.diags(C),
                              un + self.spatial_forcing + C * previous)
            states.append(previous)
        return np.concatenate(states)

    def apply_transpose(self, vector):
        """Independent backward slab action for the trajectory adjoint."""
        if not len(self.steps):
            return self.spatial_A.T @ vector
        v = np.asarray(vector).reshape(len(self.steps), self.spatial_size)
        result = np.empty_like(v)
        for n, dt in enumerate(self.steps):
            result[n] = self.spatial_A.T @ v[n] + self.nodal_capacity * v[n] / dt
            if n + 1 < len(self.steps):
                result[n] -= self.nodal_capacity * v[n + 1] / self.steps[n + 1]
        return result.ravel()


def build_mesh_control(assembly, alpha=0.001, time_steps=None, initial=None,
                       boundary_value=0.0, assembled_restriction=True):
    """Eliminate the control from K*y = M_l*u + f after boundary lifting.

    Coordinates, coefficients, sources, and temperatures must already use the
    units declared by the caller. Capacity is a lumped spatial integral. Steady
    weights are divided by their mean; transient weights additionally contain
    dt/mean(dt). The removed common factor is restored in objective().
    """
    alpha = positive_real(alpha, "Regularization")
    mesh = assembly.mesh
    I, J = mesh.free, mesh.dirichlet
    if not len(I):
        raise ValueError("Optimization requires a degree of freedom outside the Dirichlet boundary")
    mass = assembly.mass[I]
    if np.any(mass <= 0):
        raise ValueError("Lumped mass must be positive")
    boundary = np.broadcast_to(real_array(boundary_value, "Dirichlet temperature"), (len(J),))
    if not np.isfinite(boundary).all():
        raise ValueError("Dirichlet temperature must be finite")
    K = assembly.stiffness
    spatial_A = (sparse.diags(1 / mass) @ K[I][:, I]).tocsr()
    forcing = (assembly.load[I] - K[I][:, J] @ boundary) / mass
    capacity = assembly.capacity[I] / mass
    steps = np.array([], dtype=float) if time_steps is None else real_array(time_steps, "Time steps")
    if steps.ndim != 1 or not np.isfinite(steps).all() or np.any(steps <= 0):
        raise ValueError("Time steps must be a positive finite vector")
    y0 = np.zeros(len(I)) if initial is None else real_array(initial, "Initial temperature")
    if y0.shape != (len(I),) or not np.isfinite(y0).all():
        raise ValueError("Initial temperature must match the spatial degrees of freedom")
    weights = mass / mass.mean()
    A, f, scale = spatial_A, forcing.copy(), float(mass.mean())
    if len(steps):
        T = sparse.diags([1 / steps, -1 / steps[1:]], [0, -1],
                         shape=(len(steps), len(steps)), format="csr")
        A = (sparse.kron(sparse.eye(len(steps)), spatial_A, format="csr")
             + sparse.kron(T, sparse.diags(capacity), format="csr")).tocsr()
        f = np.tile(forcing, len(steps))
        f[:len(I)] += capacity * y0 / steps[0]
        weights = np.kron(steps / steps.mean(), weights)
        scale *= float(steps.mean())
    return MeshControlProblem(
        assembly, A, weights, f, alpha, steps.copy(), y0.copy(), spatial_A,
        forcing.copy(), capacity, scale,
        WeightedReducedOperator(A, weights, alpha, assembled_restriction),
    )
