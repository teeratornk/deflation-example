"""Complete backward-Euler trajectory optimization for the steady CHT geometry.

Unknowns are ordered by time slab, then by the existing spatial C-order nodes.
Both objective terms use right-endpoint quadrature. The common spatial volume
times mean time step is removed from the reduced objective and multipliers.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .benchmark_extended import target_values
from .problems import Problem, build_problem
from .solvers import relative_norm
from .validation import integer, positive_real, real_array


@dataclass
class ThermalTrajectory:
    spatial: Problem
    steps: np.ndarray
    capacity: np.ndarray
    initial: np.ndarray
    time_matrix: sparse.csr_matrix
    A: sparse.csr_matrix
    H: sparse.csr_matrix
    weights: np.ndarray
    forcing: np.ndarray
    objective_scale: float

    @property
    def spatial_size(self):
        return self.spatial.A.shape[0]

    @property
    def size(self):
        return self.A.shape[0]

    @property
    def times(self):
        return np.cumsum(self.steps)

    def apply_linear(self, state):
        """Independent slabwise linear action, with zero initial contribution."""
        y = real_array(state, "Trajectory state").reshape(len(self.steps), self.spatial_size)
        out = np.empty_like(y)
        for n, dt in enumerate(self.steps):
            out[n] = self.spatial.A @ y[n] + self.capacity * y[n] / dt
            if n:
                out[n] -= self.capacity * y[n - 1] / dt
        return out.ravel()

    def apply_transpose(self, adjoint):
        """Independent backward action, including the final block."""
        p = real_array(adjoint, "Trajectory adjoint").reshape(len(self.steps), self.spatial_size)
        out = np.empty_like(p)
        for n, dt in enumerate(self.steps):
            out[n] = self.spatial.A.T @ p[n] + self.capacity * p[n] / dt
            if n + 1 < len(self.steps):
                out[n] -= self.capacity * p[n + 1] / self.steps[n + 1]
        return out.ravel()

    def forward(self, control):
        """Independent block forward substitution for verification."""
        control = real_array(control, "Trajectory control").reshape(
            len(self.steps), self.spatial_size
        )
        y = np.empty_like(control)
        previous, factors = self.initial, {}
        for n, dt in enumerate(self.steps):
            if dt not in factors:
                factors[dt] = splu((self.spatial.A + sparse.diags(self.capacity / dt)).tocsc())
            y[n] = factors[dt].solve(control[n] + self.capacity * previous / dt)
            previous = y[n]
        return y.ravel()

    def target(self, parameters):
        """Moving and pulsed targets; one query specifies the entire trajectory."""
        fields = []
        for physical_time in self.times:
            fraction = physical_time / self.times[-1]
            centers = np.array(parameters["centers"], dtype=float)
            angle = 0.8 * fraction
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            centers[:, :2] = (centers[:, :2] - 0.5) @ rotation.T + 0.5
            centers[:, 2] += 0.05 * np.sin(2 * np.pi * fraction)
            pulse = 0.15 + 0.85 * np.sin(np.pi * fraction) ** 2
            temporal = {
                **parameters,
                "centers": centers.tolist(),
                "weights": [pulse * value for value in parameters["weights"]],
            }
            fields.append(target_values(self.spatial.coordinates, temporal))
        return np.concatenate(fields)

    def load(self, desired):
        return self.weights * desired + self.spatial.alpha * (
            self.A.T @ (self.weights * self.forcing)
        )

    def recover(self, state):
        control = self.A @ state - self.forcing
        # Multiplier of the unweighted assembled equation in the normalized
        # discrete Lagrangian. Uniform steps give adjoint = alpha * control.
        adjoint = self.spatial.alpha * self.weights * control
        return control, adjoint

    def objective(self, state, desired):
        control, _ = self.recover(state)
        return float(
            0.5
            * self.objective_scale
            * np.sum(self.weights * ((state - desired) ** 2 + self.spatial.alpha * control**2))
        )

    def recovery_metrics(self, state, desired, multiplier):
        control, adjoint = self.recover(state)
        state_residual = self.apply_linear(state) - self.forcing - control
        adjoint_residual = (
            self.weights * (state - desired) + self.apply_transpose(adjoint) + multiplier
        )
        scale = max(1.0, float(np.linalg.norm(self.load(desired), np.inf)))
        return {
            "state_relative": relative_norm(state_residual, control + self.forcing),
            "adjoint_normalized_max": float(np.linalg.norm(adjoint_residual, np.inf) / scale),
            "control_normalized_max": float(
                np.linalg.norm(self.spatial.alpha * self.weights * control - adjoint, np.inf)
                / scale
            ),
        }


def build_transient_cht(
    n=8,
    slabs=8,
    horizon=0.1,
    alpha=0.001,
    solid_capacity=2.0,
    fluid_capacity=1.0,
    initial=None,
    time_steps=None,
):
    """Assemble the coupled trajectory reduction, including nonzero initial data."""
    n, slabs = integer(n, "Spatial grid", 2), integer(slabs, "Time slabs", 1)
    horizon = positive_real(horizon, "Physical horizon")
    solid_capacity = positive_real(solid_capacity, "Solid thermal capacity")
    fluid_capacity = positive_real(fluid_capacity, "Fluid thermal capacity")
    spatial = build_problem("cht", n, alpha)
    N = n**3
    steps = (
        np.full(slabs, horizon / slabs)
        if time_steps is None
        else real_array(time_steps, "Time steps")
    )
    if (
        steps.shape != (slabs,)
        or not np.isfinite(steps).all()
        or np.any(steps <= 0)
        or not np.isclose(steps.sum(), horizon, rtol=1e-12, atol=0)
    ):
        raise ValueError("Positive time steps must match the slab count and physical horizon")
    steps = steps.copy()
    initial = np.zeros(N) if initial is None else real_array(initial, "Initial temperature")
    if initial.shape != (N,) or not np.isfinite(initial).all():
        raise ValueError("Initial temperature must be finite and match the spatial grid")
    initial = initial.copy()
    capacity = np.where(spatial.coordinates[2] < 0.5, solid_capacity, fluid_capacity)
    temporal = sparse.diags(
        [1 / steps, -1 / steps[1:]], [0, -1], shape=(slabs, slabs), format="csr"
    )
    A = (
        sparse.kron(sparse.eye(slabs), spatial.A, format="csr")
        + sparse.kron(temporal, sparse.diags(capacity), format="csr")
    ).tocsr()
    weights = np.repeat(steps / steps.mean(), N)
    W = sparse.diags(weights)
    H = (W + alpha * (A.T @ W @ A)).tocsr()
    forcing = np.zeros(slabs * N)
    forcing[:N] = capacity * initial / steps[0]
    return ThermalTrajectory(
        spatial,
        steps,
        capacity,
        initial,
        temporal,
        A,
        H,
        weights,
        forcing,
        float(steps.mean() / (n + 1) ** 3),
    )
