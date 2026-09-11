"""Exact discrete derivatives for temperature-eliminated buoyant heat transfer.

The momentum Jacobian is indefinite and is solved by sparse LU. Its tangent
and transpose use the same factors. The resulting control Jacobian is used
only in positive-definite Gauss--Newton normal products.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from .meshes import simplex_geometry, triangle_quadrature


def buoyancy_jacobian(flow, expansion, temperature_scale, gravity=(0.0, -9.81)):
    """Map nodal dimensionless temperature changes to momentum load changes."""
    local = np.einsum("eq,qi,qj->eij", flow.measure, flow.shape, flow.bary)
    cells = flow.mesh.cells[flow.fluid_cells]
    block = flow._matrix(local, flow.p2, cells, (flow.nv, len(flow.mesh.nodes)))
    return sparse.vstack(
        [-expansion * temperature_scale * float(g) * block for g in gravity]
        + [sparse.csr_matrix((flow.np, len(flow.mesh.nodes)))],
        format="csr",
    )


def thermal_velocity_jacobian(flow, velocity, state, capacity, conductivity, velocity_scale):
    """Derivative of K(v)y, including the active streamline-diffusion branch.

    K is exactly the P2-velocity thermal operator in assemble_thermal. At a
    switch between its two stabilization branches the derivative is undefined;
    such a point is explicitly rejected. At zero velocity the diffusion-limited
    branch is smooth and has zero stabilization derivative.
    """
    mesh = flow.mesh
    state = np.asarray(state, dtype=float)
    if state.shape != (len(mesh.nodes),) or not np.isfinite(state).all():
        raise ValueError("Thermal derivative requires a finite full-domain state")
    flow.sampled_velocity(velocity)
    grad, lump = simplex_geometry(mesh)
    cells = mesh.cells[flow.fluid_cells]
    grad, lump = grad[flow.fluid_cells], lump[flow.fluid_cells]
    c = np.asarray(capacity)[flow.fluid_cells]
    kmin = np.linalg.eigvalsh(np.asarray(conductivity)[flow.fluid_cells])[:, 0]
    bary, weights = triangle_quadrature()
    shape = np.column_stack(
        (
            bary * (2 * bary - 1),
            4 * bary[:, 0] * bary[:, 1],
            4 * bary[:, 0] * bary[:, 2],
            4 * bary[:, 1] * bary[:, 2],
        )
    )
    vertices = mesh.nodes[cells]
    area = np.linalg.det(vertices[:, 1:] - vertices[:, :1]) / 2
    measure = area[:, None] * weights * 2 * np.pi * (vertices[:, :, 0] @ bary.T)
    gradient_y = np.einsum("ei,eid->ed", state[cells], grad)
    # Derivative of capacity * integral N_i v . grad(y).
    moments = np.einsum("eq,qi,qa->eia", measure, bary, shape)
    transport = c[:, None, None, None] * moments[:, :, :, None] * gradient_y[:, None, None, :]
    center_shape = np.array([-1, -1, -1, 4, 4, 4]) / 9
    v = velocity_scale * np.einsum("a,ead->ed", center_shape, velocity[flow.p2])
    norm = np.linalg.norm(v, axis=1)
    h = np.max(np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=3), axis=(1, 2))
    diffusion_tau = h**2 / (12 * kmin)
    advection_tau = np.full_like(norm, np.inf)
    np.divide(h, 2 * c * norm, out=advection_tau, where=norm > 0)
    if np.any(np.isclose(advection_tau, diffusion_tau, rtol=1e-12, atol=0)):
        raise ValueError("Streamline stabilization is at a nondifferentiable branch switch")
    tau = np.minimum(advection_tau, diffusion_tau)
    dtau = np.zeros_like(v)
    advective = advection_tau < diffusion_tau
    dtau[advective] = -tau[advective, None] * v[advective] / norm[advective, None] ** 2
    g = c[:, None] * np.einsum("eid,ed->ei", grad, v)
    s = c * np.einsum("ed,ed->e", gradient_y, v)
    derivative = lump.sum(axis=1)[:, None, None] * (
        tau[:, None, None] * c[:, None, None] * grad * s[:, None, None]
        + tau[:, None, None] * g[:, :, None] * c[:, None, None] * gradient_y[:, None, :]
        + dtau[:, None, :] * g[:, :, None] * s[:, None, None]
    )
    local = velocity_scale * (
        transport + derivative[:, :, None, :] * center_shape[None, None, :, None]
    )
    blocks = [
        flow._matrix(local[:, :, :, d], cells, flow.p2, (len(mesh.nodes), flow.nv))
        for d in range(2)
    ]
    return sparse.hstack(blocks + [sparse.csr_matrix((len(mesh.nodes), flow.np))], format="csr")


class ControlJacobian(LinearOperator):
    """J_U = T + S F_v^{-1} L with coupled lower-triangular momentum blocks.

    T contains all thermal time blocks. The momentum history matrices C satisfy
    F_v[n] dv[n] = L dy[n] + C[n] dv[n-1]. Transpose application traverses time
    backwards, including the final block. All LU factors are fixed at creation.
    """

    def __init__(self, thermal, velocity_actions, buoyancy, factors, history):
        self.thermal = sparse.csr_matrix(thermal)
        self.velocity_actions = tuple(velocity_actions)
        self.buoyancy = sparse.csr_matrix(buoyancy)
        self.factors, self.history = tuple(factors), tuple(history)
        self.slabs = len(self.factors)
        self.spatial_size = self.buoyancy.shape[1]
        if not self.slabs or self.thermal.shape != (self.slabs * self.spatial_size,) * 2:
            raise ValueError("Control derivative dimensions do not match the trajectory")
        if len(self.velocity_actions) != self.slabs or len(self.history) != self.slabs:
            raise ValueError("Each time slab needs thermal, momentum and history blocks")
        super().__init__(dtype=np.dtype(float), shape=self.thermal.shape)

    def _matvec(self, direction):
        return self._matmat(np.asarray(direction).reshape(-1, 1)).ravel()

    def _matmat(self, directions):
        columns = directions.shape[1]
        if not columns:
            return np.empty_like(directions)
        dy = np.asarray(directions).reshape(self.slabs, self.spatial_size, columns)
        result = (self.thermal @ directions).reshape(dy.shape)
        previous = np.zeros((self.buoyancy.shape[0], columns))
        for n, factor in enumerate(self.factors):
            previous = factor.solve(self.buoyancy @ dy[n] + self.history[n] @ previous)
            result[n] += self.velocity_actions[n] @ previous
        return result.reshape(self.shape[0], columns)

    def _rmatvec(self, vector):
        return self._rmatmat(np.asarray(vector).reshape(-1, 1)).ravel()

    def _rmatmat(self, vectors):
        columns = vectors.shape[1]
        if not columns:
            return np.empty_like(vectors)
        z = np.asarray(vectors).reshape(self.slabs, self.spatial_size, columns)
        result = (self.thermal.T @ vectors).reshape(z.shape)
        following = np.zeros((self.buoyancy.shape[0], columns))
        for n in range(self.slabs - 1, -1, -1):
            rhs = self.velocity_actions[n].T @ z[n]
            if n + 1 < self.slabs:
                rhs = rhs + self.history[n + 1].T @ following
            following = self.factors[n].solve(rhs, trans="T")
            result[n] += self.buoyancy.T @ following
        return result.reshape(self.shape[0], columns)


class GaussNewtonOperator(LinearOperator):
    """W + alpha J_U.T W J_U + damping W with positive tracking weights."""

    def __init__(self, jacobian, weights, alpha, damping=0.0):
        self.jacobian = jacobian
        self.weights = np.asarray(weights, dtype=float).copy()
        self.alpha, self.damping = float(alpha), float(damping)
        if (
            jacobian.shape != (len(self.weights),) * 2
            or not np.isfinite(self.weights).all()
            or np.any(self.weights <= 0)
            or not np.isfinite(alpha)
            or alpha <= 0
            or not np.isfinite(damping)
            or damping < 0
        ):
            raise ValueError("Gauss--Newton requires positive weights and regularization")
        super().__init__(dtype=np.dtype(float), shape=jacobian.shape)

    def _matvec(self, x):
        x = np.asarray(x).reshape(-1)
        return (1 + self.damping) * self.weights * x + self.alpha * (
            self.jacobian.T @ (self.weights * (self.jacobian @ x))
        )

    def _rmatvec(self, x):
        return self._matvec(x)

    def _matmat(self, x):
        return (1 + self.damping) * self.weights[:, None] * x + self.alpha * (
            self.jacobian.T @ (self.weights[:, None] * (self.jacobian @ x))
        )

    def _rmatmat(self, x):
        return self._matmat(x)

    def restrict(self, indices):
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or len(np.unique(indices)) != len(indices)
            or (len(indices) and (indices.min() < 0 or indices.max() >= self.shape[0]))
        ):
            raise ValueError("Inactive indices must be distinct and in range")

        def action(x):
            full = np.zeros(self.shape[0])
            full[indices] = np.asarray(x).ravel()
            return (self @ full)[indices]

        def block(x):
            full = np.zeros((self.shape[0], x.shape[1]))
            full[indices] = x
            return (self @ full)[indices]

        operator = LinearOperator(
            (len(indices),) * 2,
            matvec=action,
            rmatvec=action,
            matmat=block,
            rmatmat=block,
            dtype=float,
        )
        operator.coupled_parent = self
        operator.inactive_indices = indices.copy()
        return operator
