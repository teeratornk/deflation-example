"""Exact discrete derivatives for temperature-eliminated buoyant heat transfer.

The momentum Jacobian is indefinite and is solved by sparse LU. Its tangent
and transpose use the same factors. The resulting control Jacobian is used
only in positive-definite Gauss--Newton normal products.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from .meshes import simplex_geometry, triangle_quadrature


class StabilizationBranchError(ValueError):
    """The streamline coefficient is at a nondifferentiable branch switch."""


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


def streamline_parameter(grad, lump, c, kmin, v, vertices, limit_rows):
    """The streamline parameter and its velocity derivative, per fluid cell.

    The parameter is the declared one, the smaller of the advective and diffusive
    limits on the longest edge. At a switch between those branches the derivative is
    undefined and such a point is rejected, which is the guard the symmetric term has
    always carried.

    The consistent weighting additionally bounds the parameter so that no cell takes
    more out of a node's row than that node's own share of the cell mass. Where that
    bound binds, the parameter is the smallest nodal share over the Euclidean norm of
    the cell's streamline derivatives, which no longer depends on the element length;
    its derivative follows from the same expression. A point where the bound is
    exactly at its switch is rejected for the same reason a branch switch is.
    """
    norm = np.linalg.norm(v, axis=1)
    h = np.max(np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=3), axis=(1, 2))
    diffusion_tau = h**2 / (12 * kmin)
    advection_tau = np.full_like(norm, np.inf)
    np.divide(h, 2 * c * norm, out=advection_tau, where=norm > 0)
    if np.any(np.isclose(advection_tau, diffusion_tau, rtol=1e-12, atol=0)):
        raise StabilizationBranchError(
            "Streamline stabilization is at a nondifferentiable branch switch"
        )
    tau = np.minimum(advection_tau, diffusion_tau)
    dtau = np.zeros_like(v)
    advective = advection_tau < diffusion_tau
    dtau[advective] = -tau[advective, None] * v[advective] / norm[advective, None] ** 2
    if not limit_rows:
        return tau, dtau
    # The bound is written with the Euclidean norm of the cell's streamline
    # derivatives rather than their largest entry. Both give a valid bound, because
    # the largest entry never exceeds the norm, but the largest entry is a maximum
    # over three nodes and on a structured mesh with a nearly axial flow two of them
    # tie in whole rows of cells at once, which leaves the model itself without a
    # derivative there. The norm has one, everywhere except at rest, and costs at
    # most a factor of the square root of three in how far the bound reaches.
    share = (lump / lump.sum(axis=1)[:, None]).min(axis=1)
    u = np.einsum("eid,ed->ei", grad, v)
    reach = np.linalg.norm(u, axis=1)
    bound = np.full_like(tau, np.inf)
    np.divide(share, c * reach, out=bound, where=reach > 0)
    if np.any(np.isclose(bound, tau, rtol=1e-12, atol=0)):
        raise StabilizationBranchError("The streamline row limit is at its own switch")
    limited = bound < tau
    moving = reach > 0
    dbound = np.zeros_like(v)
    if np.any(moving):
        gradient = np.einsum("ei,eid->ed", u[moving], grad[moving]) / reach[moving, None]
        dbound[moving] = -bound[moving, None] * gradient / reach[moving, None]
    return np.where(limited, bound, tau), np.where(limited[:, None], dbound, dtau)


def thermal_velocity_jacobian(
    flow, velocity, state, capacity, conductivity, velocity_scale, limit_rows=False
):
    """Derivative of K(v)y, including the active streamline-diffusion branch.

    K is exactly the P2-velocity thermal operator in assemble_thermal. At a
    switch between its two stabilization branches the derivative is undefined;
    such a point is explicitly rejected. At zero velocity the diffusion-limited
    branch is smooth and has zero stabilization derivative.

    ``limit_rows`` follows the assembly's consistent weighting, which bounds the
    streamline parameter by the nodal share of the cell mass. The symmetric term
    carries the same bounded parameter, so this derivative has to know about it.
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
    tau, dtau = streamline_parameter(grad, lump, c, kmin, v, vertices, limit_rows)
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

    def __init__(self, thermal, velocity_actions, buoyancy, factors, history, source_factors=None):
        # With a consistent stabilisation the source is weighted by the streamline
        # test function, so the control is recovered through a factored action
        # rather than a division by the lumped mass. The thermal and velocity blocks
        # are then unnormalised and the action is applied here, which is exact
        # because J = S^-1 R gives J^T = R^T S^-T.
        self.source_factors = None if source_factors is None else tuple(source_factors)
        self.thermal = sparse.csr_matrix(thermal)
        self.velocity_actions = tuple(sparse.csr_matrix(A, copy=True) for A in velocity_actions)
        for action in self.velocity_actions:
            action.eliminate_zeros()
        self.thermal_only = not any(A.nnz for A in self.velocity_actions)
        self.buoyancy = sparse.csr_matrix(buoyancy)
        self.factors, self.history = tuple(factors), tuple(history)
        self.slabs = len(self.factors)
        self.spatial_size = self.buoyancy.shape[1]
        if not self.slabs or self.thermal.shape != (self.slabs * self.spatial_size,) * 2:
            raise ValueError("Control derivative dimensions do not match the trajectory")
        if len(self.velocity_actions) != self.slabs or len(self.history) != self.slabs:
            raise ValueError("Each time slab needs thermal, momentum and history blocks")
        if self.source_factors is not None and len(self.source_factors) != self.slabs:
            raise ValueError("Each time slab needs its own source action factor")
        super().__init__(dtype=np.dtype(float), shape=self.thermal.shape)

    def _matvec(self, direction):
        return self._matmat(np.asarray(direction).reshape(-1, 1)).ravel()

    def _matmat(self, directions):
        columns = directions.shape[1]
        if not columns:
            return np.empty_like(directions)
        if self.thermal_only:
            return self._normalize((self.thermal @ directions).copy())
        dy = np.asarray(directions).reshape(self.slabs, self.spatial_size, columns)
        result = (self.thermal @ directions).reshape(dy.shape)
        previous = np.zeros((self.buoyancy.shape[0], columns))
        for n, factor in enumerate(self.factors):
            previous = factor.solve(self.buoyancy @ dy[n] + self.history[n] @ previous)
            result[n] += self.velocity_actions[n] @ previous
        return self._normalize(result.reshape(self.shape[0], columns))

    def _normalize(self, values):
        """Apply the source action inverse slab by slab, if there is one."""
        if self.source_factors is None:
            return values
        blocks = values.reshape(self.slabs, self.spatial_size, -1)
        for n, factor in enumerate(self.source_factors):
            blocks[n] = factor.solve(blocks[n])
        return blocks.reshape(values.shape)

    def _normalize_transpose(self, values):
        if self.source_factors is None:
            return values
        blocks = np.array(values, dtype=float).reshape(self.slabs, self.spatial_size, -1)
        for n, factor in enumerate(self.source_factors):
            blocks[n] = factor.solve(blocks[n], trans="T")
        return blocks.reshape(np.shape(values))

    def _rmatvec(self, vector):
        return self._rmatmat(np.asarray(vector).reshape(-1, 1)).ravel()

    def _rmatmat(self, vectors):
        columns = vectors.shape[1]
        if not columns:
            return np.empty_like(vectors)
        vectors = self._normalize_transpose(vectors)
        if self.thermal_only:
            return self.thermal.T @ vectors
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


def consistent_velocity_jacobian(flow, velocity, weights, capacity, conductivity, velocity_scale):
    """Velocity derivative of the three residual pieces a consistent weighting adds.

    Each of them has the same shape: the streamline weight on the test function,
    ``w_i = tau * c * (v . grad N_i)``, multiplied by one scalar per cell. So they
    share one derivative and differ only in that scalar, which the caller supplies
    already combined. The scalar is held fixed here because it is built from the
    state, the control and the source, not from the velocity.

    The branch guard lives in the symmetric term's derivative, which the caller
    evaluates first, so a branch switch is rejected before this is reached.
    """
    mesh = flow.mesh
    weights = np.asarray(weights, dtype=float)
    flow.sampled_velocity(velocity)
    grad, lump = simplex_geometry(mesh)
    cells = mesh.cells[flow.fluid_cells]
    grad, lump = grad[flow.fluid_cells], lump[flow.fluid_cells]
    if weights.shape != (len(cells),) or not np.isfinite(weights).all():
        raise ValueError("The consistent derivative needs one finite scalar per fluid cell")
    c = np.asarray(capacity)[flow.fluid_cells]
    kmin = np.linalg.eigvalsh(np.asarray(conductivity)[flow.fluid_cells])[:, 0]
    vertices = mesh.nodes[cells]
    center_shape = np.array([-1, -1, -1, 4, 4, 4]) / 9
    v = velocity_scale * np.einsum("a,ead->ed", center_shape, velocity[flow.p2])
    # The consistent weighting always carries the row limit, so this always asks for
    # it. The symmetric term's derivative, which the caller evaluates first, asks for
    # the same and raises at either switch before this is reached.
    tau, dtau = streamline_parameter(grad, lump, c, kmin, v, vertices, True)
    g = c[:, None] * np.einsum("eid,ed->ei", grad, v)
    # The derivative of the streamline weight that every one of the three shares.
    dw = tau[:, None, None] * c[:, None, None] * grad + dtau[:, None, :] * g[:, :, None]
    local = velocity_scale * (
        (dw * weights[:, None, None])[:, :, None, :] * center_shape[None, None, :, None]
    )
    blocks = [
        flow._matrix(local[:, :, :, d], cells, flow.p2, (len(mesh.nodes), flow.nv))
        for d in range(2)
    ]
    return sparse.hstack(blocks + [sparse.csr_matrix((len(mesh.nodes), flow.np))], format="csr")
