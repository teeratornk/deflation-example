"""Independent strong-residual quadrature, including radial P1 diffusion."""

import numpy as np
import pytest

from deflation_example.axisymmetric_flow import AxisymmetricFlow
from deflation_example.coupled_derivatives import thermal_velocity_jacobian
from deflation_example.meshes import (
    ThermalMesh,
    assemble_thermal,
    simplex_geometry,
    triangle_quadrature,
)
from test_axisymmetric_flow import annular_rectangle


@pytest.mark.parametrize("axisymmetric", [False, True])
@pytest.mark.parametrize("quadratic", [False, True])
def test_cell_stabilization_is_the_integrated_strong_residual(axisymmetric, quadratic):
    base = annular_rectangle(2)
    mesh = ThermalMesh(base.nodes, base.cells, base.materials, base.dirichlet, axisymmetric)
    vertices = mesh.nodes[mesh.cells]
    grad, lump = simplex_geometry(mesh)
    area = np.linalg.det(vertices[:, 1:] - vertices[:, :1]) / 2
    bary, weights = triangle_quadrature()
    points = np.einsum("qi,eid->eqd", bary, vertices)
    k = np.tile([[0.3, 0.07], [0.07, 0.2]], (len(mesh.cells), 1, 1))
    capacity = np.full(len(mesh.cells), 1.3)

    def field(points):
        r, z = points[..., 0], points[..., 1]
        return np.stack((0.4 + 0.2 * r * z, 0.3 + 0.1 * z**2), axis=-1)

    center = field(vertices.mean(axis=1))
    velocity = center
    if quadratic:
        mids = np.stack(
            [(vertices[:, i] + vertices[:, j]) / 2 for i, j in ((0, 1), (0, 2), (1, 2))], axis=1
        )
        velocity = field(np.concatenate((vertices, mids), axis=1))
    sampled = field(points) if quadratic else np.broadcast_to(center[:, None], points.shape)
    built = assemble_thermal(mesh, k, capacity, velocity, streamline=True, consistent=True)
    h = np.max(np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=3), axis=(1, 2))
    tau = np.minimum(
        h / (2 * capacity * np.linalg.norm(center, axis=1)),
        h**2 / (12 * np.linalg.eigvalsh(k)[:, 0]),
    )
    tau *= built.streamline_limit
    test_weight = tau[:, None] * capacity[:, None] * np.einsum("ed,eid->ei", center, grad)
    measure = area[:, None] * weights
    if axisymmetric:
        measure = measure * 2 * np.pi * points[:, :, 0]
    # A linear field has zero Cartesian Hessian but nonzero cylindrical diffusion.
    state = mesh.nodes[:, 0] + 0.3 * mesh.nodes[:, 1]
    gradient = np.array([1.0, 0.3])
    residual = capacity[:, None] * np.einsum("eqd,d->eq", sampled, gradient)
    if axisymmetric:
        residual -= np.einsum("ed,d->e", k[:, 0, :], gradient)[:, None] / points[:, :, 0]
    local = test_weight * np.sum(measure * residual, axis=1)[:, None]
    expected = np.bincount(mesh.cells.ravel(), weights=local.ravel(), minlength=len(mesh.nodes))
    np.testing.assert_allclose(built.stabilization @ state, expected, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(np.asarray(built.stabilization.sum(axis=0)), 0, atol=2e-14)


@pytest.mark.parametrize("speed", [0.02, 0.3])
def test_weighted_velocity_derivative_includes_radial_and_quadratic_terms(speed):
    mesh = annular_rectangle(3)
    flow = AxisymmetricFlow(mesh, 0.03)
    r, z = flow.points.T
    velocity = speed * np.column_stack((0.4 + r * z, 0.3 + z**2))
    state = mesh.nodes[:, 0] + 0.3 * mesh.nodes[:, 1]
    k = np.tile([[0.03, 0.007], [0.007, 0.02]], (len(mesh.cells), 1, 1))
    capacity = np.full(len(mesh.cells), 1.3)
    scale = 1.7
    rng = np.random.default_rng(441)
    direction = rng.normal(size=velocity.shape)
    derivative = thermal_velocity_jacobian(
        flow, velocity, state, capacity, k, scale, limit_rows=True, residual_weighted=True
    )

    def action(epsilon):
        built = assemble_thermal(
            mesh,
            k,
            capacity,
            scale * flow.thermal_velocity(velocity + epsilon * direction),
            streamline=True,
            consistent=True,
        )
        return built.stiffness @ state

    epsilon = 1e-6
    measured = (action(epsilon) - action(-epsilon)) / (2 * epsilon)
    packed = np.r_[direction[:, 0], direction[:, 1], np.zeros(flow.np)]
    np.testing.assert_allclose(derivative @ packed, measured, rtol=2e-7, atol=2e-9)


def test_weighted_skew_form_requires_an_explicit_reaction_residual():
    mesh = annular_rectangle(2)
    with pytest.raises(ValueError, match="requires advective"):
        assemble_thermal(
            mesh,
            np.tile(np.eye(2), (len(mesh.cells), 1, 1)),
            np.ones(len(mesh.cells)),
            np.zeros((len(mesh.cells), 6, 2)),
            streamline=True,
            consistent=True,
            transport_form="skew",
        )
