"""Verify the thermal transport energy identity independently on small meshes."""

import numpy as np
import pytest

from deflation_example.mesh_control import build_mesh_control
from deflation_example.mesh_showcases import build_showcase
from deflation_example.meshes import ThermalMesh, assemble_thermal
from deflation_example.temporal_stability import amplification_mode


def rectangle(axisymmetric):
    return ThermalMesh(
        np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
        np.zeros(2, dtype=int),
        np.array([0, 1]),
        axisymmetric,
    )


def velocity(points):
    r, z = points[..., 0], points[..., 1]
    return np.stack((r**2 + 0.3 * r * z, -r * z + 0.5 * z**2), axis=-1)


def inputs(mesh, field=velocity):
    x = mesh.nodes[mesh.cells]
    points = np.concatenate(
        (x, (x[:, :1] + x[:, 1:2]) / 2, (x[:, :1] + x[:, 2:3]) / 2, (x[:, 1:2] + x[:, 2:3]) / 2),
        axis=1,
    )
    return np.tile(np.eye(2), (2, 1, 1)), np.full(2, 1.2), field(points)


@pytest.mark.parametrize("axisymmetric", [False, True])
def test_skew_transport_equals_boundary_energy_for_continuous_p2_flow(axisymmetric):
    mesh = rectangle(axisymmetric)
    k, capacity, v = inputs(mesh)
    adv = assemble_thermal(mesh, k, capacity, v)
    skew = assemble_thermal(mesh, k, capacity, v, transport_form="skew")
    state = np.array([0.3, -0.8, 1.2, 0.1])
    # Independent boundary integration uses analytic velocity and three-point
    # edge quadrature. Counterclockwise boundary edges give outward normals.
    q, weights = np.polynomial.legendre.leggauss(3)
    q, weights = (q + 1) / 2, weights / 2
    boundary = 0.0
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 0)):
        edge = mesh.nodes[right] - mesh.nodes[left]
        normal = np.array([edge[1], -edge[0]])
        points = (1 - q[:, None]) * mesh.nodes[left] + q[:, None] * mesh.nodes[right]
        temperature = (1 - q) * state[left] + q * state[right]
        measure = 2 * np.pi * points[:, 0] if axisymmetric else np.ones(3)
        boundary += 1.2 * np.sum(weights * measure * (velocity(points) @ normal) * temperature**2)
    np.testing.assert_allclose(state @ (skew.transport @ state), boundary / 2, rtol=2e-13)
    assert abs(state @ (adv.transport @ state) - boundary / 2) > 0.1
    correction = skew.transport - adv.transport
    np.testing.assert_allclose(correction.toarray(), correction.T.toarray(), atol=1e-13)
    for name in ("mass", "capacity", "load"):
        np.testing.assert_array_equal(getattr(adv, name), getattr(skew, name))
    for name in ("diffusion", "stabilization"):
        assert (getattr(adv, name) != getattr(skew, name)).nnz == 0


@pytest.mark.parametrize("axisymmetric", [False, True])
def test_divergence_free_axial_velocity_preserves_original_transport(axisymmetric):
    mesh = rectangle(axisymmetric)
    k, capacity, v = inputs(mesh, lambda x: np.broadcast_to([0.0, 1.0], x.shape))
    original = assemble_thermal(mesh, k, capacity, v, streamline=True)
    explicit = assemble_thermal(mesh, k, capacity, v, streamline=True, transport_form="advective")
    skew = assemble_thermal(mesh, k, capacity, v, streamline=True, transport_form="skew")
    assert (original.stiffness != explicit.stiffness).nnz == 0
    np.testing.assert_allclose(original.stiffness.toarray(), skew.stiffness.toarray(), atol=1e-13)


def test_corrected_transport_preserves_coupled_time_blocks_and_recovery():
    mesh = rectangle(True)
    k, capacity, v = inputs(mesh)
    a = assemble_thermal(mesh, k, capacity, v, transport_form="skew")
    p = build_mesh_control(a, time_steps=[0.02, 0.04, 0.01], initial=np.array([0.2, 0.1]))
    state = np.linspace(0.1, 0.5, p.size)
    np.testing.assert_allclose(p.forward(p.recover(state)[0]), state, atol=1e-13)
    np.testing.assert_allclose(p.apply_transpose(state), p.A.T @ state, atol=1e-12)


def test_skew_pilot_rejects_unsupported_velocity_and_geometry():
    mesh = rectangle(False)
    k, capacity, _ = inputs(mesh)
    with pytest.raises(ValueError, match="quadratic 2D"):
        assemble_thermal(mesh, k, capacity, transport_form="skew")
    with pytest.raises(ValueError, match="quadratic transformer"):
        build_showcase("engine_3d", transport_form="skew")


def test_computed_modes_verify_energy_and_diffusion_decay():
    mesh = rectangle(True)
    k, capacity, v = inputs(mesh, lambda x: np.zeros_like(x))
    a = assemble_thermal(mesh, k, capacity, v, transport_form="skew")
    result = amplification_mode(a, 0.05)
    assert result["largest_computed_amplification_modulus"] < 1
    assert result["relative_eigenpair_residual"] < 1e-12
    assert result["energy_identity_absolute_error"] < 1e-12


def test_growing_mode_is_reported_even_with_an_accurate_eigenpair():
    from scipy import sparse

    mesh = rectangle(True)
    k, capacity, v = inputs(mesh, lambda x: np.zeros_like(x))
    a = assemble_thermal(mesh, k, capacity, v)
    a.transport = -2 * a.diffusion
    result = amplification_mode(a, 0.005)
    assert result["largest_computed_amplification_modulus"] > 1
    assert result["spatial_eigenvalue_real_per_second"] < 0
    assert result["relative_eigenpair_residual"] < 1e-12
    assert sparse.issparse(a.stiffness)
