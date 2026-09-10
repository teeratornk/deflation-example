"""Analytic and conservation checks of the axisymmetric momentum discretization."""

import numpy as np
import pytest
from math import factorial

from deflation_example.axisymmetric_flow import AxisymmetricFlow, flow_quadrature
from deflation_example.meshes import ThermalMesh


def annular_rectangle(n):
    r, z = np.meshgrid(np.linspace(1, 2, n + 1), np.linspace(0, 1, n + 1))
    nodes = np.column_stack((r.ravel(), z.ravel()))
    a = (np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]).ravel()
    cells = np.vstack(
        (np.column_stack((a, a + 1, a + n + 2)), np.column_stack((a, a + n + 2, a + n + 1)))
    )
    return ThermalMesh(nodes, cells, np.zeros(len(cells), dtype=int), np.arange(n + 1), True)


def test_axisymmetric_convection_quadrature_integrates_polynomial_products():
    bary, weights = flow_quadrature()
    assert np.all(weights > 0)
    for a in range(9):
        for b in range(9 - a):
            for c in range(9 - a - b):
                expected = 2 * factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 2)
                actual = weights @ (bary[:, 0] ** a * bary[:, 1] ** b * bary[:, 2] ** c)
                assert actual == pytest.approx(expected, rel=1e-13, abs=1e-16)


@pytest.mark.parametrize("convection_form", ["advective", "skew"])
def test_discrete_kinetic_identity_includes_pointwise_divergence(convection_form):
    flow = AxisymmetricFlow(annular_rectangle(3), 0.1, convection_form=convection_form)
    random = np.random.default_rng(27).normal(size=(flow.nv, 2)) * 0.1
    diagnostics = flow.kinetic_diagnostics(random)
    assert abs(diagnostics["integration_identity_defect"]) < 1e-12
    assert diagnostics["pointwise_divergence_relative_L2"] > 0.1
    r, z = flow.points.T
    divergence_free = np.column_stack((0.02 * r * z, -0.02 * z * z))
    diagnostics = flow.kinetic_diagnostics(divergence_free)
    assert diagnostics["pointwise_divergence_relative_L2"] < 1e-13
    assert abs(diagnostics["divergence_energy_term"]) < 1e-15


def test_grad_div_form_matches_the_divergence_integral():
    flow = AxisymmetricFlow(annular_rectangle(3), 0.1, grad_div=0.5)
    velocity = np.random.default_rng(37).normal(size=(flow.nv, 2)) * 0.1
    x = np.r_[velocity[:, 0], velocity[:, 1], np.zeros(flow.np)]
    divergence = np.einsum("eia,eqia->eq", velocity[flow.p2], flow.div_basis)
    expected = np.sum(flow.measure * divergence**2)
    assert x @ (flow.grad_div_operator @ x) == pytest.approx(expected, rel=1e-13)
    assert expected > 0
    r, z = flow.points.T
    exact = np.r_[0.02 * r * z, -0.02 * z * z, np.zeros(flow.np)]
    assert np.max(np.abs(flow.grad_div_operator @ exact)) < 1e-14


@pytest.mark.parametrize("convection_form", ["advective", "skew"])
def test_matrix_free_force_and_quadratic_line_search_identity(convection_form):
    flow = AxisymmetricFlow(
        annular_rectangle(3), 0.03, convection_form=convection_form, grad_div=0.2
    )
    rng = np.random.default_rng(123)
    x, step, rhs = (rng.normal(size=flow.size) * 0.1 for _ in range(3))

    def velocity(a):
        return np.column_stack((a[: flow.nv], a[flow.nv : 2 * flow.nv]))

    v, dv = velocity(x), velocity(step)
    np.testing.assert_allclose(
        flow.nonlinear_force(v), flow.convection(v) @ v, rtol=1e-12, atol=1e-14
    )
    A = flow.operator(v, time_step=0.2)
    J = A + flow.convection_derivative(v)
    nonlinear = flow.nonlinear_force(dv)
    quadratic = np.r_[nonlinear[:, 0], nonlinear[:, 1], np.zeros(flow.np)]
    for damping in (1.0, 0.1, 1e-5):
        trial = x + damping * step
        actual = flow.operator(velocity(trial), time_step=0.2) @ trial - rhs
        predicted = A @ x - rhs + damping * (J @ step) + damping * damping * quadratic
        np.testing.assert_allclose(actual, predicted, rtol=1e-12, atol=1e-13)


def test_p2_partition_affine_gradient_and_cylindrical_divergence():
    flow = AxisymmetricFlow(annular_rectangle(3), 0.1)
    np.testing.assert_allclose(flow.shape.sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(flow.grad.sum(axis=2), 0, atol=1e-13)
    r, z = flow.points.T
    velocity = np.column_stack((r, -2 * z))
    Br, Bz = flow.divergence
    np.testing.assert_allclose(Br @ r + Bz @ (-2 * z), 0, atol=2e-14)
    np.testing.assert_allclose(flow.boundary_flux(velocity).sum(), 0, atol=2e-14)
    # The radial hoop term is needed for the vector Laplacian of v_r=r to vanish.
    interior = np.setdiff1d(np.arange(flow.nv), flow.boundary)
    np.testing.assert_allclose(((flow.diffusion + flow.hoop) @ r)[interior], 0, atol=5e-13)


@pytest.mark.parametrize("method", ["picard", "newton"])
@pytest.mark.parametrize("convection_form", ["advective", "skew"])
def test_nonlinear_manufactured_affine_flow_with_buoyancy(method, convection_form):
    flow = AxisymmetricFlow(annular_rectangle(4), 0.3, convection_form=convection_form)
    r, z = flow.points.T
    exact = np.column_stack((0.02 * r, -0.04 * z))
    rq, zq = flow.quadrature_points.transpose(2, 0, 1)
    # Exact pressure = 0.03*r + 0.07*z. Vector Laplacian vanishes.
    force = np.stack((0.02**2 * rq + 0.03, 0.04**2 * zq + 0.07), axis=2)
    temperature = 300 + 0.5 * flow.mesh.nodes[:, 1]
    buoyancy = flow.buoyancy(temperature, 300, 0.001)
    prescribed_body_force = force - buoyancy
    result = flow.solve(
        prescribed_body_force + buoyancy,
        flow.boundary,
        exact[flow.boundary],
        pressure_gauge=(0, 0.03),
        tolerance=1e-10,
        method=method,
    )
    assert result.status == "converged"
    assert len(result.history) > 1
    np.testing.assert_allclose(result.velocity, exact, atol=1e-10)
    exact_pressure = (
        0.03 * flow.mesh.nodes[flow.vertices, 0] + 0.07 * flow.mesh.nodes[flow.vertices, 1]
    )
    np.testing.assert_allclose(result.pressure, exact_pressure, atol=1e-9)


@pytest.mark.parametrize("convection_form", ["advective", "skew"])
@pytest.mark.parametrize("grad_div", [0.0, 10.0])
def test_full_newton_jacobian_matches_directional_difference(convection_form, grad_div):
    flow = AxisymmetricFlow(
        annular_rectangle(3), 0.03, convection_form=convection_form, grad_div=grad_div
    )
    rng = np.random.default_rng(18)
    x = rng.normal(size=flow.size) * 0.01
    direction = rng.normal(size=flow.size)

    def residual(y):
        v = np.column_stack((y[: flow.nv], y[flow.nv : 2 * flow.nv]))
        return flow.operator(v, time_step=0.2) @ y

    velocity = np.column_stack((x[: flow.nv], x[flow.nv : 2 * flow.nv]))
    J = flow.operator(velocity, time_step=0.2) + flow.convection_derivative(velocity)
    h = 1e-6
    numerical = (residual(x + h * direction) - residual(x - h * direction)) / (2 * h)
    np.testing.assert_allclose(J @ direction, numerical, rtol=1e-8, atol=2e-10)


def test_poiseuille_annulus_refinement():
    errors = []
    for n in (3, 6, 12):
        flow = AxisymmetricFlow(annular_rectangle(n), 0.1)
        r = flow.points[:, 0]
        axial = (4 - r * r - 3 * np.log(2 / r) / np.log(2)) / 4
        exact = np.column_stack((np.zeros(len(r)), axial))
        result = flow.solve(
            np.zeros_like(flow.quadrature_points),
            flow.boundary,
            exact[flow.boundary],
            pressure_gauge=(0, 0.0),
            convection=False,
        )
        assert result.status == "converged"
        rq = flow.quadrature_points[:, :, 0]
        exact_q = (4 - rq * rq - 3 * np.log(2 / rq) / np.log(2)) / 4
        difference = flow.sampled_velocity(result.velocity)[:, :, 1] - exact_q
        errors.append(np.sqrt(np.sum(flow.measure * difference**2)))
        assert abs(flow.boundary_flux(result.velocity).sum()) < 1e-12
        # Pressure solves a separate block with the correct gradient and sign.
        pressure = -0.1 * flow.mesh.nodes[flow.vertices, 1]
        assert np.max(np.abs(result.pressure - pressure)) < 0.003
    assert errors[0] / errors[1] > 5
    assert errors[1] / errors[2] > 5


def test_transient_momentum_includes_previous_velocity_and_mass():
    flow = AxisymmetricFlow(annular_rectangle(3), 0.1)
    previous = np.column_stack((np.zeros(flow.nv), np.ones(flow.nv) * 0.2))
    exact = previous + [0, 0.03]
    force = np.zeros_like(flow.quadrature_points)
    force[:, :, 1] = 0.03 / 0.5
    result = flow.solve(
        force,
        flow.boundary,
        exact[flow.boundary],
        previous=previous,
        time_step=0.5,
        pressure_gauge=(0, 0),
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.velocity, exact, atol=1e-12)
    np.testing.assert_allclose(result.pressure, 0, atol=1e-12)


def test_buoyancy_sign_and_disabled_feedback():
    flow = AxisymmetricFlow(annular_rectangle(2), 0.1)
    temp = np.full(len(flow.mesh.nodes), 310.0)
    force = flow.buoyancy(temp, 300, 0.001)
    np.testing.assert_allclose(force[:, :, 0], 0)
    np.testing.assert_allclose(force[:, :, 1], 0.0981)
    np.testing.assert_array_equal(flow.buoyancy(temp, 300, 0), 0)
    with pytest.raises(ValueError):
        flow.solve(force, flow.boundary, np.zeros((len(flow.boundary), 2)), time_step=1)


def test_natural_outlet_preserves_mass_without_pressure_pinning():
    flow = AxisymmetricFlow(annular_rectangle(6), 0.1)
    r, z = flow.points.T
    # A unidirectional inlet profile with no slip on the cylindrical walls.
    axial = (4 - r * r - 3 * np.log(2 / r) / np.log(2)) / 4
    exact = np.column_stack((np.zeros(len(r)), axial))
    fixed = flow.boundary[
        (z[flow.boundary] < 1 - 1e-12)
        | np.isclose(r[flow.boundary], 1)
        | np.isclose(r[flow.boundary], 2)
    ]
    result = flow.solve(
        np.zeros_like(flow.quadrature_points),
        fixed,
        exact[fixed],
        convection=False,
    )
    assert result.status == "converged"
    flux = flow.boundary_flux(result.velocity)
    assert abs(flux.sum()) / np.abs(flux).sum() < 1e-12
    assert np.max(np.abs(result.velocity - exact)) < 0.001


def test_iteration_cap_remains_a_failure():
    flow = AxisymmetricFlow(annular_rectangle(3), 0.1)
    r, z = flow.points.T
    boundary = np.column_stack((0.1 * r, -0.2 * z))[flow.boundary]
    rq, zq = flow.quadrature_points.transpose(2, 0, 1)
    force = np.stack((0.01 * rq, 0.04 * zq), axis=2)
    result = flow.solve(
        force,
        flow.boundary,
        boundary,
        pressure_gauge=(0, 0),
        max_iterations=1,
        tolerance=1e-12,
    )
    assert result.status == "iteration_cap"
    assert len(result.history) == 1
    assert result.history[-1]["momentum_relative_residual"] > 1e-12


@pytest.mark.parametrize("strategy", ["steady_initialization", "pseudo_time_initialization"])
def test_continuation_restores_physical_viscosity_and_checks_final_equation(strategy):
    flow = AxisymmetricFlow(annular_rectangle(3), 0.1)
    r, z = flow.points.T
    exact = np.column_stack((0.02 * r, -0.04 * z))
    # An open upper boundary determines pressure; use a zero-flow case to
    # verify continuation and immutable physical coefficients independently.
    fixed = flow.boundary[z[flow.boundary] < 1 - 1e-12]
    result = getattr(flow, strategy)(
        np.zeros_like(flow.quadrature_points), fixed, np.zeros_like(exact[fixed])
    )
    assert result.status == "converged"
    assert flow.viscosity == 0.1
    if strategy == "steady_initialization":
        assert result.history[-1]["viscosity_factor"] == 1
    assert (
        max(
            flow.verify(
                result, np.zeros_like(flow.quadrature_points), fixed, np.zeros_like(exact[fixed])
            ).values()
        )
        <= 1e-9
    )
