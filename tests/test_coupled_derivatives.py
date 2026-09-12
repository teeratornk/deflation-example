"""Discrete tangent, transpose and Gauss--Newton verification."""

import numpy as np
import pytest

from deflation_example.axisymmetric_flow import AxisymmetricFlow
from deflation_example.coupled_control import CoupledControlProblem, FlowEvaluationError
from deflation_example.coupled_derivatives import (
    ControlJacobian,
    GaussNewtonOperator,
    buoyancy_jacobian,
    thermal_velocity_jacobian,
)
from test_axisymmetric_flow import annular_rectangle


def test_zero_velocity_sensitivity_never_invokes_momentum_tangent_solves():
    from scipy import sparse

    class ForbiddenSolve:
        def solve(self, *args, **kwargs):
            raise AssertionError("Zero thermal sensitivity must bypass momentum solves")

    T = sparse.diags([np.full(3, -0.4), np.arange(1, 5)], [-1, 0], shape=(4, 4), format="csr")
    zero = sparse.csr_matrix((2, 3))
    J = ControlJacobian(
        T,
        [zero, zero],
        sparse.csr_matrix(np.ones((3, 2))),
        [ForbiddenSolve(), ForbiddenSolve()],
        [sparse.eye(3), sparse.eye(3)],
    )
    vectors = np.arange(12, dtype=float).reshape(4, 3)
    np.testing.assert_array_equal(J @ vectors, T @ vectors)
    np.testing.assert_array_equal(J.T @ vectors, T.T @ vectors)
    assert J.thermal_only


def test_zero_trajectory_skips_derivative_factorization_and_verifies_zero_flow_adjoint(monkeypatch):
    import deflation_example.coupled_control as control

    problem = small_coupled_problem([0.2, 0.35])
    problem.thermal_boundary[:] = 0

    def forbidden(*args, **kwargs):
        raise AssertionError("An exact zero trajectory requires no momentum derivative factors")

    monkeypatch.setattr(control, "splu", forbidden)
    evaluation = problem.evaluate(np.zeros(problem.size))
    assert evaluation.jacobian.thermal_only
    assert evaluation.jacobian.factors == (None, None)
    direction = np.arange(problem.size, dtype=float)
    np.testing.assert_array_equal(
        evaluation.jacobian @ direction, evaluation.frozen_operator @ direction
    )
    np.testing.assert_array_equal(
        evaluation.jacobian.T @ direction, evaluation.frozen_operator.T @ direction
    )
    adjoint = problem.verify_adjoint(evaluation, np.full(problem.size, 0.1))
    assert adjoint["maximum_momentum_adjoint_relative_residual"] == 0
    assert adjoint["gradient_relative_difference"] < 1e-14


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_independent_momentum_adjoint_checks_detect_an_incorrect_transpose_solve(steps):
    problem = small_coupled_problem(steps)
    evaluation = problem.evaluate(np.linspace(0.04, 0.1, problem.size))
    desired = np.full(problem.size, 0.12)
    report = problem.verify_adjoint(evaluation, desired)
    assert report["maximum_momentum_adjoint_relative_residual"] < 1e-11
    assert report["gradient_relative_difference"] < 1e-11
    original = evaluation.jacobian.factors[0]

    class IncorrectTranspose:
        def solve(self, rhs, trans="N"):
            answer = original.solve(rhs, trans=trans)
            return 2 * answer if trans == "T" else answer

    evaluation.jacobian.factors = (IncorrectTranspose(), *evaluation.jacobian.factors[1:])
    failed = problem.verify_adjoint(evaluation, desired)
    assert failed["maximum_momentum_adjoint_relative_residual"] > 0.9


def small_coupled_problem(steps=None, feedback=0.003, uniform_capacity=False):
    mesh = annular_rectangle(3)
    flow = AxisymmetricFlow(mesh, 0.1)
    velocity = np.column_stack((np.zeros(flow.nv), np.full(flow.nv, 0.02)))
    baseline = flow.solve(
        np.zeros_like(flow.quadrature_points),
        flow.boundary,
        velocity[flow.boundary],
        pressure_gauge=(0, 0),
        method="newton",
    )
    assert baseline.status == "converged"
    nc = len(mesh.cells)
    conductivity = np.tile(np.diag([0.07, 0.12]), (nc, 1, 1))
    capacity = np.ones(nc) if uniform_capacity else np.linspace(0.8, 1.3, nc)
    return CoupledControlProblem(
        flow,
        conductivity,
        capacity,
        np.linspace(0, 0.02, nc),
        flow.boundary,
        velocity[flow.boundary],
        baseline,
        alpha=0.003,
        expansion=feedback,
        temperature_scale=2.0,
        physical_steps=steps,
        time_scale=3.0,
        velocity_scale=5.0,
        pressure_gauge=(0, 0),
        initial_temperature=np.full(len(mesh.free), 0.04),
        thermal_boundary=0.03,
        flow_tolerance=1e-11,
    )


def test_buoyancy_linear_map_includes_temperature_scale_and_pressure_rows():
    flow = AxisymmetricFlow(annular_rectangle(3), 0.1)
    y = np.random.default_rng(10).normal(size=len(flow.mesh.nodes))
    J = buoyancy_jacobian(flow, 0.002, 20.0)
    load = flow.load(flow.buoyancy(300 + 20 * y, 300, 0.002))
    np.testing.assert_allclose(J @ y, np.r_[load[:, 0], load[:, 1], np.zeros(flow.np)], atol=1e-14)


@pytest.mark.parametrize("speed", [0.0, 0.001, 4.0])
def test_thermal_velocity_derivative_includes_both_stabilization_branches(speed):
    problem = small_coupled_problem()
    flow = problem.flow
    rng = np.random.default_rng(13)
    velocity = speed * rng.normal(size=(flow.nv, 2))
    direction = rng.normal(size=velocity.shape)
    y = rng.normal(size=len(flow.mesh.nodes))
    J = thermal_velocity_jacobian(
        flow, velocity, y, problem.capacity, problem.conductivity, problem.velocity_scale
    )
    h = 1e-6
    numerical = (
        (problem.assemble(velocity + h * direction).stiffness @ y)
        - (problem.assemble(velocity - h * direction).stiffness @ y)
    ) / (2 * h)
    analytic = J @ np.r_[direction[:, 0], direction[:, 1], np.zeros(flow.np)]
    np.testing.assert_allclose(analytic, numerical, rtol=2e-7, atol=1e-7)


@pytest.mark.parametrize("steps", [None, [0.2, 0.35, 0.15]])
def test_coupled_control_tangent_transpose_gradient_and_spd(steps):
    problem = small_coupled_problem(steps)
    rng = np.random.default_rng(41)
    y = rng.uniform(0.05, 0.15, problem.size)
    d, z = rng.normal(size=(2, problem.size))
    evaluation = problem.evaluate(y)
    J = evaluation.jacobian
    assert abs(z @ (J @ d) - d @ (J.T @ z)) < 1e-10
    pair = np.column_stack((d, z))
    np.testing.assert_allclose(J @ pair, np.column_stack((J @ d, J @ z)), atol=1e-12)
    np.testing.assert_allclose(J.T @ pair, np.column_stack((J.T @ d, J.T @ z)), atol=1e-12)
    h = 1e-5
    plus, minus = problem.evaluate(y + h * d), problem.evaluate(y - h * d)
    np.testing.assert_allclose(
        (plus.control - minus.control) / (2 * h), J @ d, rtol=5e-7, atol=5e-7
    )
    target = np.full(problem.size, 0.3)
    value, gradient = problem.objective_gradient(evaluation, target)
    remainders = []
    for step in (0.002, 0.001, 0.0005, 0.00025):
        trial, _ = problem.objective_gradient(problem.evaluate(y + step * d), target)
        remainders.append(abs(trial - value - step * (gradient @ d)))
    orders = np.log2(np.array(remainders[:-1]) / remainders[1:])
    assert np.min(orders) > 1.95
    H = GaussNewtonOperator(J, problem.weights, problem.alpha)
    dense = H @ np.eye(problem.size)
    np.testing.assert_allclose(dense, dense.T, atol=1e-11)
    assert np.linalg.eigvalsh(dense).min() >= problem.weights.min() * (1 - 1e-10)
    indices = np.arange(0, problem.size, 2)
    np.testing.assert_allclose(
        H.restrict(indices) @ d[indices], dense[np.ix_(indices, indices)] @ d[indices], atol=1e-11
    )
    if steps:
        full_j = J @ np.eye(problem.size)
        n = problem.spatial_size
        assert np.linalg.norm(full_j[n : 2 * n, :n]) > 0
        np.testing.assert_allclose(full_j[:n, n:], 0, atol=1e-12)
        # Feedback carries additional history beyond the thermal first subdiagonal.
        assert np.linalg.norm(full_j[2 * n :, :n]) > 1e-8


def test_feedback_disabled_control_jacobian_matches_frozen_operator():
    problem = small_coupled_problem([0.2, 0.3], feedback=0)
    y = np.linspace(0.02, 0.2, problem.size)
    evaluation = problem.evaluate(y)
    np.testing.assert_allclose(evaluation.jacobian @ y, evaluation.frozen_operator @ y, atol=1e-12)
    np.testing.assert_allclose(
        evaluation.jacobian.T @ y, evaluation.frozen_operator.T @ y, atol=1e-12
    )


def test_objective_difference_matches_quadratic_change_with_large_constant_penalty():
    from types import SimpleNamespace

    problem = small_coupled_problem()
    n = problem.size
    reference = SimpleNamespace(state=np.zeros(n), control=np.full(n, 1e10))
    candidate = SimpleNamespace(state=np.full(n, 1e-6), control=reference.control.copy())
    target = np.ones(n)
    expected = np.sum(problem.weights * (0.5e-12 - 1e-6))
    assert problem.objective_difference(candidate, reference, target) == pytest.approx(expected)
    assert problem.objective_difference(reference, candidate, target) == pytest.approx(-expected)
    assert problem.objective_difference(reference, reference, target) == 0


def test_objective_difference_matches_evaluated_coupled_objectives():
    problem = small_coupled_problem([0.2, 0.35])
    initial = problem.evaluate(np.linspace(0.04, 0.1, problem.size))
    trial = problem.evaluate(initial.state + 0.002, initial=initial)
    target = np.full(problem.size, 0.15)
    expected = (
        problem.objective_gradient(trial, target)[0]
        - problem.objective_gradient(initial, target)[0]
    )
    assert problem.objective_difference(trial, initial, target) == pytest.approx(
        expected, abs=1e-13
    )


def test_failed_flow_evaluation_retains_termination_details():
    problem = small_coupled_problem()
    problem.flow_cap = 1
    with pytest.raises(FlowEvaluationError) as caught:
        problem.evaluate(np.linspace(0.1, 1.0, problem.size))
    assert caught.value.result.status != "converged"
    assert caught.value.slab == 0
    assert caught.value.metrics["momentum_relative_residual"] > problem.flow_tolerance


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_recovered_control_reproduces_full_coupled_trajectory(steps):
    from deflation_example.coupled_forward import CoupledForward

    problem = small_coupled_problem(steps, uniform_capacity=True)
    state = np.linspace(0.03, 0.15, problem.size)
    evaluation = problem.evaluate(state)
    checks = problem.verify(evaluation)
    assert max(row["mass_relative_imbalance"] for row in checks) < 1e-10
    assert max(row["energy"]["relative_defect"] for row in checks) < 1e-9
    assert max(row["thermal_relative_residual"] for row in checks) < 1e-12
    from deflation_example.meshes import assemble_thermal

    def build(cell_velocity):
        return assemble_thermal(
            problem.mesh,
            problem.conductivity,
            problem.capacity,
            problem.velocity_scale * cell_velocity,
            problem.source,
            streamline=True,
        )

    forward = CoupledForward(
        problem.flow,
        build,
        problem.boundary_indices,
        problem.boundary_values,
        expansion=problem.expansion,
        temperature_scale=problem.temperature_scale,
        temperature_offset=problem.temperature_offset,
        buoyancy_reference=problem.buoyancy_reference,
        time_scale=problem.time_scale,
        thermal_boundary=problem.thermal_boundary,
        transport_factor=problem.velocity_scale,
        pressure_gauge=problem.pressure_gauge,
        flow_method="newton",
    )
    previous_y = problem.full_temperature(problem.initial)
    previous_v = problem.initial_flow
    for n in range(problem.slabs):
        control = np.zeros(len(problem.mesh.nodes))
        control[problem.free] = evaluation.control.reshape(problem.slabs, -1)[n]
        options = (
            {}
            if steps is None
            else {
                "time_step": steps[n],
                "previous_state": previous_y,
                "previous_velocity": previous_v.velocity,
            }
        )
        result = forward.solve(
            control, previous_y, previous_v, tolerance=1e-9, max_iterations=150, **options
        )
        assert result.status == "converged", result.history[-1]
        expected = problem.full_temperature(state.reshape(problem.slabs, -1)[n])
        np.testing.assert_allclose(result.state, expected, atol=1e-8)
        np.testing.assert_allclose(result.flow.velocity, evaluation.flows[n].velocity, atol=1e-9)
        previous_y, previous_v = result.state, result.flow
