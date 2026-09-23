"""Bounds, derivatives, conservation and complete smooth-model optimization."""

import numpy as np
import pytest
from scipy.optimize import minimize

from deflation_example.coupled_derivatives import streamline_parameter
from deflation_example.coupled_optimizer import minimize_coupled
from deflation_example.meshes import assemble_thermal, simplex_geometry
from deflation_example.streamline import smooth_parameter
from test_axisymmetric_flow import annular_rectangle
from test_coupled_derivatives import small_coupled_problem
from test_coupled_optimizer import solver


def cell_data():
    mesh = annular_rectangle(3)
    grad, lump = simplex_geometry(mesh)
    count = len(mesh.cells)
    return grad, lump, np.ones(count), np.full(count, 0.07), mesh.nodes[mesh.cells]


@pytest.mark.parametrize("limited", [False, True])
def test_smooth_coefficient_respects_all_limits_and_rest(limited):
    grad, lump, c, k, vertices = cell_data()
    h = np.max(np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=3), axis=(1, 2))
    diffusion = h**2 / (12 * k)
    rng = np.random.default_rng(2009)
    unit = rng.normal(size=(len(c), 2))
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    for speed in (0, 1e-10, 0.1, 1, 1e8, 1e80):
        velocity = speed * unit
        tau, derivative = smooth_parameter(grad, lump, c, k, velocity, vertices, limited)
        limits = [diffusion]
        if speed:
            limits.append(h / (2 * c * speed))
            if limited:
                share = (lump / lump.sum(axis=1)[:, None]).min(axis=1)
                limits.append(
                    share / (c * np.linalg.norm(np.einsum("eid,ed->ei", grad, velocity), axis=1))
                )
        minimum = np.min(limits, axis=0)
        assert np.all(tau > 0) and np.isfinite(derivative).all()
        assert np.all(tau <= minimum * (1 + 2e-15))
        assert np.all(tau >= (3 if limited else 2) ** (-1 / 8) * minimum * (1 - 2e-15))
        if speed == 0:
            np.testing.assert_allclose(tau, diffusion, rtol=2e-15)
            np.testing.assert_array_equal(derivative, 0)


@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("switch", ["advection", "row"])
def test_smooth_derivative_at_old_switches(limited, switch):
    grad, lump, c, k, vertices = cell_data()
    h = np.max(np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=3), axis=(1, 2))
    direction = np.tile([0.6, 0.8], (len(c), 1))
    if switch == "advection":
        speed = 6 * k / (c * h)
    else:
        share = (lump / lump.sum(axis=1)[:, None]).min(axis=1)
        reach = np.linalg.norm(np.einsum("eid,ed->ei", grad, direction), axis=1)
        speed = share * 12 * k / (c * reach * h**2)
    velocity = speed[:, None] * direction
    tau, derivative = streamline_parameter(
        grad, lump, c, k, velocity, vertices, limited, "smooth_p8"
    )
    delta = np.tile([-0.4, 0.7], (len(c), 1))
    step = 1e-6
    plus = smooth_parameter(grad, lump, c, k, velocity + step * delta, vertices, limited)[0]
    minus = smooth_parameter(grad, lump, c, k, velocity - step * delta, vertices, limited)[0]
    expected = np.einsum("ed,ed->e", derivative, delta)
    np.testing.assert_allclose((plus - minus) / (2 * step), expected, rtol=2e-7, atol=1e-10)
    assert np.isfinite(tau).all()


def test_invalid_rule_is_rejected_without_changing_hard_default():
    problem = small_coupled_problem(consistent=True)
    assert problem.streamline_rule == "hard_min"
    with pytest.raises(ValueError, match="coefficients"):
        small_coupled_problem(streamline_rule="unknown")
    with pytest.raises(ValueError, match="edge length"):
        assemble_thermal(
            problem.mesh, problem.conductivity, problem.capacity, streamline_rule="smooth_p8"
        )


@pytest.mark.parametrize("steps", [None, [0.2, 0.35, 0.15]])
@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("inlet", [0.02, 0.2])
def test_complete_smooth_control_derivatives_and_equations(steps, consistent, inlet):
    problem = small_coupled_problem(
        steps,
        uniform_capacity=True,
        consistent=consistent,
        inlet=inlet,
        streamline_rule="smooth_p8",
    )
    state = np.linspace(0.04, 0.1, problem.size)
    desired = np.linspace(0.08, 0.15, problem.size)
    evaluation = problem.evaluate(state)
    rng = np.random.default_rng(205)
    direction, dual = rng.normal(size=(2, problem.size))
    direction /= np.linalg.norm(direction)
    J = evaluation.jacobian
    np.testing.assert_allclose(dual @ (J @ direction), direction @ (J.T @ dual), rtol=1e-11)
    _, gradient = problem.objective_gradient(evaluation, desired)
    remainders = []
    for step in (0.01, 0.005, 0.0025):
        trial = problem.evaluate(state + step * direction, initial=evaluation)
        remainders.append(
            np.linalg.norm(trial.control - evaluation.control - step * (J @ direction))
        )
    assert np.min(np.log2(np.array(remainders[:-1]) / remainders[1:])) > 1.8
    h = 1e-5
    plus = problem.evaluate(state + h * direction, initial=evaluation)
    minus = problem.evaluate(state - h * direction, initial=evaluation)
    fd = problem.objective_difference(plus, minus, desired) / (2 * h)
    assert fd == pytest.approx(gradient @ direction, rel=1e-7, abs=1e-9)
    adjoint = problem.verify_adjoint(evaluation, desired)
    assert adjoint["gradient_weight_normalized_difference"] < 1e-10
    checks = problem.verify(evaluation)
    assert max(row["thermal_relative_residual"] for row in checks) < 1e-12
    assert max(row["energy"]["relative_defect"] for row in checks) < 1e-9
    if consistent:
        assembly = problem.assembly
        for action in (assembly.source_action, assembly.storage):
            assert np.min(np.asarray(action.sum(axis=1))) > 0
        for weighted in (assembly.stabilized_source, assembly.stabilized_storage):
            np.testing.assert_allclose(np.asarray(weighted.sum(axis=0)), 0, atol=1e-13)


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_complete_smooth_optimization_matches_independent_optimizer(steps):
    problem = small_coupled_problem(
        steps, uniform_capacity=True, consistent=True, inlet=0.2, streamline_rule="smooth_p8"
    )
    desired = np.linspace(-0.5, 0.8, problem.size)
    result = minimize_coupled(
        problem,
        desired,
        -0.04,
        0.15,
        solver(),
        tolerance=1e-8,
        qp_tolerance=1e-11,
        max_iterations=60,
    )

    def objective(state):
        return problem.objective_gradient(problem.evaluate(state), desired)

    independent = minimize(
        objective,
        np.zeros(problem.size),
        jac=True,
        bounds=[(-0.04, 0.15)] * problem.size,
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 200},
    )
    assert result.status == "converged", (result.status, result.kkt)
    assert max(result.kkt.values()) <= 1e-8
    assert independent.success
    assert result.objective == pytest.approx(independent.fun, abs=1e-10)
    np.testing.assert_allclose(result.evaluation.state, independent.x, atol=2e-6)
