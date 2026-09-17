"""Branch selection and retained-field checks for the forward-only diagnostic."""

import numpy as np
import pytest

from deflation_example.coupled_derivatives import (
    StabilizationBranchError,
    thermal_velocity_jacobian,
)
from deflation_example.coupled_newton_replay import newton_step, step_equations
from deflation_example.coupled_resolution import forward_model
from test_coupled_derivatives import small_coupled_problem
from test_coupled_newton_replay import data


def switch_problem(side=0):
    problem = small_coupled_problem([0.2], uniform_capacity=True)
    velocity = np.column_stack((np.zeros(problem.flow.nv), np.ones(problem.flow.nv)))
    vertices = problem.mesh.nodes[problem.mesh.cells]
    h = np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=3).max(axis=(1, 2))
    # tau_advection = tau_diffusion when k = h c |v_scaled| / 6.
    k = h * problem.capacity * problem.velocity_scale / 6
    problem.conductivity[:] = (1 + side) * k[:, None, None] * np.eye(2)
    return problem, velocity


@pytest.mark.parametrize("side", [-1e-13, 0, 1e-13])
def test_strict_policy_rejects_the_switch_and_active_policy_is_finite(side):
    problem, velocity = switch_problem(side)
    state = np.linspace(0.03, 0.2, len(problem.mesh.nodes))
    arguments = (
        problem.flow,
        velocity,
        state,
        problem.capacity,
        problem.conductivity,
        problem.velocity_scale,
    )
    with pytest.raises(StabilizationBranchError):
        thermal_velocity_jacobian(*arguments)
    action = thermal_velocity_jacobian(*arguments, branch_policy="active")
    assert np.isfinite(action.data).all()


@pytest.mark.parametrize("side", [-1e-13, 1e-13])
def test_one_sided_action_near_the_switch_matches_the_selected_branch(side):
    problem, velocity = switch_problem(side)
    state = np.linspace(0.03, 0.2, len(problem.mesh.nodes))
    action = thermal_velocity_jacobian(
        problem.flow,
        velocity,
        state,
        problem.capacity,
        problem.conductivity,
        problem.velocity_scale,
        branch_policy="active",
    )
    direction = -np.sign(side) * velocity
    step = 1e-6
    numerical = (
        problem.assemble(velocity + step * direction).stiffness @ state
        - problem.assemble(velocity).stiffness @ state
    ) / step
    selected = action @ np.r_[direction[:, 0], direction[:, 1], np.zeros(problem.flow.np)]
    np.testing.assert_allclose(selected, numerical, rtol=3e-6, atol=1e-8)


@pytest.mark.parametrize("side", [-1e-3, 1e-3])
def test_active_policy_matches_the_smooth_branch_and_finite_differences(side):
    problem, velocity = switch_problem(side)
    state = np.linspace(0.03, 0.2, len(problem.mesh.nodes))
    arguments = (
        problem.flow,
        velocity,
        state,
        problem.capacity,
        problem.conductivity,
        problem.velocity_scale,
    )
    active = thermal_velocity_jacobian(*arguments, branch_policy="active")
    strict = thermal_velocity_jacobian(*arguments)
    np.testing.assert_array_equal(active.toarray(), strict.toarray())
    direction = np.random.default_rng(341).normal(size=velocity.shape)
    step = 1e-6
    numerical = (
        problem.assemble(velocity + step * direction).stiffness @ state
        - problem.assemble(velocity - step * direction).stiffness @ state
    ) / (2 * step)
    exact = active @ np.r_[direction[:, 0], direction[:, 1], np.zeros(problem.flow.np)]
    np.testing.assert_allclose(exact, numerical, rtol=2e-7, atol=1e-8)


def test_switch_termination_returns_fields_matching_original_residual(monkeypatch):
    import deflation_example.coupled_newton_replay as module

    problem, _, _, source = data()
    previous = problem.full_temperature(problem.initial)

    def at_switch(*args, **kwargs):
        raise StabilizationBranchError("test switch")

    monkeypatch.setattr(module, "step_linearization", at_switch)
    result = newton_step(problem, source, previous, problem.initial_flow, 0)
    assert result.status == "newton_stabilization_switch"
    np.testing.assert_array_equal(result.state, previous)
    np.testing.assert_array_equal(result.flow.velocity, problem.initial_flow.velocity)
    _, checks = step_equations(
        problem,
        forward_model(problem),
        result.state,
        result.flow,
        source,
        previous,
        problem.initial_flow,
        0,
    )
    for key, value in checks.items():
        assert result.history[-1][key] == value


def test_branch_policy_is_validated_even_for_zero_budget():
    problem, _, _, source = data()
    with pytest.raises(ValueError, match="branch policy"):
        newton_step(
            problem,
            source,
            problem.full_temperature(problem.initial),
            problem.initial_flow,
            0,
            branch_policy="unknown",
            max_iterations=0,
        )


def test_active_branch_is_forwarded_without_changing_acceptance():
    problem, expected, _, source = data()
    result = newton_step(
        problem,
        source,
        problem.full_temperature(problem.initial),
        problem.initial_flow,
        0,
        branch_policy="active",
        line_search="fixed_scaled",
        tolerance=1e-11,
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.state[problem.free], expected.reshape(2, -1)[0], atol=2e-10)
