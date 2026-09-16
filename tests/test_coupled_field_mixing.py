"""Matched complete-field Anderson updates and original-equation acceptance."""

import numpy as np
import pytest

from deflation_example.coupled_fixed_point import solve_forward
from deflation_example.coupled_newton_replay import criteria_met, step_equations
from deflation_example.coupled_resolution import forward_model
from test_coupled_newton_replay import data


@pytest.mark.parametrize("mixing", ["temperature", "coupled"])
def test_known_coupled_solution_with_fixed_control_and_consistent_checks(mixing):
    problem, expected, _, source = data()
    model = forward_model(problem)
    previous = problem.full_temperature(problem.initial)
    flow = problem.initial_flow
    original = [x.copy() for x in (source, previous, flow.velocity, flow.pressure)]
    result = solve_forward(
        model,
        source,
        previous,
        flow,
        previous_state=previous,
        previous_velocity=flow.velocity,
        time_step=problem.physical_steps[0],
        mixing=mixing,
        tolerance=1e-12,
        depth=5,
    )
    _, checks = step_equations(problem, model, result.state, result.flow, source, previous, flow, 0)
    assert result.status == "converged", result.history[-1]
    assert criteria_met(checks, 1e-12)
    np.testing.assert_allclose(result.state[problem.free], expected.reshape(2, -1)[0], atol=1e-10)
    for a, b in zip(original, (source, previous, flow.velocity, flow.pressure), strict=True):
        np.testing.assert_array_equal(a, b)


def test_coupled_cap_returns_the_checked_retained_pair():
    problem, _, _, source = data()
    previous, flow = problem.full_temperature(problem.initial), problem.initial_flow
    result = solve_forward(
        forward_model(problem),
        source,
        previous,
        flow,
        previous_state=previous,
        previous_velocity=flow.velocity,
        time_step=problem.physical_steps[0],
        mixing="coupled",
        max_iterations=2,
        tolerance=1e-12,
    )
    _, checks = step_equations(
        problem, forward_model(problem), result.state, result.flow, source, previous, flow, 0
    )
    for key in (
        "momentum_relative_residual",
        "continuity_relative_residual",
        "thermal_relative_residual",
        "mass_relative_imbalance",
    ):
        assert result.history[-1][key] == pytest.approx(checks[key], rel=1e-12, abs=1e-15)
    assert result.status == "coupling_iteration_cap"


def test_invalid_mixing_is_rejected():
    problem, _, _, source = data()
    with pytest.raises(ValueError, match="mixing"):
        solve_forward(
            forward_model(problem),
            source,
            problem.full_temperature(problem.initial),
            problem.initial_flow,
            mixing="invalid",
        )
