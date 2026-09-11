"""Residual-load continuation verifies the original terminal equation."""

import numpy as np
import pytest

from deflation_example.coupled_flow_solve import solve_momentum
from test_coupled_derivatives import small_coupled_problem


@pytest.mark.parametrize("transient", [False, True])
def test_continuation_removes_the_artificial_load_and_retains_failed_attempt(
    monkeypatch, transient
):
    problem = small_coupled_problem()
    flow = problem.flow
    real_solve = flow.solve
    calls = []

    def solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        calls.append(result.status)
        if len(calls) == 1:
            # Exercise the fallback independently of platform-specific Newton paths.
            result.status = "iteration_cap"
        return result

    monkeypatch.setattr(flow, "solve", solve)
    temperature = problem.full_temperature(np.linspace(0.01, 0.1, problem.size))
    acceleration = flow.buoyancy(300 + 2 * temperature, 300, 0.003)
    options = {"previous": problem.initial_flow.velocity, "time_step": 0.2} if transient else {}
    result = solve_momentum(
        flow,
        acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        initial=problem.initial_flow,
        pressure_gauge=(0, 0),
        continuation=True,
        tolerance=1e-10,
        **options,
    )
    assert result.status == "converged"
    assert result.history[0]["status"] == "iteration_cap"
    assert result.history[-1]["load_fraction"] == 1.0
    assert (
        max(
            flow.verify(
                result,
                acceleration,
                problem.boundary_indices,
                problem.boundary_values,
                pressure_gauge=(0, 0),
                **options,
            ).values()
        )
        < 1e-10
    )
    expected = real_solve(
        acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        initial=problem.initial_flow,
        pressure_gauge=(0, 0),
        method="newton",
        tolerance=1e-10,
        **options,
    )
    np.testing.assert_allclose(result.velocity, expected.velocity, atol=1e-10)
    np.testing.assert_allclose(result.pressure, expected.pressure, atol=1e-10)


def test_incomplete_continuation_does_not_claim_a_physical_solution(monkeypatch):
    problem = small_coupled_problem()
    flow = problem.flow
    real_solve = flow.solve
    calls = []

    def solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            result.status = "iteration_cap"
        return result

    monkeypatch.setattr(flow, "solve", solve)
    temperature = problem.full_temperature(np.linspace(0.01, 0.1, problem.size))
    acceleration = flow.buoyancy(300 + 2 * temperature, 300, 0.003)
    result = solve_momentum(
        flow,
        acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        initial=problem.initial_flow,
        pressure_gauge=(0, 0),
        continuation=True,
        tolerance=1e-10,
        max_stages=1,
    )
    assert result.status == "load_continuation_cap"
    assert result.history[-1]["load_fraction"] < 1
    assert (
        flow.verify(
            result,
            acceleration,
            problem.boundary_indices,
            problem.boundary_values,
            pressure_gauge=(0, 0),
        )["momentum_relative_residual"]
        > 1e-4
    )
