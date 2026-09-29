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


def test_expired_deadline_returns_initial_flow_without_linear_solve(monkeypatch):
    from deflation_example import axisymmetric_flow

    problem = small_coupled_problem()

    def forbidden(*args, **kwargs):
        raise AssertionError("An expired solve must not factor a matrix")

    monkeypatch.setattr(axisymmetric_flow, "spsolve", forbidden)
    result = solve_momentum(
        problem.flow,
        problem.acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        initial=problem.initial_flow,
        pressure_gauge=problem.pressure_gauge,
        continuation=True,
        stop_requested=lambda: True,
    )
    assert result.status == "budget_exhausted"
    np.testing.assert_array_equal(result.velocity, problem.initial_flow.velocity)
    np.testing.assert_array_equal(result.pressure, problem.initial_flow.pressure)
    assert result.history == []


def test_deadline_stops_continuation_without_accepting_artificial_load(monkeypatch):
    problem = small_coupled_problem()
    real_solve = problem.flow.solve
    calls = []

    def solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        calls.append(result)
        if len(calls) == 1:
            result.status = "iteration_cap"
        return result

    monkeypatch.setattr(problem.flow, "solve", solve)
    result = solve_momentum(
        problem.flow,
        problem.acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        initial=problem.initial_flow,
        pressure_gauge=problem.pressure_gauge,
        continuation=True,
        stop_requested=lambda: len(calls) >= 2,
    )
    assert result.status == "budget_exhausted"
    assert len(calls) == 2
    assert result.history[-1]["load_fraction"] == 0.25


@pytest.mark.parametrize("polish", [False, True])
def test_optional_load_polish_keeps_original_tolerance(monkeypatch, polish):
    problem = small_coupled_problem()
    flow, tolerance = problem.flow, 1e-10
    real_solve, real_verify = flow.solve, flow.verify
    calls, checks = [], []

    def solve(*args, **kwargs):
        assert kwargs["tolerance"] == tolerance
        result = real_solve(*args, **kwargs)
        calls.append(result)
        if len(calls) == 1:
            result.status = "iteration_cap"
        return result

    def verify(*args, **kwargs):
        result = real_verify(*args, **kwargs)
        checks.append(result)
        if len(checks) == 1:
            return {**result, "momentum_relative_residual": 1.24 * tolerance}
        return result

    monkeypatch.setattr(flow, "solve", solve)
    monkeypatch.setattr(flow, "verify", verify)
    result = solve_momentum(
        flow,
        problem.acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        initial=problem.initial_flow,
        pressure_gauge=problem.pressure_gauge,
        continuation=True,
        tolerance=tolerance,
        polish_load=polish,
    )
    assert result.status == ("converged" if polish else "load_initialization_failed")
    if polish:
        assert result.history[2]["status"] == "initial_load_polish"
        assert result.history[-1]["load_fraction"] == 1.0
        assert (
            max(
                real_verify(
                    result,
                    problem.acceleration,
                    problem.boundary_indices,
                    problem.boundary_values,
                    pressure_gauge=problem.pressure_gauge,
                ).values()
            )
            <= tolerance
        )


def test_nonfinite_final_continuation_check_is_never_accepted(monkeypatch):
    problem = small_coupled_problem()
    flow = problem.flow
    real_solve, real_verify = flow.solve, flow.verify
    calls, checks = [], []

    def solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            result.status = "iteration_cap"
        return result

    def verify(*args, **kwargs):
        result = real_verify(*args, **kwargs)
        checks.append(1)
        if len(checks) > 1:
            result["continuity_relative_residual"] = float("nan")
        return result

    monkeypatch.setattr(flow, "solve", solve)
    monkeypatch.setattr(flow, "verify", verify)
    result = solve_momentum(
        flow,
        problem.acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        initial=problem.initial_flow,
        pressure_gauge=problem.pressure_gauge,
        continuation=True,
    )
    assert result.status == "final_residual_failed"
