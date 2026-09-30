"""Local perturbations preserve units, fixed history, and independent checks."""

import numpy as np
import pytest

from deflation_example.coupled_flow_response import (
    response_case,
    tangent_response,
    temperature_direction,
)
from test_coupled_derivatives import small_coupled_problem


def test_direction_has_physical_kelvin_scale_and_fixed_boundary():
    problem = small_coupled_problem([0.2, 0.35])
    state = np.linspace(0.04, 0.1, problem.size)
    direction = temperature_direction(problem, state, np.zeros_like(state), 1)
    assert np.max(np.abs(direction)) == 1
    np.testing.assert_array_equal(direction[problem.mesh.dirichlet], 0)
    for invalid in (state.copy(), np.full_like(state, np.nan), state[:-1]):
        with pytest.raises(ValueError):
            temperature_direction(problem, state, invalid, 1)


def test_local_flow_tangent_matches_fixed_history_perturbation():
    problem = small_coupled_problem([0.2, 0.35])
    state = np.linspace(0.04, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    saved = [f.velocity.copy() for f in evaluation.flows]
    retained, previous = evaluation.flows[1], evaluation.flows[0].velocity
    direction = temperature_direction(problem, state, np.zeros_like(state), 1)
    tangent, checks = tangent_response(problem, retained, previous, state, 1, direction)
    assert max(checks["retained_residuals"].values()) < problem.flow_tolerance
    assert checks["linear_relative_residual"] < 1e-12
    step = 1e-3
    answers = []
    for sign in (-1, 1):
        result, row = response_case(
            problem, state, 1, direction, retained, previous, retained, sign * step
        )
        assert row["verified"]
        answers.append(result.velocity)
    numerical = (answers[1] - answers[0]) / (2 * step)
    expected = np.column_stack(
        (tangent[: problem.flow.nv], tangent[problem.flow.nv : 2 * problem.flow.nv])
    )
    np.testing.assert_allclose(numerical, expected, atol=1e-9, rtol=1e-6)
    for flow, old in zip(evaluation.flows, saved, strict=True):
        np.testing.assert_array_equal(flow.velocity, old)


def test_failed_solver_status_cannot_be_promoted_by_small_residual(monkeypatch):
    from deflation_example import coupled_flow_response as diagnostic

    problem = small_coupled_problem([0.2, 0.35])
    state = np.linspace(0.04, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    retained = evaluation.flows[1]
    retained.status = "iteration_cap"
    monkeypatch.setattr(diagnostic, "solve_momentum", lambda *a, **kw: retained)
    direction = temperature_direction(problem, state, np.zeros_like(state), 1)
    _, row = response_case(
        problem, state, 1, direction, retained, evaluation.flows[0].velocity, retained, 0.0
    )
    assert max(row["residuals"].values()) < problem.flow_tolerance
    assert not row["verified"]
    assert row["status"] == "iteration_cap"
