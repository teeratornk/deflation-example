"""Captured operators retain the saved flow rather than polishing it again."""

from dataclasses import replace

import numpy as np
import pytest

from deflation_example.coupled_control import FlowEvaluationError
from deflation_example.coupled_retention_replay import rebuild
from test_coupled_derivatives import small_coupled_problem


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_reassembly_preserves_saved_flow_and_operator_without_solving(steps, monkeypatch):
    problem = small_coupled_problem(steps)
    state = np.linspace(0.04, 0.1, problem.size)
    original = problem.evaluate(state)

    def forbidden(*args, **kwargs):
        raise AssertionError("Captured flows must not receive another nonlinear update")

    monkeypatch.setattr("deflation_example.coupled_control.solve_momentum", forbidden)
    restored = problem.reassemble(state, original.flows)
    np.testing.assert_array_equal(restored.control, original.control)
    for a, b in zip(restored.flows, original.flows, strict=True):
        np.testing.assert_array_equal(a.velocity, b.velocity)
        np.testing.assert_array_equal(a.pressure, b.pressure)
    direction = np.arange(problem.size, dtype=float)
    np.testing.assert_array_equal(restored.jacobian @ direction, original.jacobian @ direction)
    np.testing.assert_array_equal(restored.jacobian.T @ direction, original.jacobian.T @ direction)


def test_reassembly_still_checks_momentum_and_temporal_coupling():
    problem = small_coupled_problem([0.2, 0.35])
    state = np.linspace(0.04, 0.1, problem.size)
    original = problem.evaluate(state)
    with pytest.raises(ValueError, match="every time slab"):
        problem.reassemble(state, original.flows[:1])
    wrong = list(original.flows)
    changed = wrong[1].velocity.copy()
    free = np.setdiff1d(np.arange(problem.flow.nv), problem.boundary_indices)
    changed[free] *= 1.5
    wrong[1] = replace(wrong[1], velocity=changed)
    with pytest.raises(FlowEvaluationError):
        problem.reassemble(state, wrong)
    with pytest.raises(FlowEvaluationError):
        problem.reassemble(state + 0.1, original.flows)
    wrong[1] = replace(wrong[1], pressure=np.full_like(wrong[1].pressure, np.nan))
    with pytest.raises(ValueError, match="finite matching dimensions"):
        problem.reassemble(state, wrong)


def test_replay_uses_reassembly_and_keeps_saved_gradients(monkeypatch):
    problem = small_coupled_problem([0.2, 0.35])
    state = np.linspace(0.04, 0.1, problem.size)
    original = problem.evaluate(state)
    desired = np.full(problem.size, 0.12)
    _, gradient = problem.objective_gradient(original, desired)
    arrays = {
        "state": state,
        "velocity": np.stack([f.velocity for f in original.flows]),
        "pressure": np.stack([f.pressure for f in original.flows]),
        "desired": desired,
        "secant_steps": np.empty((0, problem.size)),
        "secant_gradients": np.empty((0, problem.size)),
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must use the captured flow")

    monkeypatch.setattr(problem, "evaluate", forbidden)
    evaluation, _, _, actual = rebuild(problem, arrays, {"damping": 0.0})
    np.testing.assert_array_equal(actual, gradient)
    np.testing.assert_array_equal(evaluation.control, original.control)
