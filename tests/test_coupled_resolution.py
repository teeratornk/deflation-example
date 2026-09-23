"""Fixed-source replay and piecewise-constant control transfer checks."""

import hashlib
from types import SimpleNamespace
import numpy as np
import pytest

from deflation_example import coupled_resolution as resolution
from test_coupled_derivatives import small_coupled_problem


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("streamline_rule", ["hard_min", "smooth_p8"])
def test_forward_builder_preserves_optimization_thermal_assembly(consistent, streamline_rule):
    problem = small_coupled_problem(
        [0.2, 0.35],
        uniform_capacity=True,
        inlet=0.2,
        consistent=consistent,
        streamline_rule=streamline_rule,
    )
    velocity = problem.initial_flow.velocity
    expected = problem.assemble(velocity)
    actual = resolution.forward_model(problem).thermal_builder(
        problem.flow.thermal_velocity(velocity)
    )
    for key in ("stiffness", "storage", "source_action"):
        left, right = getattr(actual, key), getattr(expected, key)
        np.testing.assert_array_equal(left.toarray(), right.toarray())
    np.testing.assert_array_equal(actual.load, expected.load)


def test_forward_replay_recovers_the_temperature_eliminated_equations():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    state = np.linspace(0.045, 0.09, problem.size)
    evaluation = problem.evaluate(state)
    original = evaluation.control.copy()
    result = resolution.replay_controls(problem, original, subdivision=1, tolerance=1e-9)
    assert result["status"] == "converged"
    np.testing.assert_allclose(result["states"].ravel(), state, atol=2e-9)
    np.testing.assert_array_equal(original, evaluation.control)
    assert result["control_sha256"] == hashlib.sha256(original.tobytes()).hexdigest()


def test_time_subdivision_repeats_each_source_without_clipping_temperature(monkeypatch):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    calls = []

    class Forward:
        def __init__(self, *args, **kwargs):
            pass

        def solve(self, source, state, flow, **kwargs):
            calls.append((source.copy(), kwargs["time_step"], state.copy()))
            return SimpleNamespace(
                state=state + 10, flow=flow, status="converged", history=[], seconds=0.0
            )

    monkeypatch.setattr(resolution, "CoupledForward", Forward)
    control = np.r_[np.full(problem.spatial_size, 2.0), np.full(problem.spatial_size, -3.0)]
    result = resolution.replay_controls(problem, control, subdivision=2)
    assert len(calls) == 4
    for index, value in enumerate((2.0, 2.0, -3.0, -3.0)):
        np.testing.assert_array_equal(calls[index][0][problem.free], value)
        np.testing.assert_array_equal(calls[index][0][problem.mesh.dirichlet], 0)
    np.testing.assert_allclose([c[1] for c in calls], [0.1, 0.1, 0.175, 0.175])
    assert result["states"].max() > 40
