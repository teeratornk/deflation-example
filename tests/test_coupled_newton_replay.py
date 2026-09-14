"""Check unchanged controls, coupled equations and retained Newton candidates."""

import numpy as np
import pytest

from deflation_example.coupled_newton_replay import newton_step, step_equations, criteria_met
from deflation_example.coupled_resolution import forward_model
from test_coupled_derivatives import small_coupled_problem


def data():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    state = np.linspace(0.04, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    source = np.zeros(len(problem.mesh.nodes))
    source[problem.free] = evaluation.control.reshape(problem.slabs, -1)[0]
    return problem, state, evaluation, source


def test_forward_newton_recovers_complete_known_trajectory_from_fixed_sources():
    problem, expected, evaluation, _ = data()
    state = problem.full_temperature(problem.initial)
    flow = problem.initial_flow
    for n in range(problem.slabs):
        source = np.zeros(len(problem.mesh.nodes))
        source[problem.free] = evaluation.control.reshape(problem.slabs, -1)[n]
        original = source.copy()
        result = newton_step(problem, source, state, flow, n, tolerance=1e-11)
        assert result.status == "converged", result.history
        _, metrics = step_equations(
            problem, forward_model(problem), result.state, result.flow, source, state, flow, n
        )
        assert criteria_met(metrics, 1e-11)
        for name, value in metrics.items():
            assert result.history[-1][name] == value
        np.testing.assert_allclose(
            result.state[problem.free], expected.reshape(2, -1)[n], atol=2e-10
        )
        np.testing.assert_array_equal(source, original)
        assert len(result.history) <= 10
        state, flow = result.state, result.flow


def test_converged_initial_guard_and_zero_budget_keep_matching_fields():
    problem, expected, evaluation, source = data()
    previous = problem.full_temperature(problem.initial)
    initial = problem.full_temperature(expected.reshape(2, -1)[0])
    saved = newton_step(
        problem,
        source,
        previous,
        problem.initial_flow,
        0,
        initial_state=initial,
        initial_flow=evaluation.flows[0],
        tolerance=1e-10,
        max_iterations=0,
    )
    assert saved.status == "converged"
    assert len(saved.history) == 1
    np.testing.assert_array_equal(saved.state, initial)
    capped = newton_step(problem, source, previous, problem.initial_flow, 0, max_iterations=0)
    assert capped.status == "newton_iteration_cap"
    np.testing.assert_array_equal(capped.state, previous)


def test_linear_failure_retains_previous_temperature_velocity_and_residual(monkeypatch):
    import deflation_example.coupled_newton_replay as module

    problem, _, _, source = data()
    previous = problem.full_temperature(problem.initial)

    def fail(*args, **kwargs):
        raise RuntimeError("test factorization failure")

    monkeypatch.setattr(module, "splu", fail)
    result = newton_step(problem, source, previous, problem.initial_flow, 0)
    assert result.status == "newton_linear_failure"
    np.testing.assert_array_equal(result.state, previous)
    np.testing.assert_array_equal(result.flow.velocity, problem.initial_flow.velocity)
    _, metrics = step_equations(
        problem,
        forward_model(problem),
        result.state,
        result.flow,
        source,
        previous,
        problem.initial_flow,
        0,
    )
    for name, value in metrics.items():
        assert result.history[-1][name] == value


@pytest.mark.parametrize("tolerance", [0, -1, np.nan, 1e-7])
def test_invalid_forward_targets_are_rejected(tolerance):
    problem, _, _, source = data()
    with pytest.raises(ValueError):
        newton_step(
            problem,
            source,
            problem.full_temperature(problem.initial),
            problem.initial_flow,
            0,
            tolerance=tolerance,
        )


def test_zero_temperature_and_source_need_no_newton_factorization(monkeypatch):
    import deflation_example.coupled_newton_replay as module

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    problem.initial[:] = 0
    problem.thermal_boundary[:] = 0
    problem.source[:] = 0

    def forbidden(*args, **kwargs):
        raise AssertionError("The verified zero state must return before factorization")

    monkeypatch.setattr(module, "splu", forbidden)
    result = newton_step(
        problem,
        np.zeros(len(problem.mesh.nodes)),
        problem.full_temperature(problem.initial),
        problem.initial_flow,
        0,
        tolerance=1e-10,
    )
    assert result.status == "converged"
    assert len(result.history) == 1
    assert not np.any(result.state)


def test_rejected_newton_trials_keep_the_initial_fields_and_metrics(monkeypatch):
    import deflation_example.coupled_newton_replay as module

    problem, _, _, source = data()
    previous = problem.full_temperature(problem.initial)
    evaluate = module.step_equations
    first = None

    def reject_trials(*args, **kwargs):
        nonlocal first
        residual, metrics = evaluate(*args, **kwargs)
        if first is None:
            first = dict(metrics)
        else:
            metrics = {name: 2 * value for name, value in first.items()}
        return residual, metrics

    monkeypatch.setattr(module, "step_equations", reject_trials)
    result = newton_step(problem, source, previous, problem.initial_flow, 0)
    assert result.status == "newton_line_search_stagnation"
    assert len(result.history[-1]["trials"]) == 21
    assert not result.history[-1]["candidate_retained"]
    np.testing.assert_array_equal(result.state, previous)
    np.testing.assert_array_equal(result.flow.velocity, problem.initial_flow.velocity)
    for name, value in first.items():
        assert result.history[-1][name] == value
