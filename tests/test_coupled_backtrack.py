"""Shorten one verified quadratic direction without changing final accuracy."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from deflation_example.axisymmetric_flow import FlowResult
from deflation_example.coupled_control import FlowEvaluationError
from deflation_example.coupled_trust import minimize_trust
from test_coupled_derivatives import small_coupled_problem
from test_coupled_optimizer import solver


class QuadraticProblem:
    size = 4
    alpha = 0.1
    temperature_scale = 1.0
    weights = np.ones(4)

    def __init__(self, maximum_step=1.0):
        self.maximum_step = maximum_step
        self.failed_flow = FlowResult(np.zeros((1, 2)), np.zeros(1), "stagnation", [])

    def evaluate(self, state, initial=None):
        if initial is not None and np.max(np.abs(state - initial.state)) > self.maximum_step:
            raise FlowEvaluationError(0, self.failed_flow, {"momentum_relative_residual": 0.01})
        return SimpleNamespace(
            state=np.asarray(state).copy(),
            control=np.asarray(state).copy(),
            jacobian=sparse.eye(self.size, format="csr"),
            seconds=0.0,
            flows=[self.failed_flow],
        )

    def objective_gradient(self, evaluation, desired):
        x = evaluation.state
        return 0.5 * ((x - desired) @ (x - desired) + self.alpha * (x @ x)), (
            1 + self.alpha
        ) * x - desired

    def objective_difference(self, trial, initial, desired):
        a, b = trial.state, initial.state
        return float((a - b) @ (0.5 * (1 + self.alpha) * (a + b) - desired))

    def preconditioning_diagonal(self, evaluation, damping):
        return np.full(self.size, 1 + self.alpha)


def test_one_quadratic_direction_survives_multiple_failed_flow_trials(monkeypatch):
    import deflation_example.coupled_trust as trust

    original, calls, captured = trust.box_quadratic, [], []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(trust, "box_quadratic", counted)
    problem = QuadraticProblem(maximum_step=0.1)
    desired = np.full(4, 0.4)

    def capture(event, base, candidate, failure):
        captured.append((dict(event), base.state.copy(), candidate.copy()))

    result = minimize_trust(
        problem,
        desired,
        -0.2,
        0.3,
        solver(),
        inner_preconditioner="jacobi",
        trial_policy="backtrack",
        trial_callback=capture,
        max_iterations=1,
    )
    assert len(calls) == 1
    trials = result.history[0]["attempts"][0]["trials"]
    assert [r["step"] for r in trials] == [1, 0.5, 0.25]
    assert [r["accepted"] for r in trials] == [False, False, True]
    np.testing.assert_allclose(result.evaluation.state, 0.0625)
    for event, base, candidate in captured:
        d = candidate - base
        expected = -float(((1.1 * base - desired) @ d) + 0.55 * (d @ d))
        assert event["predicted_reduction"] == pytest.approx(expected)
        assert np.min(candidate) >= -0.2 and np.max(candidate) <= 0.3
    assert result.objective == problem.objective_gradient(result.evaluation, desired)[0]
    assert result.history[0]["retained"]["objective"] == result.objective


def test_failed_direction_exhausts_seven_trials_before_new_quadratic(monkeypatch):
    import deflation_example.coupled_trust as trust

    monkeypatch.setitem(trust.POLICY, "minimum_radius_K", 0.2)
    problem = QuadraticProblem(maximum_step=0)
    result = minimize_trust(
        problem,
        np.ones(4),
        -0.2,
        0.3,
        solver(),
        inner_preconditioner="jacobi",
        trial_policy="backtrack",
    )
    assert result.status == "trust_radius_exhausted"
    attempts = result.history[0]["attempts"]
    assert len(attempts) == 1
    assert len(attempts[0]["trials"]) == 7
    np.testing.assert_array_equal(result.evaluation.state, np.zeros(4))


def test_budget_between_trials_retains_previous_nonlinear_state():
    problem, linear = QuadraticProblem(maximum_step=0), solver()
    stop = [False]

    def previous():
        return stop[0]

    original_rtol = linear.rtol
    linear.stop_requested = previous

    def capture(event, base, candidate, failure):
        if event["phase"] == "finished":
            stop[0] = True

    result = minimize_trust(
        problem,
        np.ones(4),
        -0.2,
        0.3,
        linear,
        inner_preconditioner="jacobi",
        trial_policy="backtrack",
        trial_callback=capture,
    )
    assert result.status == "budget_exhausted"
    assert len(result.history[0]["attempts"][0]["trials"]) == 1
    np.testing.assert_array_equal(result.evaluation.state, np.zeros(4))
    assert linear.stop_requested is previous
    assert linear.rtol == original_rtol


def test_shortened_steps_converge_to_independent_quadratic_optimum():
    problem, desired = QuadraticProblem(maximum_step=0.05), np.array([0.4, -0.3, 0.15, -0.1])
    result = minimize_trust(
        problem,
        desired,
        -0.2,
        0.3,
        solver(),
        inner_preconditioner="jacobi",
        trial_policy="backtrack",
        accuracy="adaptive",
    )
    assert result.status == "converged"
    assert max(result.kkt.values()) <= 1e-8
    np.testing.assert_allclose(
        result.evaluation.state, np.clip(desired / 1.1, -0.2, 0.3), atol=1e-8
    )


def test_shortened_roundoff_step_uses_actual_increment_for_next_radius(monkeypatch):
    from deflation_example.coupled_trust import POLICY

    problem, checkpoints = QuadraticProblem(maximum_step=1e-8), []
    # Simulate cancellation in the objective difference at near stationarity.
    monkeypatch.setattr(problem, "objective_difference", lambda *args: 0.0)
    result = minimize_trust(
        problem,
        np.full(4, 2e-8),
        -0.2,
        0.3,
        solver(),
        inner_preconditioner="jacobi",
        trial_policy="backtrack",
        max_iterations=1,
        checkpoint=lambda row: checkpoints.append(deepcopy(row)),
    )
    trial = result.history[0]["attempts"][0]["trials"][-1]
    assert trial["step"] == 0.5 and trial["status"] == "roundoff_kkt_decrease"
    assert checkpoints[-1]["radius_K"] == max(
        POLICY["minimum_radius_K"], 2 * trial["temperature_step_K"]
    )
    assert max(result.kkt.values()) <= 1e-8


@pytest.mark.parametrize("accuracy", ["strict", "adaptive"])
def test_backtracking_solves_small_coupled_equations(accuracy):
    problem = small_coupled_problem([0.2, 0.35])
    desired = np.linspace(-0.2, 0.4, problem.size)
    result = minimize_trust(
        problem,
        desired,
        -0.05,
        0.15,
        solver(),
        inner_preconditioner="jacobi",
        trial_policy="backtrack",
        accuracy=accuracy,
        max_iterations=80,
    )
    assert result.status == "converged"
    assert max(result.kkt.values()) <= 1e-8
    value, gradient = problem.objective_gradient(result.evaluation, desired)
    assert value == result.objective
    np.testing.assert_allclose(gradient, result.gradient)


def test_checkpoint_rejects_policy_change_and_resumes_same_policy():
    problem, desired, saved = QuadraticProblem(maximum_step=0.1), np.ones(4), []

    class Interrupted(Exception):
        pass

    def checkpoint(payload):
        if payload["iteration"] == 1:
            saved.append(deepcopy(payload))
            raise Interrupted

    arguments = dict(inner_preconditioner="jacobi", trial_policy="backtrack")
    with pytest.raises(Interrupted):
        minimize_trust(problem, desired, -0.2, 0.3, solver(), checkpoint=checkpoint, **arguments)
    with pytest.raises(ValueError, match="trial policy"):
        minimize_trust(problem, desired, -0.2, 0.3, solver(), resume=saved[0])
    result = minimize_trust(problem, desired, -0.2, 0.3, solver(), resume=saved[0], **arguments)
    assert result.status == "converged"
    np.testing.assert_allclose(result.evaluation.state, 0.3)


def test_legacy_policy_uses_one_trial_per_quadratic(monkeypatch):
    import deflation_example.coupled_trust as trust

    monkeypatch.setitem(trust.POLICY, "minimum_radius_K", 0.2)
    result = minimize_trust(
        QuadraticProblem(0),
        np.ones(4),
        -0.2,
        0.3,
        solver(),
        inner_preconditioner="jacobi",
    )
    assert len(result.history[0]["attempts"][0]["trials"]) == 1


@pytest.mark.parametrize("kw", [{"trial_policy": "unknown"}, {"trial_callback": 1}])
def test_invalid_trial_configuration(kw):
    with pytest.raises(ValueError):
        minimize_trust(QuadraticProblem(), np.ones(4), -0.2, 0.3, solver(), **kw)
