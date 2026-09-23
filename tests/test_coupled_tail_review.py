import numpy as np
import pytest

from deflation_example.coupled_tail_review import frozen_prefix_problem, one_sided_trial
from deflation_example.coupled_sequence import RestoredEvaluation
from test_coupled_derivatives import small_coupled_problem


@pytest.mark.parametrize("first", [1, 2])
def test_frozen_prefix_preserves_source_gradient_and_temporal_weights(first):
    full = small_coupled_problem([0.2, 0.35, 0.15], consistent=True, uniform_capacity=True)
    state = np.linspace(0.04, 0.1, full.size)
    evaluation = full.evaluate(state)
    desired = np.linspace(0.13, 0.2, full.size)
    tail = frozen_prefix_problem(full, state, evaluation.flows, first)
    offset = first * full.spatial_size
    saved = RestoredEvaluation(
        state[offset:],
        np.stack([f.velocity for f in evaluation.flows[first:]]),
        np.stack([f.pressure for f in evaluation.flows[first:]]),
    )
    actual = tail.evaluate(state[offset:], initial=saved)
    np.testing.assert_allclose(actual.control, evaluation.control[offset:], atol=1e-10)
    expected_gradient = full.objective_gradient(evaluation, desired)[1][offset:]
    actual_gradient = tail.objective_gradient(actual, desired[offset:])[1]
    np.testing.assert_allclose(actual_gradient, expected_gradient, atol=1e-10)
    np.testing.assert_array_equal(tail.weights, full.weights[offset:])
    np.testing.assert_array_equal(tail.physical_steps, full.physical_steps[first:])
    assert full.slabs == 3


def test_suffix_rejects_missing_predecessor():
    full = small_coupled_problem([0.2, 0.35])
    evaluation = full.evaluate(np.zeros(full.size))
    with pytest.raises(ValueError):
        frozen_prefix_problem(full, evaluation.state, evaluation.flows, 0)


def test_one_sided_branch_failure_is_retained_without_mutating_the_state(monkeypatch):
    from deflation_example.coupled_derivatives import StabilizationBranchError

    problem = small_coupled_problem([0.2, 0.35], consistent=True)
    evaluation = problem.evaluate(np.full(problem.size, 0.06))
    original = evaluation.state.copy()

    def switch(*args, **kwargs):
        raise StabilizationBranchError("test switch")

    monkeypatch.setattr(problem, "evaluate", switch)
    row = one_sided_trial(problem, evaluation, np.zeros(problem.size), 0, 0, 1e-6)
    assert row["status"] == "stabilization_branch_switch"
    np.testing.assert_array_equal(evaluation.state, original)
    with pytest.raises(ValueError, match="nonzero"):
        one_sided_trial(problem, evaluation, np.zeros(problem.size), 0, 0, 0)


def test_explicit_rule_override_preserves_original_configuration():
    from deflation_example.coupled_tail_review import review_configuration

    original = {"alpha": 1e-14, "streamline_rule": "hard_min", "inner_tolerance": 1e-10}
    assert review_configuration(original, None) == original
    changed = review_configuration(original, "smooth_p8")
    assert changed == {**original, "streamline_rule": "smooth_p8"}
    assert original["streamline_rule"] == "hard_min"
    with pytest.raises(ValueError, match="coefficients"):
        review_configuration(original, "invalid")
