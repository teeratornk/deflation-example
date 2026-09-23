import numpy as np
import pytest

from deflation_example.coupled_review_derivatives import direction_checks
from test_coupled_derivatives import small_coupled_problem


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_checkpoint_directional_checks_match_consistent_discretization(steps):
    problem = small_coupled_problem(steps, consistent=True, uniform_capacity=True)
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    direction = np.random.default_rng(33).normal(size=problem.size)
    progress = []
    report = direction_checks(
        problem,
        evaluation,
        np.full(problem.size, 0.12),
        direction,
        lambda r: progress.append(len(r["rows"])),
    )
    assert progress == [1, 2, 3, 4]
    assert report["tangent_transpose_relative_difference"] < 1e-10
    assert all(r["status"] == "evaluated" for r in report["rows"])
    assert min(r["control_derivative_relative_error"] for r in report["rows"]) < 1e-6
    assert min(r["objective_derivative_relative_error"] for r in report["rows"]) < 1e-6


def test_invalid_direction_is_rejected():
    problem = small_coupled_problem()
    evaluation = problem.evaluate(np.zeros(problem.size))
    with pytest.raises(ValueError, match="nonzero"):
        direction_checks(problem, evaluation, np.zeros(problem.size), np.zeros(problem.size))


def test_directional_switch_does_not_discard_later_perturbations(monkeypatch):
    from deflation_example.coupled_derivatives import StabilizationBranchError

    problem = small_coupled_problem(consistent=True, uniform_capacity=True)
    evaluation = problem.evaluate(np.full(problem.size, 0.06))
    original, calls = problem.evaluate, 0

    def switch_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StabilizationBranchError("test switch")
        return original(*args, **kwargs)

    monkeypatch.setattr(problem, "evaluate", switch_once)
    report = direction_checks(
        problem, evaluation, np.full(problem.size, 0.1), np.ones(problem.size)
    )
    assert report["rows"][0]["status"] == "stabilization_branch_switch"
    assert [r["status"] for r in report["rows"][1:]] == ["evaluated"] * 3
