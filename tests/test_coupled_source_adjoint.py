"""Independent adjoints include the consistent source action before time reversal."""

import numpy as np
import pytest

from test_coupled_derivatives import small_coupled_problem


@pytest.mark.parametrize("steps", [None, [0.2, 0.35, 0.15]])
@pytest.mark.parametrize("consistent", [False, True])
def test_independent_source_and_momentum_adjoints_match_gradient(steps, consistent):
    problem = small_coupled_problem(steps, consistent=consistent, uniform_capacity=True)
    state = np.linspace(0.03, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    report = problem.verify_adjoint(evaluation, np.linspace(0.12, 0.2, problem.size))
    assert report["gradient_relative_difference"] < 1e-10
    assert report["gradient_weight_normalized_difference"] < 1e-10
    assert report["maximum_source_adjoint_relative_residual"] < 1e-11
    assert report["maximum_momentum_adjoint_relative_residual"] < 1e-11


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_independent_verifier_detects_corrupted_source_transpose(steps):
    problem = small_coupled_problem(steps, consistent=True, uniform_capacity=True)
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    factor = evaluation.jacobian.source_factors[0]

    class WrongTranspose:
        def solve(self, rhs, trans="N"):
            return factor.solve(rhs, trans=trans) * (2 if trans == "T" else 1)

    evaluation.jacobian.source_factors = (WrongTranspose(), *evaluation.jacobian.source_factors[1:])
    report = problem.verify_adjoint(evaluation, np.full(problem.size, 0.2))
    assert report["gradient_weight_normalized_difference"] > 1e-6
    assert report["maximum_source_adjoint_relative_residual"] < 1e-11


def test_near_zero_gradient_uses_outer_normalization_for_agreement():
    problem = small_coupled_problem([0.2, 0.35], consistent=True, uniform_capacity=True)
    state = np.full(problem.size, 0.06)
    evaluation = problem.evaluate(state)
    desired = (
        state
        + problem.alpha
        * (evaluation.jacobian.T @ (problem.weights * evaluation.control))
        / problem.weights
    )
    report = problem.verify_adjoint(evaluation, desired)
    assert report["gradient_weight_normalized_difference"] < 1e-10


@pytest.mark.parametrize(
    "field",
    [
        "maximum_momentum_adjoint_relative_residual",
        "maximum_source_adjoint_relative_residual",
        "gradient_weight_normalized_difference",
    ],
)
@pytest.mark.parametrize("value", [1e-6, np.nan, np.inf, -1.0, None])
def test_adjoint_gate_requires_every_independent_check(field, value):
    from deflation_example.coupled_optimize import adjoint_acceptance

    report = dict(
        maximum_momentum_adjoint_relative_residual=1e-12,
        maximum_source_adjoint_relative_residual=1e-12,
        gradient_weight_normalized_difference=1e-12,
    )
    assert adjoint_acceptance(report)
    if value is None:
        del report[field]
    else:
        report[field] = value
    assert not adjoint_acceptance(report)
