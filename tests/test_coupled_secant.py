"""Secant-model positivity and agreement with explicit BFGS updates."""

import numpy as np
import pytest
from scipy.sparse.linalg import aslinearoperator

from deflation_example.coupled_derivatives import GaussNewtonOperator
from deflation_example.coupled_secant import SecantGaussNewton
from deflation_example.coupled_optimizer import minimize_coupled
from test_coupled_derivatives import small_coupled_problem
from test_coupled_optimizer import solver


def test_damped_secants_match_explicit_positive_definite_updates():
    rng = np.random.default_rng(818)
    J = rng.normal(size=(15, 15))
    base = GaussNewtonOperator(aslinearoperator(J), np.linspace(1, 2, 15), 0.1)
    pairs = [(rng.normal(size=15), rng.normal(size=15)) for _ in range(5)]
    model = SecantGaussNewton(base, pairs)
    explicit = base @ np.eye(15)
    for s, y in pairs:
        q = explicit @ s
        sq, sy = s @ q, s @ y
        weight = 1.0 if sy >= 0.2 * sq else 0.8 * sq / (sq - sy)
        z = weight * y + (1 - weight) * q
        explicit -= np.outer(q, q) / sq
        explicit += np.outer(z, z) / (s @ z)
    actual = model @ np.eye(15)
    np.testing.assert_allclose(actual, explicit, atol=1e-12)
    assert np.linalg.eigvalsh(actual)[0] > 0
    np.testing.assert_allclose(model @ pairs[0][0], actual @ pairs[0][0], atol=1e-12)
    indices = np.arange(0, 15, 2)
    np.testing.assert_allclose(
        model.restrict(indices) @ np.eye(len(indices)), actual[np.ix_(indices, indices)], atol=1e-12
    )
    zero = SecantGaussNewton(base, [(np.zeros(15), np.zeros(15))])
    assert not zero.corrections


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_secant_and_gauss_newton_reach_the_same_verified_solution(steps):
    problem = small_coupled_problem(steps)
    desired = np.linspace(-0.1, 0.3, problem.size)
    plain = minimize_coupled(problem, desired, -0.05, 0.15, solver(), qp_tolerance=1e-11)
    secant = minimize_coupled(
        problem,
        desired,
        -0.05,
        0.15,
        solver(),
        qp_tolerance=1e-11,
        secant_memory=5,
        backtracking="quadratic",
    )
    assert plain.status == secant.status == "converged"
    assert max(secant.kkt.values()) <= 1e-8
    np.testing.assert_allclose(secant.evaluation.state, plain.evaluation.state, atol=2e-8)
