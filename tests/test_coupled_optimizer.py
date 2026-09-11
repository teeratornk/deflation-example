"""Independent box-QP and nonlinear-optimizer checks on complete coupled systems."""

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize

from deflation_example.coupled_optimizer import box_kkt, box_quadratic, minimize_coupled
from deflation_example.study_solvers import ArrayReference, StudySolver
from test_coupled_derivatives import small_coupled_problem


def solver(method="jacobi", reference=None, rank=3):
    return StudySolver(
        method,
        reference=reference,
        rank=rank,
        window=6,
        rtol=1e-11,
        cg_factor=0.1,
        maxiter=1000,
        residual_policy="refine",
    )


def test_box_qp_releases_both_bounds_and_matches_independent_optimizer():
    rng = np.random.default_rng(75)
    A = rng.normal(size=(12, 12))
    H = sparse.csr_matrix(A.T @ A + np.eye(12))
    g = rng.normal(size=12) * 3
    lower, upper = np.full(12, -0.08), np.full(12, 0.1)
    actual = box_quadratic(H, g, H.diagonal(), lower, upper, solver(), tolerance=1e-10)
    expected = minimize(
        lambda x: (0.5 * x @ (H @ x) + g @ x, H @ x + g),
        np.zeros(12),
        jac=True,
        bounds=list(zip(lower, upper)),
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 200},
    )
    assert expected.success
    assert actual.status == "converged"
    assert max(actual.kkt.values()) < 1e-10
    np.testing.assert_allclose(actual.x, expected.x, atol=1e-7)
    assert np.any(actual.x == lower) and np.any(actual.x == upper)
    # The zero-start sign test activates an index subsequently released.
    initially_active = (g / H.diagonal() >= -lower) | (g / H.diagonal() <= -upper)
    assert np.any(initially_active & (actual.x > lower) & (actual.x < upper))


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_complete_coupled_optimizer_matches_independent_slsqp(steps):
    problem = small_coupled_problem(steps)
    desired = np.linspace(-0.5, 0.8, problem.size)
    lower, upper = -0.04, 0.15
    result = minimize_coupled(
        problem,
        desired,
        lower,
        upper,
        solver(),
        tolerance=1e-8,
        qp_tolerance=1e-11,
        max_iterations=60,
    )

    def objective(x):
        return problem.objective_gradient(problem.evaluate(x), desired)

    expected = minimize(
        objective,
        np.zeros(problem.size),
        jac=True,
        bounds=[(lower, upper)] * problem.size,
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 200},
    )
    assert expected.success
    assert result.status == "converged", (result.status, result.kkt)
    assert max(result.kkt.values()) <= 1e-8
    np.testing.assert_allclose(result.evaluation.state, expected.x, atol=2e-6)
    assert result.objective == pytest.approx(expected.fun, abs=1e-11)
    assert any(np.isclose(result.evaluation.state, lower))
    assert any(np.isclose(result.evaluation.state, upper))


@pytest.mark.parametrize("method", ["reference", "recycling"])
def test_three_inner_policies_solve_the_same_nonlinear_problem(method):
    problem = small_coupled_problem([0.2, 0.35])
    desired = np.linspace(-0.1, 0.3, problem.size)
    reference = ArrayReference(np.eye(problem.size)[:, :3], {"construction": "test"})
    plain = minimize_coupled(problem, desired, -0.05, 0.15, solver(), qp_tolerance=1e-11)
    other = minimize_coupled(
        problem, desired, -0.05, 0.15, solver(method, reference), qp_tolerance=1e-11
    )
    assert plain.status == other.status == "converged"
    np.testing.assert_allclose(other.evaluation.state, plain.evaluation.state, atol=2e-8)
    assert max(other.kkt.values()) <= 1e-8


def test_converged_initial_temperature_skips_all_qp_solves():
    problem = small_coupled_problem()
    initial = np.full(problem.size, 0.1)
    evaluation = problem.evaluate(initial)
    # Construct a target giving exactly zero reduced gradient at this temperature.
    target = (
        initial
        + problem.alpha
        * (evaluation.jacobian.T @ (problem.weights * evaluation.control))
        / problem.weights
    )
    result = minimize_coupled(problem, target, -0.05, 0.2, solver(), initial=initial)
    assert result.status == "converged"
    assert result.history == []
    np.testing.assert_array_equal(result.evaluation.state, initial)


def test_iteration_cap_preserves_state_gradient_and_residual_correspondence():
    problem = small_coupled_problem(feedback=0.03)
    target = np.linspace(-0.03, 0.18, problem.size)
    result = minimize_coupled(
        problem, target, -0.2, 0.4, solver(), max_iterations=1, tolerance=1e-13
    )
    value, gradient = problem.objective_gradient(result.evaluation, target)
    assert result.status == "nonlinear_iteration_cap"
    assert value == result.objective
    np.testing.assert_array_equal(gradient, result.gradient)
    assert len(result.history) == 1


def test_kkt_primal_is_absolute_and_multipliers_have_correct_signs():
    assert max(box_kkt(np.array([-1.0, 0.0, 1.0]), np.array([2.0, 0.0, -3.0]), -1, 1).values()) == 0
    assert box_kkt(np.array([1.01]), np.array([0.0]), -1, 1, scale=100)[
        "primal_absolute"
    ] == pytest.approx(0.01)


def test_nonfinite_optimality_is_a_failed_check():
    for gradient in (np.nan, np.inf, -np.inf):
        assert all(np.isinf(v) for v in box_kkt([0.0], [gradient], -1, 1).values())


def test_qp_verifies_candidate_on_last_allowed_active_set_step():
    result = box_quadratic(
        sparse.eye(2, format="csr"),
        np.array([-3.0, 2.0]),
        np.ones(2),
        -1.0,
        1.0,
        solver(),
        max_steps=1,
    )
    assert result.status == "converged"
    np.testing.assert_array_equal(result.x, [1.0, -1.0])
