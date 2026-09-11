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


@pytest.mark.parametrize("correction_factor", [1.0, 0.0, -1.0])
def test_fixed_mask_error_solve_resolves_norm_mismatch_and_retains_best_state(correction_factor):
    from deflation_example.solvers import LinearResult

    class InexactSolver:
        rtol = 1e-6

        def __init__(self):
            self.calls = []

        def solve(self, B, rhs, indices, initial=None):
            self.calls.append((rhs.copy(), initial.copy()))
            value = rhs * (1 - 1e-8) if len(self.calls) == 1 else correction_factor * rhs
            return LinearResult(value, 1, 0.0, "converged"), {}

    inner = InexactSolver()
    H = sparse.eye(3, format="csr")
    g = np.array([-1.0, 4.0, -4.0])
    result = box_quadratic(H, g, np.ones(3), -2, 2, inner, tolerance=1e-10)
    assert len(inner.calls) == 2
    assert inner.calls[1][0][0] == pytest.approx(1e-8, abs=1e-15)
    np.testing.assert_array_equal(inner.calls[1][1], [0.0])
    assert result.history[1]["linear_equation"] == "correction"
    assert result.history[1]["linear_residual"] <= inner.rtol
    if correction_factor == 1:
        assert result.status == "converged"
        np.testing.assert_array_equal(result.x, [1.0, -2.0, 2.0])
    else:
        assert result.status == "fixed_mask_correction_stagnation"
        assert not result.history[-1]["candidate_retained"]
        np.testing.assert_array_equal(result.x, [1 - 1e-8, -2.0, 2.0])
    assert result.kkt == box_kkt(result.x, H @ result.x + g, -2, 2, 4.0)


def test_fixed_mask_correction_cap_keeps_unsatisfied_qp_visible():
    from deflation_example.solvers import LinearResult

    class InexactSolver:
        rtol = 1e-6

        def solve(self, B, rhs, indices, initial=None):
            return LinearResult(rhs * (1 - 1e-8), 1, 1e-8, "converged"), {}

    result = box_quadratic(
        sparse.eye(1, format="csr"),
        np.array([-1.0]),
        np.ones(1),
        -2,
        2,
        InexactSolver(),
        tolerance=1e-10,
        fixed_mask_corrections=0,
    )
    assert result.status == "fixed_mask_correction_cap"
    assert len(result.history) == 1
    assert result.kkt["stationarity"] > 1e-10


def test_line_search_retains_and_backtracks_a_stabilization_branch_switch(monkeypatch):
    from deflation_example.coupled_derivatives import StabilizationBranchError

    problem = small_coupled_problem()
    actual_evaluate = problem.evaluate
    attempted = []

    def evaluate(state, initial=None):
        if initial is not None and not attempted:
            attempted.append(True)
            raise StabilizationBranchError("test branch switch")
        return actual_evaluate(state, initial=initial)

    monkeypatch.setattr(problem, "evaluate", evaluate)
    result = minimize_coupled(problem, np.linspace(-0.1, 0.3, problem.size), -0.05, 0.15, solver())
    assert result.status == "converged"
    trials = result.history[0]["attempts"][0]["trials"]
    assert trials[0]["status"] == "stabilization_branch_switch"
    assert trials[1]["step"] == 0.5 * trials[0]["step"]


@pytest.mark.parametrize("backtracking", ["halving", "quadratic"])
@pytest.mark.parametrize("objective_offset", [0.0, 1e16])
def test_safeguarded_backtracking_reaches_the_same_nonlinear_stationary_point(
    backtracking, objective_offset
):
    from types import SimpleNamespace
    from scipy.sparse.linalg import aslinearoperator

    class ScalarLeastSquares:
        size, alpha, weights = 1, 1.0, np.ones(1)

        def evaluate(self, x, initial=None):
            return SimpleNamespace(
                state=x.copy(),
                control=2 + x**2,
                jacobian=aslinearoperator(np.diag(2 * x)),
                seconds=0.0,
            )

        def objective_gradient(self, ev, desired):
            return (
                float(0.5 * np.sum((ev.state - desired) ** 2 + ev.control**2)) + objective_offset,
                ev.state - desired + 2 * ev.state * ev.control,
            )

        def preconditioning_diagonal(self, ev, damping=0):
            return 1 + 4 * ev.state**2 + damping

        def objective_difference(self, candidate, reference, desired):
            return float(
                np.sum(
                    (candidate.state - reference.state)
                    * (0.5 * (candidate.state + reference.state) - desired)
                    + 0.5
                    * (candidate.control - reference.control)
                    * (candidate.control + reference.control)
                )
            )

    problem = ScalarLeastSquares()
    result = minimize_coupled(
        problem,
        np.ones(1),
        -0.5,
        1.0,
        solver(),
        backtracking=backtracking,
        tolerance=1e-9,
        max_iterations=100,
        qp_tolerance=1e-12,
    )
    assert result.status == "converged"
    assert abs(2 * result.evaluation.state[0] ** 3 + 5 * result.evaluation.state[0] - 1) < 1e-8
    first_trials = result.history[0]["attempts"][0]["trials"]
    if backtracking == "quadratic":
        assert first_trials[1]["step"] == pytest.approx(1 / 6)
    for outer in result.history:
        for attempt in outer["attempts"]:
            for trial in attempt["trials"]:
                if trial["status"] == "roundoff_kkt_decrease":
                    assert max(outer["kkt"].values()) <= 1e-4
                    assert trial["maximum_kkt"] <= 0.9 * max(outer["kkt"].values())
                    assert trial["objective_change"] <= trial["roundoff_allowance"]
                    assert (
                        abs(trial["step"] * attempt["directional_derivative"])
                        <= trial["roundoff_allowance"]
                    )
