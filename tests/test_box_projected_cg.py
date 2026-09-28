import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize

from deflation_example.box_projected_cg import box_projected_cg, projected_search
from deflation_example.coupled_optimizer import box_kkt, box_quadratic
from deflation_example.coupled_qp_globalization import validate_reconstruction
from deflation_example.study_solvers import StudySolver


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("direction_rtol", [1e-2, 1e-4, 1e-11])
def test_feasible_descent_matches_independent_box_optimizer(seed, direction_rtol):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(16, 16))
    H = sparse.csr_matrix(A.T @ A + np.eye(16))
    g = rng.standard_normal(16)
    lo, hi = np.full(16, -0.1), np.full(16, 0.15)
    solver = StudySolver("jacobi", rtol=direction_rtol, maxiter=1000, residual_policy="refine")
    try:
        result = box_projected_cg(H, g, H.diagonal(), lo, hi, solver, tolerance=1e-8)
    finally:
        solver.close()
    expected = minimize(
        lambda x: (0.5 * x @ (H @ x) + g @ x, H @ x + g),
        np.zeros(16),
        jac=True,
        bounds=list(zip(lo, hi)),
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 300},
    )
    assert result.status == "converged", (result.status, result.kkt)
    assert expected.success
    assert max(result.kkt.values()) <= 1e-8
    np.testing.assert_allclose(result.x, expected.x, atol=2e-6)
    assert np.all(result.x >= lo) and np.all(result.x <= hi)
    for row in result.history:
        for step in row["projected_steps"] + (
            [row["reduced_search"]] if "reduced_search" in row else []
        ):
            assert step["quadratic_change"] < 0


def test_converged_initial_state_and_zero_rhs_require_no_inner_solve():
    solver = StudySolver("jacobi", rtol=1e-10)
    result = box_projected_cg(sparse.eye(2), np.zeros(2), np.ones(2), -1, 1, solver)
    assert result.status == "converged" and result.history == []
    assert solver.kernel_calls == 0
    solver.close()


def test_wrong_sign_bound_multipliers_release_both_bounds():
    solver = StudySolver("jacobi", rtol=1e-10)
    result = box_projected_cg(
        sparse.eye(2),
        np.array([-0.2, 0.3]),
        np.ones(2),
        -1,
        1,
        solver,
        initial=np.array([-1.0, 1.0]),
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, [0.2, -0.3])
    solver.close()


def test_search_failure_and_invalid_diagonal_remain_explicit():
    point, status = projected_search(sparse.eye(2), np.zeros(2), np.ones(2), np.ones(2), -1, 1)
    assert point is None and status["status"] == "projected_search_failed"
    solver = StudySolver("jacobi")
    with pytest.raises(ValueError, match="positive diagonal"):
        box_projected_cg(sparse.eye(2), np.ones(2), np.array([0, 1]), -1, 1, solver)
    solver.close()


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
@pytest.mark.parametrize("direction_rtol", [1e-2, 1e-4, 1e-11])
def test_coupled_quadratic_matches_pdas_with_weighted_optimality(steps, direction_rtol):
    from deflation_example.coupled_derivatives import GaussNewtonOperator
    from deflation_example.coupled_frozen_preconditioner import frozen_preconditioner_factory
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem(steps, consistent=True, streamline_rule="smooth_p8")
    state = np.zeros(problem.size)
    evaluation = problem.evaluate(state)
    desired = np.linspace(-0.2, 0.4, problem.size)
    _, gradient = problem.objective_gradient(evaluation, desired)
    H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
    diagonal = problem.preconditioning_diagonal(evaluation)
    factory = frozen_preconditioner_factory(problem, evaluation)
    results = []
    for procedure in (box_quadratic, box_projected_cg):
        solver = StudySolver(
            "jacobi",
            rtol=1e-11 if procedure is box_quadratic else direction_rtol,
            maxiter=2000,
            residual_policy="refine",
        )
        try:
            results.append(
                procedure(
                    H,
                    gradient,
                    diagonal,
                    -0.05,
                    0.1,
                    solver,
                    tolerance=1e-8,
                    preconditioner_factory=factory,
                    kkt_evaluator=lambda x, g: box_kkt(x, g / problem.weights, -0.05, 0.1),
                )
            )
        finally:
            solver.close()
    assert all(r.status == "converged" for r in results), [(r.status, r.kkt) for r in results]
    np.testing.assert_allclose(results[0].x, results[1].x, atol=2e-8)
    assert max(results[1].kkt.values()) <= 1e-8


def test_budget_and_linear_failure_keep_the_feasible_state_and_fresh_kkt():
    from types import SimpleNamespace

    H = sparse.csr_matrix([[2.0, 1.0], [1.0, 3.0]])
    g = np.array([-0.2, 0.1])

    class FailedSolver:
        rtol = 1e-10
        stop_requested = None

        def solve(self, B, rhs, indices, initial):
            return SimpleNamespace(x=np.zeros_like(rhs), status="iteration_cap", iterations=1), {}

    failed = box_projected_cg(H, g, H.diagonal(), -1, 1, FailedSolver())
    assert failed.status == "linear_iteration_cap"
    assert failed.kkt == box_kkt(failed.x, H @ failed.x + g, -1, 1)
    assert np.all(np.abs(failed.x) <= 1)
    solver = StudySolver("jacobi")
    solver.stop_requested = lambda: True
    try:
        stopped = box_projected_cg(H, g, H.diagonal(), -1, 1, solver)
        assert stopped.status == "budget_exhausted"
        assert stopped.kkt == box_kkt(stopped.x, H @ stopped.x + g, -1, 1)
    finally:
        solver.close()


def test_reconstruction_requires_the_recorded_gradient_and_metric():
    arrays = {"gradient": np.arange(3.0), "diagonal": np.ones(3)}
    differences = validate_reconstruction(arrays, arrays["gradient"], arrays["diagonal"])
    assert max(differences.values()) == 0
    with pytest.raises(ValueError, match="differ from the trace"):
        validate_reconstruction(arrays, arrays["gradient"] + 0.01, arrays["diagonal"])
    with pytest.raises(ValueError, match="differ from the trace"):
        validate_reconstruction(arrays, arrays["gradient"], arrays["diagonal"] * 2)
