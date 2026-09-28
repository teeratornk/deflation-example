import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize

from deflation_example.box_projected_cg import box_projected_cg, projected_search
from deflation_example.study_solvers import StudySolver


@pytest.mark.parametrize("seed", range(6))
def test_feasible_descent_matches_independent_box_optimizer(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(16, 16))
    H = sparse.csr_matrix(A.T @ A + np.eye(16))
    g = rng.standard_normal(16)
    lo, hi = np.full(16, -0.1), np.full(16, 0.15)
    solver = StudySolver("jacobi", rtol=1e-11, maxiter=1000, residual_policy="refine")
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
