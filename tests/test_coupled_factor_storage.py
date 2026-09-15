"""Checkpointed factorizations preserve complete tangent and optimization equations."""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu

from deflation_example.coupled_factor_storage import RecomputedLU
from deflation_example.coupled_derivatives import GaussNewtonOperator
from deflation_example.coupled_optimize import load_problem
from test_coupled_derivatives import small_coupled_problem
from test_coupled_optimizer import solver


def matrix():
    rng = np.random.default_rng(3)
    value = rng.normal(size=(7, 7))
    return sparse.csc_matrix(value + 9 * np.eye(7))


@pytest.mark.parametrize("transpose", ["N", "T", "H"])
@pytest.mark.parametrize("columns", [None, 1, 3, 0])
def test_checkpoint_solve_preserves_vectors_blocks_and_transposes(transpose, columns):
    A = matrix()
    factor, checkpoint = splu(A), RecomputedLU(A)
    rhs = np.ones(7) if columns is None else np.ones((7, columns))
    result = checkpoint.solve(rhs, trans=transpose)
    np.testing.assert_allclose(result, factor.solve(rhs, trans=transpose), rtol=1e-14, atol=1e-14)
    assert checkpoint.factorizations == (0 if columns == 0 else 1)
    assert not hasattr(checkpoint, "factor")
    assert checkpoint.checkpoint_bytes == sum(a.nbytes for a in (A.data, A.indices, A.indptr))


def test_matrix_checkpoint_owns_immutable_inputs_and_has_explicit_failures():
    A = matrix()
    checkpoint = RecomputedLU(A)
    expected = checkpoint.solve(np.ones(7))
    A.data[:] = 0
    np.testing.assert_allclose(checkpoint.solve(np.ones(7)), expected)
    with pytest.raises(ValueError):
        checkpoint.solve(np.ones(6))
    with pytest.raises(ValueError):
        checkpoint.solve(np.full(7, np.nan))
    with pytest.raises(ValueError):
        checkpoint.solve(np.ones(7), trans="bad")
    with pytest.raises(ValueError):
        RecomputedLU(np.eye(3, dtype=complex) * 1j)
    with pytest.raises(RuntimeError):
        RecomputedLU(sparse.csc_matrix((2, 2))).solve(np.ones(2))


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_complete_derivatives_and_gradient_agree_with_retained_factors(steps):
    problem = small_coupled_problem(steps)
    state = np.linspace(0.04, 0.1, problem.size)
    desired = np.full(problem.size, 0.15)
    retained = problem.evaluate(state)
    problem.momentum_factor_policy = "recompute"
    checkpoint = problem.evaluate(state)
    directions = np.random.default_rng(9).normal(size=(problem.size, 3))
    for transpose in (False, True):
        A = retained.jacobian.T if transpose else retained.jacobian
        B = checkpoint.jacobian.T if transpose else checkpoint.jacobian
        np.testing.assert_allclose(B @ directions, A @ directions, rtol=1e-12, atol=1e-12)
    Hr = GaussNewtonOperator(retained.jacobian, problem.weights, problem.alpha)
    Hc = GaussNewtonOperator(checkpoint.jacobian, problem.weights, problem.alpha)
    np.testing.assert_allclose(Hc @ directions, Hr @ directions, rtol=1e-12, atol=1e-12)
    vr, gr = problem.objective_gradient(retained, desired)
    vc, gc = problem.objective_gradient(checkpoint, desired)
    assert vc == vr
    np.testing.assert_allclose(gc, gr, rtol=1e-12, atol=1e-12)
    report = problem.verify_adjoint(checkpoint, desired)
    assert report["maximum_momentum_adjoint_relative_residual"] < 1e-11
    assert all(isinstance(f, RecomputedLU) for f in checkpoint.jacobian.factors)


@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_three_solvers_retain_complete_optimization_accuracy_with_checkpoints(method):
    from deflation_example.coupled_optimizer import minimize_coupled
    from deflation_example.study_solvers import ArrayReference

    problem = small_coupled_problem([0.2, 0.35])
    desired = np.linspace(-0.1, 0.3, problem.size)
    reference = ArrayReference(np.eye(problem.size)[:, :3], {"construction": "test"})
    retained = minimize_coupled(
        problem, desired, -0.05, 0.15, solver(method, reference), qp_tolerance=1e-11
    )
    problem.momentum_factor_policy = "recompute"
    checkpoint = minimize_coupled(
        problem, desired, -0.05, 0.15, solver(method, reference), qp_tolerance=1e-11
    )
    assert retained.status == checkpoint.status == "converged"
    assert max(checkpoint.kkt.values()) <= 1e-8
    np.testing.assert_allclose(checkpoint.evaluation.state, retained.evaluation.state, atol=2e-8)


def test_gpu_checkpoint_policy_is_rejected_before_loading_data_or_cuda():
    from types import SimpleNamespace
    from deflation_example.coupled_cuda import CudaControlJacobian

    for device in ("cuda", "hybrid"):
        with pytest.raises(ValueError, match="CPU backend"):
            load_problem({"momentum_factor_policy": "recompute", "device": device})
    with pytest.raises(ValueError, match="retained momentum factors"):
        CudaControlJacobian(SimpleNamespace(factors=[RecomputedLU(matrix())]))


def test_memory_screen_compares_both_solve_policies_and_accounts_for_recomputation():
    from deflation_example.coupled_memory_screen import factor_inventory

    report = factor_inventory(matrix(), checkpoint_comparison=True)
    comparison = report["checkpoint_comparison"]
    assert comparison["all_actions_verified"]
    assert comparison["recomputed_factorizations"] == 8
    assert len(comparison["actions"]) == 16
    assert {row["policy"] for row in comparison["actions"]} == {"retained", "recompute"}
    assert comparison["matrix_checkpoint_bytes"] > 0
