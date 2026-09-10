"""Cached BZ preserves the projected solve and removes an operator action per step."""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from deflation_example.solvers import deflated_cg


def test_cached_projection_matches_direct_action_and_saves_applications():
    rng = np.random.default_rng(45)
    C = rng.normal(size=(25, 25))
    A = C.T @ C + np.eye(25)
    basis = np.linalg.eigh(A)[1][:, :4]
    b = rng.normal(size=25)
    outcomes = []
    for cached in (False, True):
        calls = []

        def apply(x):
            calls.append(x.shape)
            return A @ x

        operator = LinearOperator(A.shape, matvec=apply, matmat=apply, dtype=np.float64)
        solved = deflated_cg(operator, b, basis, np.diag(A), cache_operator_product=cached)
        assert solved.status == "converged" and solved.residual <= 1e-10
        np.testing.assert_allclose(solved.x, np.linalg.solve(A, b), rtol=1e-8, atol=1e-10)
        outcomes.append((solved, len(calls)))
    assert outcomes[1][1] < outcomes[0][1] - outcomes[1][0].iterations / 2
    zero = deflated_cg(A, b, diagonal=np.diag(A), cache_operator_product=True)
    assert zero.rank == 0 and zero.status == "converged"


@pytest.mark.gpu
def test_gpu_cached_product_parity_rank_zero_and_fallback():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from deflation_example.gpu import gpu_deflated_cg

    rng = np.random.default_rng(73)
    C = rng.normal(size=(30, 30))
    A = sparse.csr_matrix(C.T @ C + np.eye(30))
    b = rng.normal(size=30)
    basis = np.linalg.eigh(A.toarray())[1][:, :5]
    for supplied, limit in [(basis, 1e10), (None, 1e10), (basis, 1.01)]:
        for cached in (False, True):
            result, metrics = gpu_deflated_cg(
                A,
                b,
                supplied,
                A.diagonal(),
                cache_operator_product=cached,
                basis_backend="gpu_qr",
                condition_limit=limit,
            )
            assert result.status == "converged" and result.residual <= 1e-10
            np.testing.assert_allclose(result.x, np.linalg.solve(A.toarray(), b), atol=1e-9)
            assert metrics["operator_product_cached"] == bool(cached and result.rank)
            assert metrics["cached_operator_product_bytes"] == (
                30 * result.rank * 8 if cached else 0
            )
            assert sum(metrics["components_seconds"].values()) == pytest.approx(
                metrics["total_seconds"]
            )
