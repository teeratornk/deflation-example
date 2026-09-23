"""Energy-metric Krylov extraction uses the complete, unchanged reference domain."""

import numpy as np
import pytest
from scipy import linalg
from scipy.sparse.linalg import aslinearoperator

from deflation_example.coupled_krylov_reference import krylov_reference
from deflation_example.coupled_preconditioned_reference import end_indices


def test_full_krylov_space_recovers_generalized_eigenvectors():
    rng = np.random.default_rng(83)
    A, M = rng.normal(size=(12, 12)), rng.normal(size=(12, 12))
    H, K = A.T @ A + 2 * np.eye(12), M.T @ M + np.eye(12)
    reference = krylov_reference(aslinearoperator(H), lambda x: K @ x, H.diagonal(), 5, steps=12)
    values = np.asarray(reference.description["ritz_values"])
    expected = linalg.eigvalsh(H @ K @ H, H)[end_indices(12)[:5]]
    np.testing.assert_allclose(values, expected, rtol=1e-9)
    np.testing.assert_allclose(K @ H @ reference.basis, reference.basis * values, atol=1e-9)
    np.testing.assert_allclose(reference.basis.T @ H @ reference.basis, np.eye(5), atol=1e-10)
    assert max(reference.description["relative_ritz_residuals_H_norm"]) < 1e-9
    np.testing.assert_array_equal(reference.restrict(np.array([0, 6])), reference.basis[[0, 6]])


def test_invariant_subspace_stops_without_replacement():
    reference = krylov_reference(aslinearoperator(np.eye(9)), lambda x: x, np.ones(9), 8)
    assert reference.rank == 1
    assert reference.description["termination"] == "invariant_subspace"
    assert reference.description["ritz_values"] == pytest.approx([1])


def test_rank_prefixes_are_nested_and_deterministic():
    H = aslinearoperator(np.diag(np.arange(1.0, 17.0)))
    a = krylov_reference(H, lambda x: x, np.ones(16), 3, steps=12)
    b = krylov_reference(H, lambda x: x, np.ones(16), 7, steps=12)
    np.testing.assert_allclose(a.basis, b.basis[:, :3], atol=1e-12)
    assert a.description["independent_candidates"] == 12


@pytest.mark.parametrize("defect", ["shape", "negative", "nonfinite", "asymmetric"])
def test_invalid_actions_are_rejected(defect):
    H = aslinearoperator(np.diag(np.arange(1.0, 7.0)))
    K = np.eye(6)
    if defect == "asymmetric":
        K[0, 1] = 1
    def inverse(x):
        if defect == "shape":
            return x[:-1]
        if defect == "negative":
            return -x
        if defect == "nonfinite":
            return x * np.nan
        return K @ x
    with pytest.raises(ValueError):
        krylov_reference(H, inverse, np.ones(6), 3, steps=6)
