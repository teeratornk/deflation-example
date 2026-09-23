"""Check the actual inverse-preconditioned Ritz pencil and unchanged full support."""

import numpy as np
import pytest
from scipy import linalg
from scipy.sparse.linalg import aslinearoperator

from deflation_example.coupled_preconditioned_reference import end_indices, preconditioned_reference
from deflation_example.study_solvers import ArrayReference


def test_full_candidates_recover_eigenvectors_of_inverse_times_operator():
    rng = np.random.default_rng(113)
    A, M = rng.normal(size=(12, 12)), rng.normal(size=(12, 12))
    H, K = A.T @ A + np.eye(12), M.T @ M + 2 * np.eye(12)
    ref = preconditioned_reference(
        ArrayReference(np.eye(12), {}),
        aslinearoperator(H),
        lambda x: K @ x,
        5,
        enrich=False,
        chunk=3,
    )
    Z = ref.basis
    values = np.asarray(ref.description["ritz_values"])
    np.testing.assert_allclose(K @ H @ Z, Z * values, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(Z.T @ H @ Z, np.eye(5), atol=1e-11)
    expected = linalg.eigvalsh(H @ K @ H, H)
    np.testing.assert_allclose(values, expected[end_indices(12)[:5]], rtol=1e-10)
    np.testing.assert_array_equal(ref.restrict(np.array([1, 7])), Z[[1, 7]])


def test_enrichment_changes_span_and_respects_block_limit():
    H = np.diag(np.arange(1.0, 9.0))
    raw = np.random.default_rng(14).normal(size=(8, 2))
    calls = []

    def action(x):
        calls.append(x.shape[1])
        return H @ x

    ref = preconditioned_reference(
        ArrayReference(raw, {}), aslinearoperator(H), lambda x: x, 4, chunk=1, block_action=action
    )
    assert ref.rank == 4
    assert ref.description["independent_candidates"] == 4
    assert max(calls) == 1
    assert np.linalg.matrix_rank(np.column_stack((raw, ref.basis))) == 4


@pytest.mark.parametrize("zero", [False, True])
def test_rank_loss_preserved_without_replacement(zero):
    raw = np.zeros((6, 4))
    if not zero:
        raw[:, :2] = np.arange(6.0)[:, None]
    ref = preconditioned_reference(
        ArrayReference(raw, {}), aslinearoperator(np.eye(6)), lambda x: x, 4
    )
    assert ref.rank == (0 if zero else 1)


def test_fixed_operator_and_inverse_symmetry_required():
    H = np.eye(6)
    K = np.eye(6)
    K[0, 1] = 0.1
    with pytest.raises(ValueError, match="symmetry"):
        preconditioned_reference(
            ArrayReference(H, {}), aslinearoperator(H), lambda x: K @ x, 2, enrich=False
        )
    with pytest.raises(ValueError, match="positive definite"):
        preconditioned_reference(
            ArrayReference(H, {}), aslinearoperator(H), lambda x: -x, 2, enrich=False
        )


def test_smaller_rank_is_nested_in_larger_reference():
    H = np.diag(np.arange(1.0, 9.0))
    candidates = ArrayReference(np.eye(8), {})
    full = preconditioned_reference(candidates, aslinearoperator(H), lambda x: x, 6, enrich=False)
    small = preconditioned_reference(candidates, aslinearoperator(H), lambda x: x, 3, enrich=False)
    np.testing.assert_array_equal(small.basis, full.basis[:, :3])


def test_recycling_retains_coarse_vectors_and_uses_the_attached_inverse():
    from deflation_example.recycling import RecycleSpace

    H = np.diag(np.arange(1.0, 9.0))
    K = np.diag(np.geomspace(0.1, 3, 8))
    A = aslinearoperator(H)
    A.preconditioner = lambda x: K @ x
    history = RecycleSpace(3, 6)
    history.begin(np.arange(8))
    for i in range(2, 8):
        history.capture(np.eye(8)[:, i])
    report = history.finish(A, H.diagonal(), np.eye(8)[:, :2], "converged")
    assert report["existing_columns"] == 2
    assert report["candidate_columns"] == 8
    assert report["policy"].endswith("preconditioned-ritz-v1")
    np.testing.assert_allclose(
        K @ H @ history.basis, history.basis * report["ritz_values"], atol=1e-11
    )
