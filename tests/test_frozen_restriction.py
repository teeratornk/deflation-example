"""Preassembled restrictions preserve the sparse preconditioner, not the exact tangent."""

import numpy as np
import pytest
from scipy import sparse

from deflation_example.coupled_frozen_preconditioner import frozen_preconditioner_factory
from deflation_example.mesh_control import WeightedReducedOperator
from deflation_example.sparse_restriction import PreassembledRestriction
from test_frozen_preconditioner import system


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("damping", [0.0, 0.3])
def test_frozen_factories_match_across_activation_release_and_damping(consistent, damping):
    problem, evaluation, _, _, _ = system(consistent)
    first = frozen_preconditioner_factory(problem, evaluation, damping)
    second = frozen_preconditioner_factory(problem, evaluation, damping, restriction="submatrix")
    all_indices = np.arange(problem.size)
    rng = np.random.default_rng(82)
    for indices in (all_indices, all_indices[::2], all_indices[1::3], all_indices):
        a, b = first(indices), second(indices)
        np.testing.assert_allclose(
            a.operator.toarray(), b.operator.toarray(), rtol=1e-13, atol=1e-15
        )
        rhs = rng.standard_normal(len(indices))
        np.testing.assert_allclose(a(rhs), b(rhs), rtol=1e-11, atol=1e-12)
    with pytest.raises(ValueError, match="product or submatrix"):
        frozen_preconditioner_factory(problem, evaluation, restriction="unknown")


def test_preassembled_restriction_matches_matrix_free_action_and_rejects_bad_indices():
    rng = np.random.default_rng(1)
    A = sparse.csr_matrix(rng.standard_normal((8, 8)))
    original = WeightedReducedOperator(A, np.linspace(0.5, 2, 8), 0.3)
    sliced = PreassembledRestriction(original)
    for indices in (np.arange(8), np.array([5, 1, 6]), np.array([], dtype=int)):
        np.testing.assert_allclose(
            original.restrict(indices).toarray(), sliced.restrict(indices).toarray()
        )
    for indices in (np.array([1, 1]), np.array([8]), np.array([-1]), np.array([0.5])):
        with pytest.raises(ValueError):
            sliced.restrict(indices)
    with pytest.raises(ValueError, match="assembled"):
        PreassembledRestriction(WeightedReducedOperator(A, np.ones(8), 0.3, False))


def test_sliced_operator_storage_is_not_mutable_through_assembled_result():
    original = WeightedReducedOperator(sparse.eye(4), np.ones(4), 0.2)
    sliced = PreassembledRestriction(original)
    exposed = sliced.assembled()
    exposed.data[:] = 99
    np.testing.assert_allclose(sliced.restrict(np.arange(4)).diagonal(), np.full(4, 1.2))


def test_setup_comparison_accounts_for_full_construction_and_every_mask():
    from deflation_example.coupled_preconditioner_setup import compare

    problem, evaluation, _, _, _ = system(True)
    rows = compare(problem, evaluation, [np.arange(problem.size), np.arange(problem.size)[::2]])
    assert len(rows) == 6 and all(r["verified"] for r in rows)
    assert [r["policy"] for r in rows] == [
        "product",
        "submatrix",
        "submatrix",
        "product",
        "product",
        "submatrix",
    ]
    for row in rows:
        assert len(row["masks"]) == 2
        assert row["setup_including_construction_seconds"] == pytest.approx(
            row["construction_seconds"] + sum(m["setup_seconds"] for m in row["masks"])
        )
