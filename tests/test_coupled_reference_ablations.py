"""Declared construction controls and loss of entries under sequential transfer."""

import numpy as np
import pytest
from scipy import sparse

from deflation_example.coupled_reference import SequentialReference, build_reference
from deflation_example.study_solvers import ArrayReference, StudySolver
from test_axisymmetric_flow import annular_rectangle
from test_coupled_reference import problem


def test_sequential_transfer_preserves_first_space_and_loses_released_entries():
    basis = np.arange(1, 13, dtype=float).reshape(6, 2)
    full = ArrayReference(basis.copy(), {})
    sequential = SequentialReference(full, 6)
    first, second, third = np.array([0, 1, 3, 5]), np.array([1, 2, 3]), np.array([0, 1, 2, 3, 5])
    sequential.begin_system(first)
    np.testing.assert_array_equal(sequential.restrict(first), full.restrict(first))
    sequential.begin_system(second)
    np.testing.assert_array_equal(
        sequential.restrict(second), np.vstack((basis[1], [0, 0], basis[3]))
    )
    sequential.begin_system(third)
    np.testing.assert_array_equal(
        sequential.restrict(third), np.vstack(([0, 0], basis[1], [0, 0], basis[3], [0, 0]))
    )
    np.testing.assert_array_equal(full.basis, basis)
    assert sequential.reference is None
    assert sequential.storage()["full_reference_bytes"] == 0
    assert sequential.storage()["previous_restricted_reference_bytes"] == 5 * 2 * 8
    copy = sequential.restrict(third)
    copy[:] = 0
    assert sequential.restrict(third).max() > 0


def test_sequential_rank_loss_is_preserved_without_replacement_directions():
    sequential = SequentialReference(ArrayReference(np.eye(4, 2), {}), 4)
    sequential.begin_system(np.array([0, 1]))
    assert np.linalg.matrix_rank(sequential.restrict(np.array([0, 1]))) == 2
    sequential.begin_system(np.array([2, 3]))
    sequential.begin_system(np.arange(4))
    assert np.linalg.matrix_rank(sequential.restrict(np.arange(4))) == 0
    with pytest.raises(ValueError, match="Begin"):
        sequential.restrict(np.array([0]))
    with pytest.raises(ValueError, match="dimension"):
        sequential.begin_system(np.array([4]))


def test_transfer_occurs_even_when_initial_residual_guard_skips_cg():
    sequential = SequentialReference(ArrayReference(np.ones((4, 1)), {}), 4)
    solver = StudySolver("reference", reference=sequential, rank=1, residual_policy="refine")
    for indices in (np.array([0, 1]), np.array([1, 2])):
        result, timing = solver.solve(sparse.eye(2), np.ones(2), indices, initial=np.ones(2))
        assert result.iterations == 0
        assert result.residual == 0
        assert timing["reference_transfer_seconds"] >= 0
        assert timing["total_seconds"] == pytest.approx(sum(timing["components_seconds"].values()))
    np.testing.assert_array_equal(sequential.restrict(np.array([1, 2])), [[1], [0]])


@pytest.mark.parametrize("construction", ["mode_dependent", "tensor"])
@pytest.mark.parametrize("spatial_policy", ["scaled_schur", "diffusion"])
def test_reference_construction_controls_match_requested_rank(construction, spatial_policy):
    p = problem(annular_rectangle(3), [0.2, 0.35])
    reference = build_reference(p, 2, construction=construction, spatial_policy=spatial_policy)
    assert reference.rank == 2
    assert reference.description["temporal_construction"] == construction
    assert reference.description["temporal_selection_metric"] == "jacobi"
    assert reference.description["construction"].startswith("generalized_" + spatial_policy)
