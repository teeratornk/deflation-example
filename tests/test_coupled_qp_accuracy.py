"""Recorded-equation matching and candidate/partition accounting."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from deflation_example.coupled_optimizer import box_kkt
from deflation_example.coupled_qp_accuracy import partition, replay_system
from deflation_example.study_solvers import StudySolver


def data():
    H = sparse.csr_matrix([[4.0, -1.0, 0.0], [-1.0, 3.0, -1.0], [0.0, -1.0, 2.0]])
    g = np.array([-4.0, -6.0, 1.0])
    d = H.diagonal()
    lo, hi = np.full(3, -1.0), np.ones(3)
    mask = np.array([0, 0, -1], dtype=np.int8)
    rhs = np.array([4.0, 5.0])
    return H, g, d, lo, hi, mask, rhs, np.zeros(2), np.array([0.1, 0.2])


@pytest.mark.parametrize("rtol", [1e-4, 1e-6, 1e-8, 1e-10])
def test_matched_replay_does_not_clip_the_candidate_or_mutate_inputs(rtol):
    args = data()
    originals = [x.copy() for x in args[1:]]
    solver = StudySolver("jacobi", rtol=rtol, maxiter=100, residual_policy="refine")
    try:
        report, fields = replay_system(
            *args,
            solver,
            lambda x, g: box_kkt(x, g, -1, 1),
        )
    finally:
        solver.close()
    exact = np.linalg.solve(args[0].toarray()[:2, :2], args[6])
    np.testing.assert_allclose(fields["candidate"], [*exact, -1.0])
    assert report["linear_verified"]
    assert report["candidate_kkt"]["primal_absolute"] > 0
    assert report["newly_active"] == 2
    assert report["rhs_relative_difference"] == 0
    for before, after in zip(originals, args[1:], strict=True):
        np.testing.assert_array_equal(before, after)


def test_changed_right_hand_side_is_rejected_before_solving():
    args = list(data())
    args[6] = args[6] + 0.01
    with pytest.raises(ValueError, match="right-hand side differs"):
        replay_system(*args, None, lambda x, g: box_kkt(x, g, -1, 1))


def test_already_converged_initial_guess_matches_returned_residual():
    args = list(data())
    args[7] = np.linalg.solve(args[0].toarray()[:2, :2], args[6])
    solver = StudySolver("jacobi", rtol=1e-10, maxiter=100, residual_policy="refine")
    try:
        report, fields = replay_system(*args, solver, lambda x, g: box_kkt(x, g, -1, 1))
    finally:
        solver.close()
    assert report["iterations"] == 0
    assert report["original_residual"] < 1e-10
    np.testing.assert_array_equal(fields["candidate"][:2], args[7])


def test_zero_right_hand_side_and_invalid_partition():
    H = sparse.eye(2, format="csr")
    solver = StudySolver("jacobi", rtol=1e-10, maxiter=100, residual_policy="refine")
    args = (
        H,
        np.zeros(2),
        np.ones(2),
        -np.ones(2),
        np.ones(2),
        np.zeros(2, dtype=int),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
    )
    try:
        report, _ = replay_system(*args, solver, lambda x, g: box_kkt(x, g, -1, 1))
        assert report["linear_verified"] and report["original_residual"] == 0
        bad = list(args)
        bad[5] = np.array([0, 2])
        with pytest.raises(ValueError, match="bound partition"):
            replay_system(*bad, solver, lambda x, g: box_kkt(x, g, -1, 1))
    finally:
        solver.close()


def test_failed_solve_keeps_failure_and_recomputes_its_residual():
    class FailedSolver:
        rtol = 1e-10

        def solve(self, B, rhs, indices, initial):
            return SimpleNamespace(x=np.zeros_like(rhs), status="iteration_cap", iterations=1), {}

    report, fields = replay_system(*data(), FailedSolver(), lambda x, g: box_kkt(x, g, -1, 1))
    assert not report["linear_verified"]
    assert report["status"] == "iteration_cap"
    assert report["original_residual"] == 1
    np.testing.assert_array_equal(fields["candidate"], [0, 0, -1])


def test_partition_distinguishes_upper_from_lower_bounds():
    np.testing.assert_array_equal(
        partition(np.zeros(3), np.array([2.0, 0.0, -2.0]), np.ones(3), -1, 1),
        [-1, 0, 1],
    )
