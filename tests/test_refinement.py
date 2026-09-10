"""Acceptance, shared iteration budgets and timing of residual correction."""

import time

import numpy as np
import pytest
from scipy import sparse

from deflation_example.refinement import verified_refinement
from deflation_example.solvers import LinearResult
from deflation_example.timing import PHASES


def controlled_solver(quality=1.0, status="residual_gap", iterations=2):
    calls = []

    def solve(rhs, initial, tolerance, cap):
        tick = time.perf_counter()
        calls.append((rhs.copy(), initial, tolerance, cap))
        fraction = 0.9 if len(calls) == 1 else quality
        n = min(iterations, cap)
        result = LinearResult(fraction * rhs, n, abs(1 - fraction), status, 1, 2.0)
        seconds = time.perf_counter() - tick
        components = dict.fromkeys(PHASES, 0.0)
        components["iteration"] = seconds
        return result, {
            "components_seconds": components,
            "callback_seconds": seconds,
            "total_seconds": seconds,
            "restriction_and_transfer_seconds": 0.0,
            "iteration_rtol": tolerance,
            "acceptance_rtol": tolerance,
            "newly_inactive": 4 if len(calls) == 1 else 0,
        }

    return solve, calls


def test_correction_checks_original_equation_and_charges_all_work():
    solve, calls = controlled_solver()
    result, metrics = verified_refinement(solve, sparse.eye(3), np.ones(3), None, 1e-10, 10)
    assert result.status == "converged" and result.residual <= 1e-10
    assert result.iterations == 4
    assert len(calls) == 2 and calls[1][3] == 8
    np.testing.assert_allclose(calls[1][0], 0.1)
    assert calls[1][1] is None and calls[1][2] == 0.1
    assert metrics["newly_inactive"] == 4
    assert sum(metrics["components_seconds"].values()) == pytest.approx(metrics["total_seconds"])
    assert metrics["callback_seconds"] == pytest.approx(metrics["total_seconds"])


def test_iteration_cap_includes_initial_solve_and_every_correction():
    solve, calls = controlled_solver(quality=0.9, iterations=3)
    result, _ = verified_refinement(solve, sparse.eye(3), np.ones(3), None, 1e-10, 4)
    assert result.status == "maxiter" and result.iterations == 4
    assert len(calls) == 2 and calls[1][3] == 1


def test_native_convergence_does_not_replace_original_residual_acceptance():
    solve, calls = controlled_solver(quality=0, status="converged")
    result, metrics = verified_refinement(solve, sparse.eye(3), np.ones(3), None, 1e-10, 20)
    assert result.status == "residual_stagnation"
    assert result.residual == pytest.approx(0.1)
    assert len(calls) == 2 and len(metrics["refinement_attempts"]) == 2


def test_correction_budget_is_bounded():
    solve, calls = controlled_solver(quality=0.1)
    result, _ = verified_refinement(solve, sparse.eye(3), np.ones(3), None, 1e-10, 100)
    assert len(calls) == 5 and result.status == "residual_failed"


@pytest.mark.parametrize(
    "scalar, expected, calls_expected",
    [
        ("nonpositive_rz", "converged", 2),
        ("nonpositive_curvature", "breakdown", 1),
        ("nonfinite_scalar", "breakdown", 1),
    ],
)
def test_only_projected_scalar_breakdown_allows_an_error_equation(scalar, expected, calls_expected):
    kernel, calls = controlled_solver(status="breakdown")

    def solve(*args):
        result, timing = kernel(*args)
        timing["termination_test"] = scalar
        return result, timing

    result, _ = verified_refinement(solve, sparse.eye(3), np.ones(3), None, 1e-10, 10)
    assert result.status == expected and len(calls) == calls_expected


def test_improving_error_candidate_can_miss_its_local_target():
    solve, calls = controlled_solver(quality=0.5, status="residual_failed")
    result, metrics = verified_refinement(solve, sparse.eye(3), np.ones(3), None, 1e-10, 100)
    assert len(calls) == 5
    attempts = metrics["refinement_attempts"]
    assert all(a["local_residual"] > a["local_relative_target"] for a in attempts[1:])
    assert all(
        b["original_residual"] < a["original_residual"] for a, b in zip(attempts, attempts[1:])
    )
    assert result.residual == pytest.approx(0.1 * 0.5**4)
    assert result.status == "residual_failed"


def test_final_acceptance_precedes_scalar_failure_and_continuation():
    kernel, calls = controlled_solver(status="breakdown")

    def solve(*args):
        result, timing = kernel(*args)
        result.x = np.ones(3)
        timing["termination_test"] = "nonpositive_curvature"
        return result, timing

    result, _ = verified_refinement(solve, sparse.eye(3), np.ones(3), None, 1e-10, 10)
    assert len(calls) == 1
    assert result.status == "converged" and result.residual == 0


def test_projected_scalar_can_be_negative_after_coarse_orthogonality_is_lost():
    B = np.array([[2.0, 1.0], [1.0, 2.0]])
    Q = np.diag([0.5, 0.0])
    P = np.eye(2) - Q @ B
    K = 0.5 * np.eye(2)
    r = np.array([3.0, 1.0])
    z = P @ K @ r
    assert r @ z == -0.25
    assert z @ B @ z > 0
    balanced = P @ K @ P.T + Q
    assert np.linalg.eigvalsh(balanced)[0] > 0
    assert r @ balanced @ r > 0
    corrected = (np.eye(2) - B @ Q) @ r
    assert corrected[0] == 0
    assert corrected @ P @ K @ corrected > 0
