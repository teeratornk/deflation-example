"""A general-metric Ritz selection and the additive two-level coarse coupling.

The campaign's coarse space is selected against the Jacobi diagonal even when the
inner solver runs an operator preconditioner, which the preconditioner ablation
recorded as a caveat. These cover the two pieces needed to remove that mismatch: a
selection posed in an arbitrary positive-definite metric, and a coarse correction that
joins the preconditioner additively instead of through an operator application.
"""

import numpy as np
import pytest
from scipy import sparse

from deflation_example.recycling import jacobi_ritz, metric_ritz
from deflation_example.solvers import deflated_cg


def spd(n, seed, spread=3.0):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    values = np.logspace(0, spread, n)
    return sparse.csr_matrix(Q @ np.diag(values) @ Q.T), rng


def test_metric_ritz_reproduces_the_jacobi_selection_when_the_metric_is_that_diagonal():
    A, rng = spd(24, 3)
    diagonal = A.diagonal()
    candidates = rng.normal(size=(24, 8))
    expected, _ = jacobi_ritz(A, diagonal, candidates, 4)
    actual, report = metric_ritz(A, sparse.diags(diagonal), candidates, 4)
    assert report["selected_rank"] == 4
    # Same subspace, so each basis is carried into the other by an orthogonal change of
    # coordinates; compare the projectors, which are independent of that choice.
    projector = lambda Z: Z @ np.linalg.solve(Z.T @ Z, Z.T)  # noqa: E731
    np.testing.assert_allclose(projector(actual), projector(expected), atol=1e-9)


def test_metric_ritz_returns_a_basis_orthonormal_in_the_supplied_metric():
    A, rng = spd(20, 5)
    M, _ = spd(20, 6, spread=2.0)
    selected, _ = metric_ritz(A, M, rng.normal(size=(20, 9)), 5)
    np.testing.assert_allclose(selected.T @ (M @ selected), np.eye(5), atol=1e-9)


def test_metric_ritz_selects_the_low_end_of_the_metric_pencil():
    """The retained Ritz values must be the smallest available, not an arbitrary five."""
    A, rng = spd(18, 7)
    M, _ = spd(18, 8, spread=1.5)
    candidates = rng.normal(size=(18, 10))
    few, low = metric_ritz(A, M, candidates, 3)
    many, full = metric_ritz(A, M, candidates, 10)
    assert low["ritz_min"] == pytest.approx(full["ritz_min"], rel=1e-10)
    assert low["ritz_max"] <= full["ritz_max"] + 1e-10
    quotients = np.diag(many.T @ (A @ many)) / np.diag(many.T @ (M @ many))
    assert np.all(np.diff(quotients) > -1e-8)
    assert few.shape[1] == 3


def test_metric_ritz_truncates_rank_deficient_candidates():
    A, rng = spd(16, 9)
    M, _ = spd(16, 10, spread=1.0)
    candidates = rng.normal(size=(16, 4))
    repeated = np.column_stack((candidates, candidates[:, :2]))
    _, report = metric_ritz(A, M, repeated, 6)
    assert report["candidate_rank"] == 4


def test_metric_ritz_rejects_incompatible_or_indefinite_input():
    A, rng = spd(12, 11)
    with pytest.raises(ValueError):
        metric_ritz(A, sparse.eye(11, format="csr"), rng.normal(size=(12, 3)), 2)
    with pytest.raises(ValueError):
        metric_ritz(A, sparse.diags(np.zeros(12)), rng.normal(size=(12, 3)), 2)
    indefinite = sparse.diags(np.r_[np.ones(11), -1.0])
    with pytest.raises(ValueError):
        metric_ritz(A, indefinite, np.eye(12)[:, 9:], 2)


def test_metric_ritz_drops_a_zero_candidate_column_without_failing():
    A, rng = spd(12, 19)
    M, _ = spd(12, 21, spread=1.0)
    candidates = np.column_stack((rng.normal(size=(12, 2)), np.zeros(12)))
    selected, report = metric_ritz(A, M, candidates, 2)
    assert report["candidate_rank"] == 2
    assert selected.shape[1] == 2


def test_additive_and_multiplicative_coarse_couplings_reach_the_same_solution():
    A, rng = spd(40, 13)
    b = rng.normal(size=40)
    basis = rng.normal(size=(40, 5))
    results = {
        coupling: deflated_cg(
            A, b, basis, A.diagonal(), rtol=1e-12, maxiter=4000, coarse_coupling=coupling
        )
        for coupling in ("multiplicative", "additive")
    }
    for coupling, result in results.items():
        assert result.status == "converged", coupling
        assert result.rank == 5
    np.testing.assert_allclose(results["additive"].x, results["multiplicative"].x, atol=1e-8)


def test_additive_coupling_spends_no_operator_application_on_the_correction():
    """The additive form reuses the residual; the multiplicative form applies A to z."""
    A, rng = spd(30, 15)
    b = rng.normal(size=30)
    basis = rng.normal(size=(30, 4))
    counts = {}
    for coupling in ("multiplicative", "additive"):
        applications = {"n": 0}
        operator = sparse.linalg.LinearOperator(
            A.shape,
            matvec=lambda v, applications=applications: (
                applications.__setitem__("n", applications["n"] + 1),
                A @ v,
            )[1],
            dtype=float,
        )
        result = deflated_cg(
            operator, b, basis, A.diagonal(), rtol=1e-10, maxiter=2000, coarse_coupling=coupling
        )
        assert result.status == "converged"
        counts[coupling] = applications["n"] / max(result.iterations, 1)
    assert counts["additive"] < counts["multiplicative"]


def test_deflated_cg_rejects_an_unknown_coarse_coupling():
    A, rng = spd(10, 17)
    with pytest.raises(ValueError):
        deflated_cg(A, rng.normal(size=10), None, A.diagonal(), coarse_coupling="two-level")
