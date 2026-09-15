"""The optional temporal eigensolver preserves the declared weighted compression."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import linalg, sparse

from deflation_example.coupled_reference import build_reference
from deflation_example.mesh_reference import _tridiagonal_reference, temporal_diagonals
from test_axisymmetric_flow import annular_rectangle
from test_coupled_reference import problem


@pytest.mark.parametrize("steps", [[0.7], [0.2, 0.35], [0.1, 0.3, 0.2, 0.7]])
@pytest.mark.parametrize("cross", [-0.3, 0.0, 0.3])
def test_temporal_diagonals_match_weighted_dense_operator(steps, cross):
    steps = np.asarray(steps)
    n = len(steps)
    T = sparse.diags([1 / steps, -1 / steps[1:]], [0, -1], shape=(n, n)).toarray()
    W = np.diag(steps / steps.mean())
    expected = 2 * W + 0.04 * (3 * W + cross * (W @ T + T.T @ W) + 0.8 * T.T @ W @ T)
    diagonal, off = temporal_diagonals(steps, 2.0, 3.0, cross, 0.8, 0.04)
    actual = np.diag(diagonal) + np.diag(off, -1) + np.diag(off, 1)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("metric", ["jacobi", "euclidean"])
@pytest.mark.parametrize("steps", [[0.2], [0.2, 0.3], [0.2, 0.3, 0.1, 0.17, 0.4]])
def test_selected_reference_matches_dense_space_and_coarse_correction(metric, steps):
    p = problem(annular_rectangle(4), steps)
    dense = build_reference(p, 5, temporal_metric=metric)
    tri = build_reference(p, 5, temporal_metric=metric, temporal_solver="tridiagonal")
    indices = np.arange(p.spatial_size * len(steps))
    Z, Y = dense.restrict(indices), tri.restrict(indices)
    q, _ = np.linalg.qr(Z)
    v, _ = np.linalg.qr(Y)
    np.testing.assert_allclose(q @ q.T, v @ v.T, atol=2e-10)
    scores = np.asarray(dense.description["selection"])[:, 0]
    np.testing.assert_allclose(np.asarray(tri.description["selection"])[:, 0], scores, rtol=1e-12)
    # Compare the operation, which is invariant to normalization and signs.
    diagonal = np.linspace(1, 3, len(indices))
    rhs = np.random.default_rng(6).normal(size=len(indices))
    expected = Z @ np.linalg.solve(Z.T @ (diagonal[:, None] * Z), Z.T @ rhs)
    actual = Y @ np.linalg.solve(Y.T @ (diagonal[:, None] * Y), Y.T @ rhs)
    np.testing.assert_allclose(actual, expected, atol=2e-10)
    assert tri.description["temporal_eigensolver"] == "tridiagonal_two_pass"
    assert dense.description["temporal_eigensolver"] == "dense"


def test_long_temporal_reference_avoids_dense_temporal_eigensolves(monkeypatch):
    n = 2048
    steps = np.full(n, 0.3)
    phi = np.eye(3)
    AP, CP = np.diag([1, 2, 3]), np.diag([0.8, 1.0, 1.2])
    p = SimpleNamespace(
        steps=steps,
        spatial_size=3,
        alpha=0.02,
        H=SimpleNamespace(diagonal=lambda: np.ones(3 * n)),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("The tridiagonal path must not call the dense eigensolver")

    monkeypatch.setattr(linalg, "eigh", forbidden)
    result = _tridiagonal_reference(p, phi, AP, CP, np.ones(3), 5, "jacobi", {})
    assert result.temporal.shape == (n, 5)
    assert np.isfinite(result.temporal).all()
    for i, (value, j, _) in enumerate(result.description["selection"]):
        d, e = temporal_diagonals(steps, 1, (j + 1) ** 2, AP[j, j] * CP[j, j], CP[j, j] ** 2, 0.02)
        column = result.temporal[:, i]
        action = d * column
        action[:-1] += e * column[1:]
        action[1:] += e * column[:-1]
        assert np.linalg.norm(action - value * column) / np.linalg.norm(action) < 1e-11


@pytest.mark.parametrize("steps", [[], [0.0], [-1.0], [np.inf], [[0.1, 0.2]]])
def test_temporal_compression_rejects_invalid_time_grids(steps):
    with pytest.raises(ValueError, match="Positive time steps"):
        temporal_diagonals(steps, 1, 2, 0.1, 3, 0.01)


def test_tridiagonal_option_rejects_unsupported_construction():
    p = problem(annular_rectangle(3), [0.2, 0.3])
    with pytest.raises(ValueError, match="mode-dependent"):
        build_reference(p, 2, construction="tensor", temporal_solver="tridiagonal")
    with pytest.raises(ValueError, match="dense or tridiagonal"):
        build_reference(p, 2, temporal_solver="unknown")


def test_temporal_reference_checks_the_returned_vectors(monkeypatch):
    p = problem(annular_rectangle(3), [0.2, 0.3, 0.4])
    original = linalg.eigh_tridiagonal

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if not kwargs.get("eigvals_only", False):
            values, vectors = result
            vectors = vectors.copy()
            vectors[0] += 0.2
            return values, vectors
        return result

    monkeypatch.setattr(linalg, "eigh_tridiagonal", corrupt)
    with pytest.raises(RuntimeError, match="independently recomputed residual"):
        build_reference(p, 2, temporal_solver="tridiagonal")
