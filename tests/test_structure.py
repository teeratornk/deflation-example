import numpy as np
from scipy import linalg, sparse
from deflation_example.benchmark_structure import restriction_example, ritz_space


def test_conditional_bound_and_inconclusive_transition():
    rows = restriction_example()["rows"]
    assert any(
        r["sufficient_angle_bound"] is not None and r["sufficient_angle_bound"] < 1 for r in rows
    )
    assert any(r["separation"] <= 0 for r in rows)
    for r in rows:
        assert r["retained_rank"] == 2 and 0 < r["minimum_restricted_singular_value"] < 1
        assert r["residual_identity_error"] < 1e-12
        if r["separation"] > 0:
            assert r["actual_sine"] <= r["residual_angle_bound"] + 1e-12
            assert r["actual_sine"] <= r["sufficient_angle_bound"] + 1e-12


def test_scaled_ritz_mapping_and_full_span_invariance():
    rng = np.random.default_rng(9)
    A = rng.normal(size=(20, 20))
    B = sparse.csr_matrix(A.T @ A + np.diag(np.geomspace(1, 100, 20)))
    W = rng.normal(size=(20, 8))
    Z = ritz_space(B, W, 3, scaled=True)
    root = np.sqrt(B.diagonal())
    np.testing.assert_allclose(Z.T @ (root[:, None] ** 2 * Z), np.eye(3), atol=1e-12)
    for scaled in (False, True):
        full = ritz_space(B, W, 8, scaled=scaled)
        assert np.max(linalg.subspace_angles(full, W)) < 1e-12
