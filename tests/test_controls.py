"""Restricted-space transfer, informative PDE bounds and matrix-free parity."""

from collections import deque
import numpy as np
import pytest
from scipy import sparse
from threadpoolctl import threadpool_limits

from deflation_example.benchmark_controls import (
    pde_separation,
    retained_directions,
    recycling_comparison,
)
from deflation_example.problems import laplacian
from deflation_example.solvers import deflated_cg, restricted_normal_operator


def test_pde_sufficient_bound_and_identity():
    rows = pde_separation()["rows"]
    informative = [
        r
        for r in rows
        if r["sufficient_angle_bound"] is not None and r["sufficient_angle_bound"] < 1
    ]
    assert informative
    for row in rows:
        assert row["residual_identity_error"] < 1e-10
        if row["sufficient_angle_bound"] is not None:
            assert row["actual_sine"] <= row["sufficient_angle_bound"] + 1e-12
    assert any(
        r["sufficient_angle_bound"] is None or r["sufficient_angle_bound"] == 1 for r in rows
    )


def test_direction_observer_preserves_cg_and_bounds_history():
    H = sparse.diags(np.arange(1.0, 31.0))
    b = np.ones(30)
    directions = deque(maxlen=12)
    a = deflated_cg(H, b)
    c = deflated_cg(H, b, direction_callback=directions.append)
    assert a.status == c.status == "converged"
    np.testing.assert_array_equal(a.x, c.x)
    assert len(directions) == 12
    Z = retained_directions(H, directions, 4)
    np.testing.assert_allclose(Z.T @ Z, np.eye(4), atol=1e-12)
    with pytest.raises(ValueError):
        deflated_cg(H, b, direction_callback=True)


def test_history_comparison_matches_rank_and_inactive_system():
    with threadpool_limits(limits=1):
        result = recycling_comparison("diffusion", 6, rank=3, targets=3, repeats=1)
    assert result["success"]
    rows = result["sequences"][0]["rows"]
    for i in range(3):
        pair = [r for r in rows if r["index"] == i]
        assert len(pair) == 2
        assert pair[0]["rhs_sha256"] == pair[1]["rhs_sha256"]
        assert pair[0]["inactive_sha256"] == pair[1]["inactive_sha256"]
        assert pair[0]["matched_rank"] == pair[1]["matched_rank"]
        assert all(r["matched_deployed_rank"] for r in pair)
        assert all(r["original_residual"] <= 1e-10 for r in pair)
        assert pair[0]["matched_rank"] == (0 if i == 0 else 3)


@pytest.mark.gpu
def test_matrix_free_stencil_and_rank_zero():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from deflation_example.benchmark_matrix_free import stencil_operator, solve

    for n in (4, 6):
        indices = np.arange(n**3)[np.arange(n**3) % 5 != 0]
        B, diagonal = stencil_operator(n, torch.tensor(indices, device="cuda"), torch)
        L = laplacian(n, 3)
        cpu = restricted_normal_operator(L, 0.001, indices)
        V = np.random.default_rng(13).normal(size=(len(indices), 3))
        np.testing.assert_allclose(
            B(torch.tensor(V, device="cuda")).cpu().numpy(), cpu @ V, rtol=1e-12, atol=1e-9
        )
        H = sparse.eye(n**3) + 0.001 * (L.T @ L)
        np.testing.assert_allclose(diagonal.cpu().numpy(), H.diagonal()[indices], rtol=1e-13)
    a, b = solve(4, 0), solve(4, 3)
    assert a["status"] == b["status"] == "converged"
    assert a["truth_sha256"] == b["truth_sha256"]
    assert a["rhs_sha256"] == b["rhs_sha256"]
    assert a["deployed_rank"] == 0 and b["deployed_rank"] == 3
    assert max(a["cpu_original_residual"], b["cpu_original_residual"]) <= 1e-10
