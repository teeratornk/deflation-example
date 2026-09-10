"""Matched transfer, retained coarse information and scaled selection contracts."""

import numpy as np
import pytest
from scipy import sparse

from deflation_example.recycling import RecycleSpace, jacobi_ritz, transfer_basis
from deflation_example.solvers import deflated_cg, orthonormalize


def test_reference_transfer_identity_and_released_nodes():
    phi = np.random.default_rng(19).normal(size=(12, 4))
    previous = np.array([0, 2, 4, 6, 8, 10])
    current = np.array([1, 2, 3, 6, 9, 10])
    transferred = transfer_basis(phi[previous], previous, current)
    missing = np.zeros_like(phi)
    missing[np.setdiff1d(np.arange(12), previous)] = phi[np.setdiff1d(np.arange(12), previous)]
    np.testing.assert_array_equal(phi[current] - transferred, missing[current])
    np.testing.assert_array_equal(transferred[[0, 2, 4]], np.zeros((3, 4)))
    assert np.linalg.norm(phi[current] - transferred) > 0


def test_sequential_transfer_does_not_restore_lost_reference_values():
    phi = np.eye(4)
    first, second = np.array([0, 1]), np.array([1, 2])
    transferred = transfer_basis(phi[first], first, second)
    restored = transfer_basis(transferred, second, np.arange(4))
    assert np.linalg.matrix_rank(restored) == 1
    np.testing.assert_array_equal(restored[2], np.zeros(4))


def test_scaled_ritz_matches_preconditioned_eigenpairs_and_preserves_span():
    rng = np.random.default_rng(4)
    C = rng.normal(size=(12, 12))
    A = C.T @ C + np.eye(12)
    diagonal = np.diag(A)
    scaled = A / np.sqrt(diagonal[:, None] * diagonal[None, :])
    values, _ = np.linalg.eigh(scaled)
    chosen, metrics = jacobi_ritz(A, diagonal, np.eye(12), 4)
    np.testing.assert_allclose(chosen.T @ (diagonal[:, None] * chosen), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(chosen.T @ A @ chosen, np.diag(values[:4]), atol=1e-12)
    assert metrics["selected_rank"] == 4
    candidates = rng.normal(size=(12, 5))
    rotated, _ = jacobi_ritz(A, diagonal, candidates, 5)
    Q = orthonormalize(candidates)
    np.testing.assert_allclose(rotated, Q @ Q.T @ rotated, atol=1e-12)


def test_recycling_retains_existing_coarse_direction_and_updates_from_accepted_solves():
    A = sparse.diags([1.0, 2.0, 4.0, 8.0])
    state = RecycleSpace(2, window=2)
    assert state.begin(np.arange(4)).shape == (4, 0)
    # Existing coarse vector is orthogonal to the new projected directions.
    existing = np.eye(4)[:, :1]
    for p in np.eye(4)[:, 1:].T:
        state.capture(p)
    assert len(state.directions) == 2
    result = state.finish(A, np.ones(4), existing, "converged")
    assert result["existing_columns"] == 1 and result["new_directions_observed"] == 3
    assert result["candidate_columns"] == 3
    Q = orthonormalize(state.basis)
    np.testing.assert_allclose(Q @ Q.T @ existing, existing, atol=1e-12)
    transferred = state.begin(np.array([1, 2]))
    assert transferred.shape == (2, 2)
    assert np.linalg.matrix_rank(transferred) <= 1
    state.finish(sparse.eye(2), np.ones(2), transferred, "maxiter")
    assert state.basis.shape == (0, 0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_recycle_configuration_and_order_validation(device):
    state = RecycleSpace(2, 3, device)
    with pytest.raises(ValueError):
        state.finish(np.eye(2), np.ones(2), None, "converged")
    with pytest.raises(ValueError):
        state.begin([1, 1])


def test_cpu_recycle_sequence_keeps_original_residual_acceptance():
    rng = np.random.default_rng(8)
    C = rng.normal(size=(12, 12))
    H = C.T @ C + np.eye(12)
    state = RecycleSpace(3, 6)
    for inactive in [np.arange(12), np.arange(8), np.arange(4, 12), np.arange(12)]:
        B = H[np.ix_(inactive, inactive)]
        b = rng.normal(size=len(inactive))
        basis = state.begin(inactive)
        result = deflated_cg(B, b, basis, np.diag(B), direction_callback=state.capture)
        assert result.status == "converged" and result.residual <= 1e-10
        state.finish(B, np.diag(B), orthonormalize(basis), result.status)
        assert state.basis.shape[1] <= 3


@pytest.mark.gpu
def test_gpu_recycle_sequence_and_cpu_selection_agree():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from deflation_example.gpu import gpu_deflated_cg
    from deflation_example.recycling import gpu_jacobi_ritz

    rng = np.random.default_rng(24)
    C = rng.normal(size=(20, 20))
    H = C.T @ C + np.eye(20)
    dense = torch.tensor(H, device="cuda")
    candidate = rng.normal(size=(20, 6))
    cpu, _ = jacobi_ritz(H, np.diag(H), candidate, 3)
    gpu, _ = gpu_jacobi_ritz(
        dense.to_sparse_csr(), dense.diag(), torch.tensor(candidate, device="cuda"), 3, torch
    )
    Q = orthonormalize(cpu)
    np.testing.assert_allclose(gpu.cpu().numpy(), Q @ Q.T @ gpu.cpu().numpy(), atol=1e-10)
    state = RecycleSpace(3, 6, "cuda")
    for inactive in [np.arange(20), np.arange(15), np.arange(5, 20)]:
        B = sparse.csr_matrix(H[np.ix_(inactive, inactive)])
        result, timing = gpu_deflated_cg(
            B,
            rng.normal(size=len(inactive)),
            state.begin(inactive),
            B.diagonal(),
            basis_backend="gpu_qr",
            direction_callback=state.capture,
            completion_callback=state.finish,
        )
        assert result.status == "converged" and result.residual <= 1e-10
        assert timing["completion"]["selected_rank"] <= 3
        assert sum(timing["components_seconds"].values()) == pytest.approx(timing["total_seconds"])
