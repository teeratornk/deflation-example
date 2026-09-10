"""CUDA checks for resident reference factors and matched resident recycling."""

import numpy as np
import pytest
from scipy import sparse


@pytest.fixture
def torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch


@pytest.mark.gpu
def test_resident_space_time_restriction_matches_host(torch_cuda):
    from deflation_example.mesh_reference import DeviceMeshReference
    from deflation_example.spacetime_reference import SpaceTimeReference
    from deflation_example.gpu import gpu_deflated_cg
    rng = np.random.default_rng(112)
    reference = SpaceTimeReference(rng.normal(size=(10, 2)), rng.normal(size=(4, 3)),
                                   [0, 1, 1], {})
    resident = DeviceMeshReference(reference, torch_cuda)
    I = np.arange(0, 40, 2)
    host = reference.restrict(I)
    device = resident.restrict_device(I)
    np.testing.assert_array_equal(device.cpu().numpy(), host)
    B = sparse.diags([-np.ones(19), np.full(20, 3.), -np.ones(19)], [-1, 0, 1], format="csr")
    b = rng.normal(size=20)
    first, _ = gpu_deflated_cg(B, b, host, basis_backend="gpu_qr", rtol=1e-11)
    second, _ = gpu_deflated_cg(B, b, device_basis=device, basis_backend="gpu_qr", rtol=1e-11)
    assert first.status == second.status == "converged"
    assert first.rank == second.rank == 3
    np.testing.assert_allclose(first.x, second.x, atol=1e-12)
    np.testing.assert_array_equal(device.cpu().numpy(), host)
    assert resident.storage()["persistent_device_reference_bytes"] > 0
    with pytest.raises(ValueError, match="Device basis"):
        gpu_deflated_cg(B, b, host, device_basis=device, basis_backend="gpu_qr")


@pytest.mark.gpu
def test_resident_recycling_transfers_existing_coarse_vectors(torch_cuda):
    from deflation_example.study_solvers import StudySolver
    B = sparse.diags([-np.ones(39), np.full(40, 3.), -np.ones(39)], [-1, 0, 1], format="csr")
    host = StudySolver("recycling", device="cuda", rank=5, window=5, torch=torch_cuda)
    device = StudySolver("recycling", device="cuda", rank=5, window=5,
                          torch=torch_cuda, resident_recycling=True)
    try:
        for shift in (0, 2, 1):
            I = np.arange(shift, shift+40)
            b = np.sin(I+.3)
            a, ma = host.solve(B, b, I)
            c, mc = device.solve(B, b, I)
            assert a.status == c.status == "converged"
            np.testing.assert_allclose(a.x, c.x, atol=1e-9)
            assert ma["input_basis_columns"] == mc["input_basis_columns"]
            assert mc["basis_transfer_device"] == "cuda"
            assert mc["completion"]["history_location"] == "cuda"
        assert device.history.basis.is_cuda
    finally:
        host.close()
        device.close()
