"""GPU and CPU recycling use the same energy-metric pencil for frozen sweeps."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import orth
from scipy.sparse.linalg import aslinearoperator

from deflation_example.coupled_cuda_solver import CudaRecycleSpace, preconditioned_ritz
from deflation_example.coupled_preconditioned_reference import preconditioned_reference
from deflation_example.study_solvers import ArrayReference


def compare(cp):
    rng = np.random.default_rng(342)
    H, K = (rng.normal(size=(12, 12)) for _ in range(2))
    H, K = H.T @ H + np.eye(12), K.T @ K + np.eye(12)
    raw = rng.normal(size=(12, 6))
    candidates = np.column_stack((raw, raw[:, :1], np.zeros(12)))
    expected = preconditioned_reference(
        ArrayReference(candidates, {}),
        aslinearoperator(H),
        lambda x: K @ x,
        3,
        enrich=False,
    )
    a, b = cp.asarray(H), cp.asarray(K)
    actual, details = preconditioned_ritz(
        cp, lambda x: a @ x, lambda x: b @ x, cp.asarray(candidates), 3
    )
    Q, V = orth(expected.basis), orth(cp.asnumpy(actual))
    np.testing.assert_allclose(Q @ Q.T, V @ V.T, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(
        details["ritz_values"], expected.description["ritz_values"], rtol=1e-9
    )
    assert details["candidate_rank"] == expected.description["independent_candidates"] == 6
    space = CudaRecycleSpace(cp, 3, 4)
    space.begin(np.arange(12))
    for column in raw[:, :4].T:
        space.capture(cp.asarray(column))
    report = space.finish(
        lambda x: a @ x,
        cp.asarray(H.diagonal()),
        cp.asarray(raw[:, 4:]),
        "converged",
        inverse=lambda x: b @ x,
    )
    assert report["policy"] == "existing-coarse-plus-new-directions-preconditioned-ritz-v1"
    assert report["existing_columns"] == 2 and report["new_directions_retained"] == 4
    assert report["candidate_columns"] == 6 and report["selected_rank"] == 3


def test_preconditioned_gpu_algebra_matches_host_selection():
    compare(SimpleNamespace(**vars(np), asnumpy=np.asarray))


@pytest.mark.gpu
def test_resident_preconditioned_ritz_matches_cpu():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA device required")
    compare(cp)


def test_empty_space_has_no_replacement_directions():
    cp = SimpleNamespace(**vars(np), asnumpy=np.asarray)
    basis, report = preconditioned_ritz(cp, lambda x: x, lambda x: x, np.zeros((6, 3)), 3)
    assert basis.shape == (6, 0) and report["candidate_rank"] == 0
