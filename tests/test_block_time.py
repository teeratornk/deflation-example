"""Block-in-time preconditioning of the prescribed-flow trajectory Hessian."""

import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy import sparse
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from deflation_example import benchmark_mesh as study
from deflation_example.block_time import BlockTimeRestriction


@pytest.fixture(autouse=True)
def bounded_blas_threads():
    with threadpool_limits(2):
        yield


@pytest.fixture(scope="module")
def trajectory():
    c = study.controls(OmegaConf.create({"transient": True, "slabs": 4}))
    return study.build_model(c)[1]


def restricted(trajectory, sweeps, fraction=0.8, seed=5):
    H = BlockTimeRestriction(trajectory.H, trajectory.spatial_size, sweeps)
    I = np.flatnonzero(np.random.default_rng(seed).random(trajectory.size) < fraction)
    return H.restrict(I), I


def test_one_sweep_is_the_exact_block_diagonal_inverse(trajectory):
    B, I = restricted(trajectory, 1)
    slabs = I // trajectory.spatial_size
    r = np.random.default_rng(1).standard_normal(len(I))
    expected = np.zeros_like(r)
    for slab in np.unique(slabs):
        rows = np.flatnonzero(slabs == slab)
        expected[rows] = spsolve(B[rows][:, rows].tocsc(), r[rows])
    np.testing.assert_allclose(B.preconditioner(r), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("sweeps", [1, 2, 3])
def test_every_sweep_count_is_symmetric_positive_definite(trajectory, sweeps):
    B, I = restricted(trajectory, sweeps)
    rng = np.random.default_rng(2)
    U = rng.standard_normal((len(I), 6))
    MU = np.column_stack([B.preconditioner(u) for u in U.T])
    gram = U.T @ MU
    np.testing.assert_allclose(gram, gram.T, rtol=1e-9, atol=1e-9 * np.abs(gram).max())
    assert np.linalg.eigvalsh(0.5 * (gram + gram.T)).min() > 0


def test_restrictions_keep_the_matrix_and_count_factorizations(trajectory):
    H = BlockTimeRestriction(trajectory.H, trajectory.spatial_size, 3)
    I = np.arange(trajectory.size)
    B = H.restrict(I)
    assert abs(B - sparse.csr_matrix(trajectory.H.restrict(I))).max() == 0
    assert H.factorizations == 1


def test_block_sequence_reaches_the_jacobi_optimum_with_fewer_iterations():
    c = study.controls(
        OmegaConf.create(
            {
                "transient": True,
                "slabs": 4,
                "targets": 2,
                "save_fields": False,
                "methods": ["jacobi", "block"],
                "rank": 8,
            }
        )
    )
    records = {m: study.sequence(c, m)[0] for m in ("jacobi", "block")}
    for m, record in records.items():
        assert record["success"], (m, record.get("failure"))
    for a, b in zip(records["jacobi"]["cases"], records["block"]["cases"]):
        assert abs(a["objective"] - b["objective"]) <= 1e-9 * abs(a["objective"])
    total = {m: sum(case["inner_iterations"] for case in r["cases"]) for m, r in records.items()}
    assert total["block"] < total["jacobi"]
    assert records["block"]["storage"]["block_preconditioner"]["sweeps"] == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"transient": True, "methods": ["block"], "device": "cuda"},
        {"transient": False, "methods": ["block"]},
        {"transient": True, "methods": ["block_reference"], "matrix_free_inner": True},
    ],
)
def test_block_methods_require_host_cg_on_assembled_trajectories(overrides):
    with pytest.raises(ValueError):
        study.controls(OmegaConf.create(overrides))
