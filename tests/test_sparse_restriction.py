"""Slicing the assembled reduced Hessian gives the sparse-product restriction."""

import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from deflation_example import benchmark_mesh as study
from deflation_example.mesh_showcases import desired_temperature
from deflation_example.solvers import LinearResult, pdas


@pytest.fixture(autouse=True)
def bounded_blas_threads():
    with threadpool_limits(2):
        yield


def pair(overrides):
    c = study.controls(OmegaConf.create(overrides))
    _, product = study.build_model(c)
    _, sliced = study.build_model({**c, "hessian_restriction": "submatrix"})
    return c, product, sliced


@pytest.mark.parametrize(
    "overrides",
    [
        {"level": 0},
        {"level": 0, "transient": True, "slabs": 3},
        {"geometry": "transformer_2d", "level": 0, "alpha": 1e-14, "bound": 0.8},
    ],
)
def test_slices_equal_the_sparse_products_to_round_off(overrides):
    _, product, sliced = pair(overrides)
    rng = np.random.default_rng(3)
    for fraction in (1.0, 0.8, 0.3):
        I = np.flatnonzero(rng.random(product.size) < fraction)
        a, b = product.H.restrict(I), sliced.H.restrict(I)
        assert a.shape == b.shape == (len(I), len(I))
        difference = abs(a - b)
        scale = abs(a).max()
        assert (difference.max() if difference.nnz else 0.0) <= 1e-14 * scale
    np.testing.assert_array_equal(product.H.diagonal(), sliced.H.diagonal())


def test_pdas_takes_the_same_steps_with_either_restriction():
    c, product, sliced = pair({"level": 0, "transient": True, "slabs": 3})
    desired = desired_temperature(product, 0, 4)
    load = product.load(desired)

    def direct(B, b, I):
        return LinearResult(spsolve(B.tocsc(), b), 1, 0.0, "converged")

    first = pdas(product.H, load, c["bound"], linear_solver=direct)
    second = pdas(sliced.H, load, c["bound"], linear_solver=direct)
    assert first["status"] == second["status"] == "converged"
    assert first["iterations"] == second["iterations"]
    scale = np.linalg.norm(first["y"])
    assert np.linalg.norm(first["y"] - second["y"]) <= 1e-12 * scale
    assert [h.get("active") for h in first["history"]] == [
        h.get("active") for h in second["history"]
    ]


def test_slicing_requires_assembled_restrictions():
    with pytest.raises(ValueError):
        study.controls(
            OmegaConf.create({"hessian_restriction": "submatrix", "matrix_free_inner": True})
        )
    with pytest.raises(ValueError):
        study.controls(OmegaConf.create({"hessian_restriction": "dense"}))


def test_principal_submatrix_matches_scipy_indexing_in_any_order():
    from scipy import sparse

    from deflation_example.sparse_restriction import principal_submatrix

    rng = np.random.default_rng(7)
    M = sparse.random(60, 60, density=0.2, random_state=1, format="csr")
    M = (M + M.T).tocsr()
    M.sort_indices()
    for I in (np.arange(60), np.sort(rng.choice(60, 25, replace=False)), rng.permutation(60)[:30]):
        expected = M[I][:, I].toarray()
        got = principal_submatrix(M, I)
        np.testing.assert_array_equal(got.toarray(), expected)
        assert got.has_sorted_indices
    assert principal_submatrix(M, np.array([], dtype=int)).shape == (0, 0)
