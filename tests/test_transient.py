"""Trajectory coupling, quadrature, recovery and independent bounded least squares."""

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import lsq_linear

from deflation_example.benchmark_extended import target_parameters
from deflation_example.problems import laplacian
from deflation_example.solvers import kkt_metrics, pdas
from deflation_example.spacetime_reference import build_space_time_reference
from deflation_example.transient import build_transient_cht


@pytest.mark.parametrize("steps", [[0.03, 0.03, 0.03], [0.01, 0.03, 0.05]])
def test_trajectory_actions_initial_forcing_final_block_and_weights(steps):
    rng = np.random.default_rng(91)
    model = build_transient_cht(3, 3, 0.09, initial=rng.normal(size=27), time_steps=steps)
    y, p = rng.normal(size=(2, 81))
    np.testing.assert_allclose(model.A @ y, model.apply_linear(y), rtol=1e-13, atol=1e-10)
    np.testing.assert_allclose(model.A.T @ p, model.apply_transpose(p), rtol=1e-13, atol=1e-10)
    assert y @ model.apply_transpose(p) == pytest.approx(model.apply_linear(y) @ p, rel=1e-13)
    np.testing.assert_array_equal(model.forcing[27:], np.zeros(54))
    np.testing.assert_allclose(model.forcing[:27], model.capacity * model.initial / steps[0])
    control, _ = model.recover(y)
    np.testing.assert_allclose(model.forward(control), y, rtol=1e-11, atol=1e-11)
    W = sparse.diags(model.weights)
    np.testing.assert_allclose(
        model.H.toarray(), (W + 0.001 * model.A.T @ W @ model.A).toarray(), rtol=1e-13, atol=1e-9
    )
    np.testing.assert_allclose(model.weights.reshape(3, 27)[:, 0], np.array(steps) / np.mean(steps))
    assert model.A[-27:, :-27].nnz > 0
    final = p.reshape(3, 27)[-1]
    np.testing.assert_allclose(
        model.apply_transpose(p)[-27:],
        model.spatial.A.T @ final + model.capacity * final / steps[-1],
        atol=1e-10,
    )


@pytest.mark.parametrize("steps", [None, [0.02, 0.03, 0.05]])
def test_complete_trajectory_matches_independent_bounded_least_squares(steps):
    model = build_transient_cht(3, 3, 0.1, initial=np.full(27, 0.0002), time_steps=steps)
    desired = model.target(target_parameters(1, 3))
    bound = 0.0001
    result = pdas(model.H, model.load(desired), bound, tolerance=1e-9)
    assert result["status"] == "converged"
    root = np.sqrt(model.weights)
    factor = np.vstack((np.diag(root), np.sqrt(0.001) * root[:, None] * model.A.toarray()))
    rhs = np.concatenate((root * desired, np.sqrt(0.001) * root * model.forcing))
    independent = lsq_linear(
        factor, rhs, bounds=(-np.inf, bound), method="bvls", tol=1e-13, max_iter=1000
    )
    assert independent.success
    multiplier = np.where(
        independent.active_mask == 1, model.load(desired) - model.H @ independent.x, 0.0
    )
    assert (
        max(kkt_metrics(model.H, model.load(desired), bound, independent.x, multiplier).values())
        < 1e-8
    )
    np.testing.assert_allclose(result["y"], independent.x, rtol=1e-7, atol=1e-11)
    assert model.objective(result["y"], desired) == pytest.approx(
        model.objective(independent.x, desired), rel=1e-12
    )
    assert max(model.recovery_metrics(result["y"], desired, result["multiplier"]).values()) < 1e-8


@pytest.mark.parametrize("construction", ["mode_dependent", "tensor"])
def test_space_time_reference_rank_storage_and_full_reference_residual(construction):
    model = build_transient_cht(3, 4, 0.1)
    reference = build_space_time_reference(model, 12, construction)
    Phi = reference.restrict(np.arange(model.size))
    np.testing.assert_allclose(Phi.T @ Phi, np.eye(12), atol=1e-12)
    I = np.array([0, 27, 53, 70, 101])
    np.testing.assert_array_equal(reference.restrict(I), Phi[I])
    assert reference.storage()["compact_reference_bytes"] < Phi.nbytes
    A0 = sparse.kron(sparse.eye(4), laplacian(3, 3)) + sparse.kron(
        model.time_matrix, sparse.eye(27)
    )
    M0 = sparse.eye(model.size) + 0.001 * A0.T @ A0
    values = np.array(reference.description["reference_rayleigh_values"])
    residual = M0 @ Phi - Phi * values
    if construction == "mode_dependent":
        np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-11)
        np.testing.assert_allclose(values, np.linalg.eigvalsh(M0.toarray())[:12], atol=1e-11)
    else:
        assert np.linalg.norm(residual) > 1e-3


def test_transient_queries_are_distinct_and_time_is_coupled():
    model = build_transient_cht(4, 8, 0.1)
    targets = [model.target(target_parameters(i, 4)) for i in range(4)]
    assert all(
        np.linalg.norm(a - b) > 0.01 for i, a in enumerate(targets) for b in targets[i + 1 :]
    )
    values = targets[0].reshape(8, 64)
    assert np.max(values[3]) > np.max(values[-1])
    assert model.H[:64, 64:128].nnz > 0


def test_trajectory_validation_and_rank_zero_reference():
    with pytest.raises(ValueError):
        build_transient_cht(3, 3, 0.1, time_steps=[0.1, 0.1, 0.1])
    with pytest.raises(ValueError):
        build_transient_cht(3, 3, initial=[1.0])
    model = build_transient_cht(3, 3)
    reference = build_space_time_reference(model, 0)
    assert reference.restrict(np.arange(model.size)).shape == (model.size, 0)
