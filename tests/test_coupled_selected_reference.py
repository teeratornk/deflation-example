"""Scaled selection, rank loss, transfer, and exact-span controls."""

import numpy as np
import pytest

from scipy import linalg
from scipy.sparse.linalg import aslinearoperator

from deflation_example.coupled_selected_reference import selected_reference
from deflation_example.study_solvers import ArrayReference

pytestmark = pytest.mark.cupy


def test_full_candidates_recover_scaled_eigenvectors_and_transfer():
    rng = np.random.default_rng(21)
    A = rng.normal(size=(12, 12))
    H = A.T @ A + np.eye(12)
    d = np.geomspace(0.2, 8, 12)
    raw = rng.normal(size=(12, 12))
    ref = selected_reference(ArrayReference(raw, {}), aslinearoperator(H), d, 4, chunk=3)
    Z = ref.restrict(np.arange(12))
    np.testing.assert_allclose(Z.T @ (d[:, None] * Z), np.eye(4), atol=2e-12)
    eig, U = linalg.eigh(H / np.sqrt(d[:, None] * d[None, :]))
    actual = np.sqrt(d)[:, None] * Z
    np.testing.assert_allclose(actual @ actual.T, U[:, :4] @ U[:, :4].T, atol=2e-12)
    np.testing.assert_allclose(ref.description["ritz_values"], eig[:4], atol=2e-12)
    np.testing.assert_allclose(ref.restrict(np.array([1, 3, 7])), Z[[1, 3, 7]])
    assert sum(ref.description["construction_components_seconds"].values()) == pytest.approx(
        ref.description["construction_seconds"]
    )


@pytest.mark.parametrize("zero", [False, True])
def test_rank_loss_never_creates_replacement_directions(zero):
    raw = np.zeros((6, 4))
    if not zero:
        raw[:, 0] = np.arange(1, 7)
        raw[:, 1] = 2 * raw[:, 0]
    ref = selected_reference(ArrayReference(raw, {}), aslinearoperator(np.eye(6)), np.ones(6), 4)
    assert ref.rank == (0 if zero else 1)
    assert ref.restrict(np.arange(6)).shape == (6, ref.rank)


def test_full_rank_rotation_preserves_exact_coarse_correction():
    rng = np.random.default_rng(91)
    C = rng.normal(size=(14, 5))
    A = rng.normal(size=(14, 14))
    H = A.T @ A + np.eye(14)
    r = rng.normal(size=14)
    ref = selected_reference(ArrayReference(C, {}), aslinearoperator(H), H.diagonal(), 5)
    Z = ref.restrict(np.arange(14))
    np.testing.assert_allclose(
        Z @ linalg.solve(Z.T @ H @ Z, Z.T @ r), C @ linalg.solve(C.T @ H @ C, C.T @ r), atol=2e-12
    )


def test_block_action_matches_cpu_and_respects_chunk():
    calls = []
    H = np.diag(np.arange(1.0, 9.0))

    def action(x):
        calls.append(x.shape[1])
        return H @ x

    basis = ArrayReference(np.eye(8), {})
    a = selected_reference(basis, aslinearoperator(H), np.ones(8), 3, chunk=3, block_action=action)
    b = selected_reference(basis, aslinearoperator(H), np.ones(8), 3)
    np.testing.assert_allclose(a.restrict(np.arange(8)), b.restrict(np.arange(8)))
    assert calls == [3, 3, 2]


@pytest.mark.parametrize("diagonal", [np.zeros(4), np.full(4, np.nan), np.ones(3)])
def test_bad_diagonal_rejected(diagonal):
    with pytest.raises(ValueError):
        selected_reference(ArrayReference(np.eye(4), {}), aslinearoperator(np.eye(4)), diagonal, 2)


def test_nonsymmetric_operator_rejected():
    H = np.eye(5)
    H[0, 1] = 0.2
    with pytest.raises(ValueError, match="symmetry"):
        selected_reference(ArrayReference(np.eye(5), {}), aslinearoperator(H), np.ones(5), 2)


@pytest.mark.gpu
def test_coupled_gpu_block_matches_full_cpu_operator():
    from deflation_example.coupled_selected_reference import CoupledBlockAction
    from deflation_example.coupled_derivatives import GaussNewtonOperator
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem([0.2, 0.35], consistent=True, streamline_rule="smooth_p8")
    ev = problem.evaluate(np.full(problem.size, 0.1))
    H = GaussNewtonOperator(ev.jacobian, problem.weights, problem.alpha)
    action = CoupledBlockAction(H)
    try:
        x = np.random.default_rng(12).normal(size=(problem.size, 3))
        np.testing.assert_allclose(action(x), H @ x, atol=1e-10, rtol=1e-10)
    finally:
        action.close()


@pytest.mark.parametrize("selection", ["nominal_coupled", "preconditioned_coupled"])
def test_configured_selection_is_fixed_at_initial_state(monkeypatch, selection):
    from deflation_example.coupled_selected_reference import configured_selected_reference
    import deflation_example.coupled_reference as references
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem(
        [0.2, 0.35], consistent=True, streamline_rule="smooth_p8", uniform_capacity=True
    )
    candidate = ArrayReference(np.eye(problem.size)[:, :6], {})
    monkeypatch.setattr(references, "configured_reference", lambda *args: candidate)
    cfg = {
        "reference_selection": selection,
        "reference_candidates": 6,
        "reference_transfer": "full",
        "rank": 3,
        "queries": [
            {"target": 0, "upper_K": problem.temperature_offset + 0.3},
            {"target": 1, "upper_K": problem.temperature_offset + 0.3},
        ],
        "lower_K": problem.temperature_offset - 0.1,
        "temperature_margin_K": 0.0,
        "device": "cpu",
    }
    selected = configured_selected_reference(
        problem, cfg, {}, initial_state=np.full(problem.size, 0.1)
    )
    assert selected.rank == 3
    original = selected.restrict(np.arange(problem.size))
    selected.restrict(np.arange(0, problem.size, 2))
    np.testing.assert_array_equal(original, selected.restrict(np.arange(problem.size)))


@pytest.mark.gpu
@pytest.mark.parametrize("width", [1, 2, 5, 20])
def test_coarse_chunks_preserve_correction_and_partition_setup_time(width):
    from deflation_example.coupled_selected_reference import CoupledBlockAction
    from deflation_example.coupled_hybrid_coarse import CudaCoarseSpace
    from deflation_example.coupled_derivatives import GaussNewtonOperator
    from deflation_example.solvers import CpuCoarseSpace
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem([0.2, 0.35], consistent=True, streamline_rule="smooth_p8")
    ev = problem.evaluate(np.full(problem.size, 0.1))
    H = GaussNewtonOperator(ev.jacobian, problem.weights, problem.alpha)
    I = np.arange(0, problem.size, 2)
    B = H.restrict(I)
    rng = np.random.default_rng(912)
    Z, rhs = rng.normal(size=(len(I), 3)), rng.normal(size=len(I))
    host = CpuCoarseSpace(B, Z, 1e10)
    action = CoupledBlockAction(H)
    calls = []
    apply = action.operator.restrict(I)

    def block(x):
        calls.append(x.shape[1])
        return apply(x)

    try:
        device = CudaCoarseSpace(action.jacobian.cp, block, len(I), Z, 1e10, chunk=width)
        np.testing.assert_allclose(device.correct(rhs), host.correct(rhs), rtol=1e-10, atol=1e-10)
        report = device.report()
        assert max(calls) <= width
        assert sum(report["setup_components_seconds"].values()) == pytest.approx(
            report["setup_seconds"]
        )
        assert all(v >= 0 for v in report["setup_components_seconds"].values())
    finally:
        action.close()
