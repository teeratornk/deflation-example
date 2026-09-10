"""Complete CPU optimization with matched starts and all non-AMG adapters."""

import numpy as np
import pytest

from deflation_example.benchmark_extended import target_parameters, target_values
from deflation_example.problems import build_problem
from deflation_example.solvers import pdas
from deflation_example.spectral import analytical_reference
from deflation_example.study_solvers import ArrayReference, StudySolver


def run_small_sequence(method, device="cpu", torch=None, api=None):
    problem = build_problem("cht", 4)
    basis, _ = analytical_reference(4, 3, 5)
    adapter = StudySolver(
        method,
        device=device,
        rank=5,
        window=10,
        reference=ArrayReference(basis, "analytical"),
        maxiter=2000,
        cg_factor=0.1,
        torch=torch,
        api=api,
    )
    previous = None
    calls = []
    for query in range(3):
        desired = target_values(problem.coordinates, target_parameters(query, 3))
        state = np.zeros(64) if previous is None else previous["y"].copy()

        def solve(B, b, I):
            result, timing = adapter.solve(B, b, I, state[I])
            state[:] = 0.0001
            state[I] = result.x
            calls.append((result, timing))
            return result

        current = pdas(
            problem.H,
            desired,
            0.0001,
            initial_active=None if previous is None else previous["active"],
            tolerance=1e-8,
            linear_solver=solve,
        )
        assert current["status"] == "converged"
        assert max(current["kkt"].values()) <= 1e-8
        direct = pdas(problem.H, desired, 0.0001, tolerance=1e-8)
        np.testing.assert_allclose(current["y"], direct["y"], rtol=1e-7, atol=1e-10)
        previous = current
    adapter.close()
    assert all(result.residual <= 1e-10 for result, _ in calls)
    assert calls[0][1]["newly_inactive"] is None
    for result, timing in calls:
        assert timing["callback_seconds"] >= timing["total_seconds"]
        assert timing["restricted_basis_bytes"] >= 0
        assert timing["iteration_rtol"] == pytest.approx(1e-11, rel=1e-12, abs=0)
        assert timing["acceptance_rtol"] == pytest.approx(1e-10, rel=1e-12, abs=0)
        if method == "jacobi":
            assert result.rank == 0 and timing["input_basis_columns"] == 0
        if method == "recycling":
            assert timing["completion"]["policy"].endswith("jacobi-ritz-v1")
    if method == "recycling":
        assert calls[0][0].rank == 0
        assert any(result.rank > 0 for result, _ in calls[1:])


@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_complete_cpu_sequence_and_history_costs(method):
    run_small_sequence(method)


@pytest.mark.gpu
def test_complete_four_way_gpu_sequences():
    torch = pytest.importorskip("torch")
    api = pytest.importorskip("pyamgx")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    api.initialize()
    try:
        for method in ("jacobi", "reference", "recycling", "amgx"):
            run_small_sequence(method, "cuda", torch, api)
    finally:
        api.finalize()
