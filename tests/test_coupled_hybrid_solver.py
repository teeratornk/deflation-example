"""The hybrid policy preserves CPU verification and matched block acceleration."""

import numpy as np
import pytest

from deflation_example.coupled_derivatives import GaussNewtonOperator
from deflation_example.coupled_hybrid_solver import HybridCoupledSolver
from deflation_example.solvers import independent_residual
from deflation_example.study_solvers import ArrayReference
from test_coupled_derivatives import small_coupled_problem


def system():
    problem = small_coupled_problem([0.2, 0.35])
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
    diagonal = problem.preconditioning_diagonal(evaluation)
    return problem, H, diagonal


@pytest.mark.parametrize("coarse_device", ["cpu", "cuda"])
def test_rank_zero_and_initial_guard_do_not_upload_factors(monkeypatch, coarse_device):
    import deflation_example.coupled_hybrid_solver as hybrid

    def forbidden(*args, **kwargs):
        raise AssertionError("A rank-zero vector solve must not initialize CUDA factors")

    monkeypatch.setattr(hybrid, "CudaControlJacobian", forbidden)
    problem, H, diagonal = system()
    indices = np.arange(problem.size)
    B = H.restrict(indices)
    B.diagonal = lambda: diagonal
    exact = np.linspace(-1, 1, problem.size)
    solver = HybridCoupledSolver(
        "jacobi",
        rank=0,
        rtol=1e-10,
        cg_factor=0.1,
        residual_policy="refine",
        coarse_device=coarse_device,
    )
    try:
        for initial in (None, exact):
            result, timing = solver.solve(B, B @ exact, indices, initial=initial)
            assert result.status == "converged"
            assert independent_residual(B, result.x, B @ exact) <= 1e-10
            assert not timing["hybrid_block_processing"]["calls"]
            assert solver.device_jacobian is None
            assert timing.get("hybrid_coarse_correction", {}).get("applications", 0) == 0
            if initial is not None:
                assert result.iterations == 0
    finally:
        solver.close()


def test_hybrid_rejects_unknown_coarse_device():
    with pytest.raises(ValueError):
        HybridCoupledSolver("jacobi", rank=0, coarse_device="tpu")


def test_cpu_coarse_space_matches_default_reference_solve():
    # With CUDA unavailable the host coarse space must equal the in-line kernel.
    problem, H, diagonal = system()
    rng = np.random.default_rng(7)
    reference = ArrayReference(rng.normal(size=(problem.size, 3)), {"construction": "test"})
    indices = np.arange(problem.size)
    B = H.restrict(indices)
    B.diagonal = lambda: diagonal
    rhs = B @ rng.normal(size=problem.size)
    results = []
    for coarse_device in ("cpu", "cpu"):
        solver = HybridCoupledSolver(
            "reference",
            reference=reference,
            rank=3,
            rtol=1e-10,
            cg_factor=0.1,
            residual_policy="refine",
            coarse_device=coarse_device,
        )
        try:
            result, timing = solver.solve(B, rhs, indices)
        finally:
            solver.close()
        assert result.status == "converged" and result.rank == 3
        assert "hybrid_coarse_correction" not in timing
        results.append(result)
    assert np.array_equal(results[0].x, results[1].x)
    assert results[0].iterations == results[1].iterations


@pytest.mark.gpu
@pytest.mark.parametrize("coarse_device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_complete_hybrid_runner_records_device_memory_and_costs(
    monkeypatch, tmp_path, method, coarse_device
):
    pytest.importorskip("cupy")
    pytest.importorskip("pynvml")
    from omegaconf import OmegaConf
    from deflation_example import coupled_sequence as sequence
    from deflation_example.coupled_report import validate_record
    from test_coupled_sequence import configuration

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    config = configuration(problem)
    config.update(
        method=method,
        device="hybrid",
        hybrid_block_min_columns=2,
        hybrid_coarse_device=coarse_device,
        memory_interval=0.01,
        repetition=0,
        rank=3,
        recycle_window=6,
        inner_tolerance=1e-11,
        inner_cap=1000,
        threads=1,
        evaluation_progress=True,
        linear_progress=True,
        output=str(tmp_path / method),
    )
    monkeypatch.setattr(
        sequence,
        "load_problem",
        lambda cfg: (
            problem,
            {"baseline_sha256": "test", "configuration": {}, "input_sha256": {}, "seconds": 0.25},
        ),
    )
    monkeypatch.setattr(
        sequence,
        "configured_reference",
        lambda *args: ArrayReference(np.eye(problem.size)[:, :3], {"construction": "test"}),
    )
    report = sequence.run(OmegaConf.create(config))
    assert report["status"] == "complete", report.get("error_type")
    assert report["all_problems_verified"]
    assert validate_record(report)
    assert "CPU vectors" in report["device"]["operator_policy"]
    assert report["memory"]["peak_gpu_process_bytes"] > 0
    assert sum(report["components_seconds"].values()) == pytest.approx(report["sequence_seconds"])


@pytest.mark.gpu
@pytest.mark.parametrize("coarse_device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", ["reference", "recycling"])
def test_gpu_blocks_preserve_original_residual_and_reuse_factors(method, coarse_device):
    pytest.importorskip("cupy")
    problem, H, diagonal = system()
    rng = np.random.default_rng(96)
    reference = ArrayReference(rng.normal(size=(problem.size, 3)), {"construction": "test"})
    solver = HybridCoupledSolver(
        method,
        reference=reference,
        rank=3,
        window=6,
        block_min_columns=2,
        rtol=1e-10,
        cg_factor=0.1,
        residual_policy="refine",
        coarse_device=coarse_device,
    )
    try:
        identity = None
        for indices in (np.arange(0, problem.size, 2), np.arange(problem.size)):
            B = H.restrict(indices)
            B.diagonal = lambda: diagonal[indices]
            exact = rng.normal(size=len(indices))
            rhs = B @ exact
            result, timing = solver.solve(B, rhs, indices)
            assert result.status == "converged"
            assert independent_residual(B, result.x, rhs) <= 1e-10
            np.testing.assert_allclose(result.x, exact, atol=1e-8)
            log = timing["hybrid_block_processing"]
            assert log["seconds"] <= timing["callback_seconds"]
            assert sum(timing["components_seconds"].values()) == pytest.approx(
                timing["total_seconds"]
            )
            if coarse_device == "cuda" and result.rank:
                # The operator-basis product runs inside the resident coarse space,
                # so block calls appear only for the factor upload itself.
                coarse = timing["hybrid_coarse_correction"]
                assert coarse["device"] == "cuda" and coarse["applications"] >= 1
                assert coarse["seconds"] <= timing["callback_seconds"]
                assert coarse["resident_operator_product_bytes"] > 0
                assert timing["cached_operator_product_bytes"] == 0
                assert coarse["spaces"][-1]["rank"] == result.rank
                assert all(row["kind"] == "factor_upload" for row in log["calls"])
            else:
                assert log["calls"] and log["seconds"] > 0
                assert "hybrid_coarse_correction" not in timing
            if identity is None:
                identity = solver.device_jacobian
                assert sum(row["factor_upload"] for row in log["calls"]) == 1
            else:
                assert solver.device_jacobian is identity
                assert not any(row["factor_upload"] for row in log["calls"])
        result, timing = solver.solve(B, rhs, indices, initial=exact)
        assert result.iterations == 0
        assert not timing["hybrid_block_processing"]["calls"]
    finally:
        solver.close()
    assert solver.device_jacobian is None


@pytest.mark.gpu
def test_device_coarse_space_matches_host_coarse_space():
    pytest.importorskip("cupy")
    problem, H, diagonal = system()
    rng = np.random.default_rng(1207)
    reference = ArrayReference(rng.normal(size=(problem.size, 3)), {"construction": "test"})
    indices = np.arange(problem.size)
    B = H.restrict(indices)
    B.diagonal = lambda: diagonal
    exact = rng.normal(size=problem.size)
    rhs = B @ exact
    solutions = {}
    for coarse_device in ("cpu", "cuda"):
        solver = HybridCoupledSolver(
            "reference",
            reference=reference,
            rank=3,
            block_min_columns=2,
            rtol=1e-10,
            cg_factor=0.1,
            residual_policy="refine",
            coarse_device=coarse_device,
        )
        try:
            result, timing = solver.solve(B, rhs, indices)
        finally:
            solver.close()
        assert result.status == "converged" and result.rank == 3
        solutions[coarse_device] = result
    np.testing.assert_allclose(solutions["cuda"].x, solutions["cpu"].x, atol=1e-8)
    assert abs(solutions["cuda"].iterations - solutions["cpu"].iterations) <= 2
    assert solutions["cuda"].coarse_condition == pytest.approx(
        solutions["cpu"].coarse_condition, rel=1e-6
    )


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_hybrid_complete_trajectory_matches_cpu_with_secant_updates(method):
    pytest.importorskip("cupy")
    from deflation_example.coupled_sequence import optimize_targets
    from test_coupled_sequence import configuration
    from test_coupled_optimizer import solver as cpu_solver

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    config = configuration(problem)
    reference = ArrayReference(np.eye(problem.size)[:, :3], {"construction": "test"})
    expected_solver = cpu_solver(method, reference)
    hybrid_solver = HybridCoupledSolver(
        method,
        reference=reference,
        rank=3,
        window=6,
        block_min_columns=2,
        rtol=1e-11,
        cg_factor=0.1,
        maxiter=1000,
        residual_policy="refine",
    )
    try:
        expected, cpu_fields = optimize_targets(problem, expected_solver, config)
        actual, hybrid_fields = optimize_targets(problem, hybrid_solver, config)
        assert all(row["verified"] for row in expected + actual)
        assert actual[1]["warm_start_used"]
        for left, right, a, b in zip(expected, actual, cpu_fields, hybrid_fields, strict=True):
            assert max(right["kkt"].values()) <= 1e-8
            assert right["objective"] == pytest.approx(left["objective"], abs=1e-10)
            np.testing.assert_allclose(a["state"], b["state"], atol=2e-8)
            np.testing.assert_allclose(a["control"], b["control"], atol=2e-7)
        if method == "jacobi":
            assert hybrid_solver.device_jacobian is None
        else:
            assert hybrid_solver.device_jacobian is not None
    finally:
        expected_solver.close()
        hybrid_solver.close()


@pytest.mark.gpu
def test_hybrid_replaces_factors_when_nonlinear_jacobian_changes():
    pytest.importorskip("cupy")
    from deflation_example.coupled_secant import SecantGaussNewton

    problem = small_coupled_problem([0.2, 0.35])
    reference = ArrayReference(np.eye(problem.size)[:, :3], {"construction": "test"})
    solver = HybridCoupledSolver(
        "reference",
        reference=reference,
        rank=3,
        block_min_columns=2,
        rtol=1e-10,
        cg_factor=0.1,
        residual_policy="refine",
    )
    rng = np.random.default_rng(208)
    previous = None
    try:
        for offset in (0.02, 0.08):
            evaluation = problem.evaluate(np.linspace(offset, offset + 0.05, problem.size))
            H = SecantGaussNewton(
                GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha),
                [(rng.normal(size=problem.size), rng.normal(size=problem.size))],
            )
            indices = np.arange(problem.size)
            B = H.restrict(indices)
            B.diagonal = lambda: problem.preconditioning_diagonal(evaluation)
            rhs = B @ rng.normal(size=problem.size)
            result, timing = solver.solve(B, rhs, indices)
            assert result.status == "converged"
            assert independent_residual(B, result.x, rhs) <= 1e-10
            assert any(row["factor_upload"] for row in timing["hybrid_block_processing"]["calls"])
            assert solver.device_jacobian is not previous
            if previous is not None:
                assert all(not factor.plans for factor in previous.factors)
            previous = solver.device_jacobian
    finally:
        solver.close()
