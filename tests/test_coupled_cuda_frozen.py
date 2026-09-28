"""CUDA frozen sweeps, deflated solves and complete small quadratic checks."""

import numpy as np
import pytest
from scipy import sparse

from deflation_example.coupled_frozen_preconditioner import (
    FrozenSweepPreconditioner,
    frozen_preconditioner_factory,
)

pytestmark = [pytest.mark.gpu, pytest.mark.cupy]


@pytest.mark.parametrize("device", ["cuda", "hybrid"])
def test_projected_nonlinear_policy_restores_final_accuracy(device):
    pytest.importorskip("cupy")
    from deflation_example.coupled_cuda_solver import CudaCoupledSolver
    from deflation_example.coupled_hybrid_solver import HybridCoupledSolver
    from deflation_example.coupled_trust import minimize_trust
    from deflation_example.study_solvers import StudySolver, ArrayReference
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem([0.2, 0.35])
    desired = np.linspace(-0.2, 0.4, problem.size)
    reference = ArrayReference(np.eye(problem.size)[:, :3], {})
    if device == "cuda":
        solver = CudaCoupledSolver(
            "reference",
            rank=3,
            reference=reference,
            frozen_layout="block_diagonal",
            rtol=1e-10,
            residual_policy="refine",
        )
    else:
        solver = HybridCoupledSolver(
            "reference",
            rank=3,
            reference=reference,
            block_device="cpu",
            coarse_device="cuda",
            rtol=1e-10,
            residual_policy="refine",
        )
    cpu = StudySolver("jacobi", rtol=1e-10, residual_policy="refine")
    try:
        expected = minimize_trust(problem, desired, -0.05, 0.15, cpu, max_iterations=80)
        actual = minimize_trust(
            problem,
            desired,
            -0.05,
            0.15,
            solver,
            max_iterations=80,
            qp_solver="projected",
            accuracy="adaptive_projected",
        )
        assert actual.status == expected.status == "converged", (actual.status, actual.kkt)
        assert max(actual.kkt.values()) <= 1e-8
        np.testing.assert_allclose(actual.evaluation.state, expected.evaluation.state, atol=2e-7)
        assert solver.rtol == 1e-10
    finally:
        cpu.close()
        solver.close()


def test_grouped_factors_preserve_permutations_transpose_and_owned_results():
    cp = pytest.importorskip("cupy")
    from scipy.sparse.linalg import splu
    from deflation_example.coupled_triangular import PersistentSuperLU

    rng = np.random.default_rng(410)
    matrices = [sparse.csc_matrix(rng.normal(size=(n, n)) + np.eye(n)) for n in (5, 8, 4)]
    factors = [splu(A) for A in matrices]
    device = PersistentSuperLU.block_diagonal(factors)
    try:
        for transpose in ("N", "T"):
            for columns in (1, 3, 2, 1):
                rhs = rng.normal(size=(17, columns))
                parts = np.split(rhs, [5, 13])
                expected = np.concatenate(
                    [f.solve(r, trans=transpose) for f, r in zip(factors, parts, strict=True)]
                )
                actual = device.solve(cp.asarray(rhs), trans=transpose)
                saved = actual.copy()
                np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=1e-11, rtol=1e-11)
                device.solve(cp.asarray(rhs * 2), trans=transpose)
                cp.testing.assert_array_equal(actual, saved)
    finally:
        device.close()


@pytest.mark.parametrize("sweeps", [1, 3, 4])
@pytest.mark.parametrize("layout", ["serial", "block_diagonal"])
def test_device_sweeps_match_host_vectors_blocks_and_do_not_mutate(sweeps, layout):
    cp = pytest.importorskip("cupy")
    from deflation_example.coupled_cuda_preconditioner import CudaFrozenSweepPreconditioner

    P = sparse.diags([-0.2 * np.ones(11), 2 * np.ones(12), -0.2 * np.ones(11)], [-1, 0, 1])
    host = FrozenSweepPreconditioner(P, np.arange(12) % 3, sweeps=sweeps)
    device = CudaFrozenSweepPreconditioner(host, layout=layout)
    rng = np.random.default_rng(48)
    try:
        for shape in ((12,), (12, 3), (12, 1), (12, 3)):
            rhs = rng.normal(size=shape)
            gpu_rhs = cp.asarray(rhs)
            actual = device(gpu_rhs)
            saved = actual.copy()
            np.testing.assert_allclose(cp.asnumpy(actual), host(rhs), rtol=2e-12, atol=1e-12)
            np.testing.assert_array_equal(cp.asnumpy(gpu_rhs), rhs)
            device(2 * gpu_rhs)
            np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(saved))
        a, b = rng.normal(size=(2, 12))
        ka, kb = cp.asnumpy(device(a)), cp.asnumpy(device(b))
        assert a @ ka > 0
        assert a @ kb == pytest.approx(b @ ka, abs=1e-11)
        assert device.storage_bytes() > 0
        with pytest.raises(ValueError, match="match"):
            device(cp.ones(11))
        with pytest.raises(ValueError, match="real"):
            device(cp.ones(12, dtype=cp.complex128))
    finally:
        device.close()
    assert device.storage_bytes() == 0
    device.close()
    with pytest.raises(RuntimeError, match="closed"):
        device(np.ones(12))
    host.correction = lambda x: x
    with pytest.raises(ValueError, match="unmodified"):
        CudaFrozenSweepPreconditioner(host)


def quadratic():
    from deflation_example.coupled_derivatives import GaussNewtonOperator
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem([0.2, 0.35], consistent=True, streamline_rule="smooth_p8")
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha, damping=0.02)
    diagonal = problem.preconditioning_diagonal(evaluation, 0.02)
    factory = frozen_preconditioner_factory(problem, evaluation, damping=0.02)
    return problem, H, diagonal, factory


@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_frozen_cuda_solvers_masks_cache_residuals_and_timing(method):
    pytest.importorskip("cupy")
    from deflation_example.coupled_cuda_solver import CudaCoupledSolver
    from deflation_example.solvers import independent_residual
    from deflation_example.study_solvers import ArrayReference, StudySolver

    problem, H, diagonal, factory = quadratic()
    rng = np.random.default_rng(384)
    reference = ArrayReference(rng.normal(size=(problem.size, 3)), {"construction": "test"})
    options = dict(
        reference=reference,
        rank=3,
        window=6,
        rtol=1e-10,
        maxiter=1000,
        cg_factor=0.1,
        residual_policy="refine",
    )
    cpu, gpu = StudySolver(method, **options), CudaCoupledSolver(method, **options)
    previous = None
    try:
        for indices in (
            np.arange(0, problem.size, 2),
            np.arange(1, problem.size),
            np.arange(problem.size),
        ):
            B = H.restrict(indices)
            B.diagonal = lambda: diagonal[indices]
            B.preconditioner = factory(indices)
            exact = rng.normal(size=len(indices))
            rhs = B @ exact
            expected, _ = cpu.solve(B, rhs, indices)
            actual, timing = gpu.solve(B, rhs, indices)
            assert expected.status == actual.status == "converged"
            assert independent_residual(B, actual.x, rhs) <= 1e-10
            np.testing.assert_allclose(actual.x, expected.x, rtol=1e-7, atol=1e-8)
            assert timing["frozen_preconditioner_device_bytes"] > 0
            assert sum(timing["components_seconds"].values()) == pytest.approx(
                timing["total_seconds"], abs=1e-8
            )
            if previous is not None:
                assert previous.closed
            previous = gpu.device_preconditioner
            _, second = gpu.solve(B, 2 * rhs, indices)
            assert gpu.device_preconditioner is previous
            assert not second["frozen_preconditioner_uploaded"]
            calls = gpu.kernel_calls
            result, warm = gpu.solve(B, rhs, indices, initial=exact)
            assert result.iterations == 0 and warm["initial_guess_accepted"]
            assert gpu.kernel_calls == calls
    finally:
        cpu.close()
        gpu.close()
    assert previous.closed


@pytest.mark.parametrize("direction_rtol", [1e-2, 1e-8])
@pytest.mark.parametrize("rank", [0, 3])
def test_complete_projected_quadratic_matches_cpu_at_unchanged_kkt(direction_rtol, rank):
    pytest.importorskip("cupy")
    from deflation_example.box_projected_cg import box_projected_cg
    from deflation_example.coupled_cuda_solver import CudaCoupledSolver
    from deflation_example.coupled_optimizer import box_kkt
    from deflation_example.study_solvers import ArrayReference, StudySolver

    problem, H, diagonal, factory = quadratic()
    rng = np.random.default_rng(123)
    gradient = problem.weights * rng.normal(size=problem.size)
    reference = ArrayReference(rng.normal(size=(problem.size, rank)), {}) if rank else None
    results = []
    for cls in (StudySolver, CudaCoupledSolver):
        solver = cls(
            "reference" if rank else "jacobi",
            reference=reference,
            rank=rank,
            rtol=direction_rtol,
            maxiter=2000,
            residual_policy="refine",
            cg_factor=0.1,
        )
        try:
            results.append(
                box_projected_cg(
                    H,
                    gradient,
                    diagonal,
                    -0.05,
                    0.1,
                    solver,
                    tolerance=1e-8,
                    preconditioner_factory=factory,
                    kkt_evaluator=lambda x, g: box_kkt(x, g / problem.weights, -0.05, 0.1),
                )
            )
        finally:
            solver.close()
    assert all(r.status == "converged" for r in results), [(r.status, r.kkt) for r in results]
    assert all(max(r.kkt.values()) <= 1e-8 for r in results)
    np.testing.assert_allclose(results[0].x, results[1].x, atol=2e-8)


def test_cooperative_budget_stop_and_progress_keep_original_residual():
    pytest.importorskip("cupy")
    from deflation_example.coupled_cuda_solver import CudaCoupledSolver
    from deflation_example.solvers import independent_residual

    problem, H, diagonal, factory = quadratic()
    indices = np.arange(problem.size)
    B = H.restrict(indices)
    B.diagonal = lambda: diagonal
    B.preconditioner = factory(indices)
    events = []
    solver = CudaCoupledSolver(
        "jacobi",
        rtol=1e-10,
        residual_policy="refine",
        progress_callback=events.append,
        stop_requested=lambda: bool(events),
    )
    try:
        rhs = np.arange(problem.size, dtype=float) + 1
        result, _ = solver.solve(B, rhs, indices)
        assert result.status == "budget_exhausted" and result.iterations == 0
        assert result.residual == pytest.approx(independent_residual(B, result.x, rhs))
        assert events[-1]["stage"] == "final"
        assert events[-1]["residual_kind"] == "independently_recomputed_cpu"
    finally:
        solver.close()
