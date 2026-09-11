"""CUDA derivative parity with independently evaluated CPU actions."""

import numpy as np
import pytest

from deflation_example.coupled_cuda import CudaControlJacobian, CudaGaussNewton
from deflation_example.coupled_derivatives import GaussNewtonOperator
from test_coupled_derivatives import small_coupled_problem


@pytest.mark.gpu
@pytest.mark.parametrize("steps", [None, [0.2, 0.35, 0.15]])
def test_cuda_coupled_tangent_adjoint_normal_products_and_restriction(steps):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("A CUDA device is required")
    problem = small_coupled_problem(steps)
    state = np.linspace(0.03, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    J = evaluation.jacobian
    device = CudaControlJacobian(J)
    rng = np.random.default_rng(92)
    vectors = rng.normal(size=(problem.size, 3))
    for x in (vectors[:, 0], vectors):
        np.testing.assert_allclose(cp.asnumpy(device.apply(x)), J @ x, rtol=2e-12, atol=1e-11)
        np.testing.assert_allclose(
            cp.asnumpy(device.apply(x, transpose=True)), J.T @ x, rtol=2e-12, atol=1e-11
        )
    H = GaussNewtonOperator(J, problem.weights, problem.alpha, damping=0.03)
    gpu = CudaGaussNewton(device, problem.weights, problem.alpha, damping=0.03)
    np.testing.assert_allclose(cp.asnumpy(gpu.apply(vectors)), H @ vectors, rtol=3e-12, atol=1e-11)
    indices = np.arange(0, problem.size, 2)
    actual = cp.asnumpy(gpu.apply(vectors[indices], indices=indices))
    np.testing.assert_allclose(
        actual, H.restrict(indices) @ vectors[indices], rtol=3e-12, atol=1e-11
    )
    assert device.storage_bytes() > 0
    a, b = vectors[:, :2].T
    assert abs(b @ cp.asnumpy(gpu.apply(a)) - a @ cp.asnumpy(gpu.apply(b))) < 1e-10
    analyzed = [factor.analysis_count for factor in device.factors]
    first = cp.asnumpy(device.apply(a))
    device.apply(b)
    np.testing.assert_allclose(cp.asnumpy(device.apply(a)), first, atol=1e-12)
    assert [factor.analysis_count for factor in device.factors] == analyzed
    device.close()


@pytest.mark.gpu
def test_cuda_secant_curvature_matches_cpu_full_and_inactive_products():
    cp = pytest.importorskip("cupy")
    from deflation_example.coupled_secant import SecantGaussNewton

    problem = small_coupled_problem([0.2, 0.35])
    ev = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    base = GaussNewtonOperator(ev.jacobian, problem.weights, problem.alpha)
    rng = np.random.default_rng(483)
    pairs = [(rng.normal(size=problem.size), rng.normal(size=problem.size)) for _ in range(4)]
    H = SecantGaussNewton(base, pairs)
    device = CudaControlJacobian(ev.jacobian)
    try:
        gpu = CudaGaussNewton(device, problem.weights, problem.alpha, corrections=H.corrections)
        x = rng.normal(size=(problem.size, 3))
        np.testing.assert_allclose(cp.asnumpy(gpu.apply(x)), H @ x, rtol=3e-12, atol=1e-11)
        indices = np.arange(0, problem.size, 2)
        np.testing.assert_allclose(
            cp.asnumpy(gpu.apply(x[indices], indices)),
            H.restrict(indices) @ x[indices],
            rtol=3e-12,
            atol=1e-11,
        )
    finally:
        device.close()


@pytest.mark.gpu
def test_cuda_isothermal_derivative_avoids_factor_upload():
    cp = pytest.importorskip("cupy")
    problem = small_coupled_problem([0.2, 0.35])
    problem.thermal_boundary[:] = 0
    ev = problem.evaluate(np.zeros(problem.size))
    assert ev.jacobian.thermal_only
    device = CudaControlJacobian(ev.jacobian)
    assert not device.factors
    x = np.arange(problem.size, dtype=float)
    np.testing.assert_allclose(cp.asnumpy(device.apply(x)), ev.jacobian @ x, atol=1e-11)
    np.testing.assert_allclose(
        cp.asnumpy(device.apply(x, transpose=True)), ev.jacobian.T @ x, atol=1e-11
    )
    device.close()


@pytest.mark.gpu
def test_persistent_triangular_factors_reuse_and_release_block_plans():
    cp = pytest.importorskip("cupy")
    from scipy import sparse
    from scipy.sparse.linalg import splu
    from deflation_example.coupled_triangular import PersistentSuperLU

    rng = np.random.default_rng(472)
    A = rng.normal(size=(17, 17)) + 2 * np.eye(17)
    factor = splu(sparse.csc_matrix(A))
    gpu = PersistentSuperLU(factor)
    try:
        for columns in (1, 3, 3, 5, 1):
            rhs = rng.normal(size=(17, columns))
            for trans in ("N", "T"):
                result = gpu.solve(cp.asarray(rhs), trans=trans)
                np.testing.assert_allclose(
                    cp.asnumpy(result), factor.solve(rhs, trans=trans), atol=1e-11
                )
        assert set(gpu.plans) == {("N", 1), ("T", 1), ("N", 5), ("T", 5)}
        assert gpu.analysis_count == 12
        prior = gpu.solve(cp.asarray(rhs)).copy()
        gpu.solve(cp.asarray(2 * rhs))
        np.testing.assert_allclose(cp.asnumpy(prior), factor.solve(rhs), atol=1e-11)
    finally:
        gpu.close()
    assert not gpu.plans


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_cuda_coupled_solvers_and_recycling_match_cpu_residuals(method):
    pytest.importorskip("cupy")
    from deflation_example.coupled_cuda_solver import CudaCoupledSolver
    from deflation_example.study_solvers import ArrayReference
    from deflation_example.solvers import independent_residual

    problem = small_coupled_problem([0.2, 0.35])
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
    diagonal = problem.preconditioning_diagonal(evaluation)
    rng = np.random.default_rng(98)
    reference = ArrayReference(rng.normal(size=(problem.size, 3)), {"construction": "test"})
    solver = CudaCoupledSolver(
        method,
        reference=reference,
        rank=3,
        window=6,
        rtol=1e-10,
        maxiter=1000,
        cg_factor=0.1,
        residual_policy="refine",
    )
    try:
        for indices in (
            np.arange(0, problem.size, 2),
            np.arange(1, problem.size),
            np.arange(problem.size),
        ):
            B = H.restrict(indices)
            B.diagonal = lambda: diagonal[indices]
            exact = rng.normal(size=len(indices))
            rhs = B @ exact
            result, timing = solver.solve(B, rhs, indices)
            assert result.status == "converged", (result.status, result.residual)
            assert independent_residual(B, result.x, rhs) <= 1e-10
            np.testing.assert_allclose(result.x, exact, atol=1e-8)
            assert sum(timing["components_seconds"].values()) == pytest.approx(
                timing["total_seconds"], abs=1e-9
            )
            if method == "recycling":
                assert solver.history.basis.shape[1] == 3
        # The inherited guard avoids device coarse work for an exact warm start.
        result, timing = solver.solve(B, rhs, indices, initial=exact)
        assert result.iterations == 0
        assert timing["initial_guess_accepted"]
    finally:
        solver.close()
