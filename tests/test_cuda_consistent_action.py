"""The device must apply the same operator the processor does, weighting included.

With a consistent stabilisation the control is recovered through a factored source
action, so the tangent is S^-1 R and the transpose R^T S^-T. The device jacobian
mirrors the processor's blocks, and it carried every one of them except that action:
it would have applied R alone. Not a slower answer, a different operator, and one
that still converges, which is the kind of wrong that reaches a paper.

These tests would have failed on that. They need a device and skip without one.
"""

import numpy as np
import pytest

from deflation_example.coupled_cuda import CudaControlJacobian, CudaGaussNewton
from deflation_example.coupled_derivatives import GaussNewtonOperator
from test_coupled_derivatives import small_coupled_problem


def consistent_problem(steps):
    problem = small_coupled_problem(steps, uniform_capacity=True, inlet=0.3)
    problem.consistent_stabilization = True
    problem.assembly = problem.assemble(problem.initial_flow.velocity)
    problem.reference_assembly = problem.assemble(problem.initial_flow.velocity, consistent=False)
    return problem


@pytest.mark.gpu
@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_the_device_applies_the_weighted_operator_the_processor_applies(steps):
    cp = pytest.importorskip("cupy")
    problem = consistent_problem(steps)
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    J = evaluation.jacobian
    assert J.source_factors is not None, "this problem must exercise the factored action"
    device = CudaControlJacobian(J)
    assert device.source_factors is not None and len(device.source_factors) == J.slabs
    rng = np.random.default_rng(4242)
    vectors = rng.normal(size=(problem.size, 3))
    for x in (vectors[:, 0], vectors):
        np.testing.assert_allclose(cp.asnumpy(device.apply(x)), J @ x, rtol=2e-11, atol=1e-10)
        np.testing.assert_allclose(
            cp.asnumpy(device.apply(x, transpose=True)), J.T @ x, rtol=2e-11, atol=1e-10
        )
    # The adjoint identity has to hold on the device in its own right.
    a, b = vectors[:, 0], vectors[:, 1]
    forward = float(b @ cp.asnumpy(device.apply(a)))
    backward = float(a @ cp.asnumpy(device.apply(b, transpose=True)))
    assert abs(forward - backward) <= 1e-9 * max(abs(forward), abs(backward), 1e-30)
    H = GaussNewtonOperator(J, problem.weights, problem.alpha, damping=0.03)
    gpu = CudaGaussNewton(device, problem.weights, problem.alpha, damping=0.03)
    np.testing.assert_allclose(cp.asnumpy(gpu.apply(vectors)), H @ vectors, rtol=3e-11, atol=1e-10)
    assert device.storage_bytes() > 0
    device.close()


@pytest.mark.gpu
def test_the_device_transpose_does_not_write_through_to_its_caller():
    """cp.asarray returns a device array unchanged, so a careless normalise mutates it."""
    cp = pytest.importorskip("cupy")
    problem = consistent_problem([0.2, 0.35])
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    device = CudaControlJacobian(evaluation.jacobian)
    x = cp.asarray(np.random.default_rng(7).normal(size=problem.size))
    keep = x.copy()
    device.apply(x, transpose=True)
    np.testing.assert_array_equal(cp.asnumpy(x), cp.asnumpy(keep))
    # And applying it twice must give the same answer, which it cannot if the first
    # application consumed its own input.
    first = cp.asnumpy(device.apply(x, transpose=True))
    second = cp.asnumpy(device.apply(x, transpose=True))
    np.testing.assert_array_equal(first, second)
    device.close()


@pytest.mark.gpu
def test_dropping_the_action_would_change_the_answer():
    """Otherwise the test above could pass against an operator that ignores it."""
    cp = pytest.importorskip("cupy")
    problem = consistent_problem([0.2, 0.35])
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    J = evaluation.jacobian
    device = CudaControlJacobian(J)
    x = np.random.default_rng(31).normal(size=problem.size)
    weighted = cp.asnumpy(device.apply(x))
    device.source_factors = None
    unweighted = cp.asnumpy(device.apply(x))
    relative = np.linalg.norm(weighted - unweighted) / max(np.linalg.norm(weighted), 1e-30)
    assert relative > 1e-3, f"the action moves the answer by only {relative}"
    device.source_factors = tuple()
    device.close()
