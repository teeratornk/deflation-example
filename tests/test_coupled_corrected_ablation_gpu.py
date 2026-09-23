"""Exercise the actual construction/transfer controls on the corrected operator."""

import numpy as np
import pytest


from deflation_example.coupled_derivatives import GaussNewtonOperator
from deflation_example.coupled_hybrid_solver import HybridCoupledSolver
from deflation_example.coupled_reference import configured_reference
from deflation_example.solvers import independent_residual
from test_coupled_derivatives import small_coupled_problem

pytestmark = pytest.mark.cupy


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["full", "sequential", "tensor", "recycling"])
def test_corrected_hybrid_variants_preserve_original_accuracy_on_released_nodes(variant):
    pytest.importorskip("cupy")
    problem = small_coupled_problem(
        [0.2, 0.35], uniform_capacity=True, consistent=True, reference_stabilization="matched"
    )
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
    diagonal = problem.preconditioning_diagonal(evaluation)
    cfg = {
        "rank": 2,
        "reference_transfer": "sequential" if variant == "sequential" else "full",
        "reference_construction": "tensor" if variant == "tensor" else "mode_dependent",
    }
    reference = None if variant == "recycling" else configured_reference(problem, cfg, {})
    solver = HybridCoupledSolver(
        "recycling" if variant == "recycling" else "reference",
        reference=reference,
        rank=2,
        window=4,
        block_min_columns=2,
        block_max_columns=2,
        coarse_device="cuda",
        rtol=1e-10,
        cg_factor=0.1,
        residual_policy="refine",
    )
    try:
        for indices in (np.arange(0, problem.size, 2), np.arange(problem.size)):
            B = H.restrict(indices)
            B.diagonal = lambda: diagonal[indices]
            exact = np.linspace(-0.1, 0.2, len(indices))
            rhs = B @ exact
            result, timing = solver.solve(B, rhs, indices)
            assert result.status == "converged"
            assert independent_residual(B, result.x, rhs) <= 1e-10
            assert 0 <= result.rank <= 2
            assert timing["total_seconds"] == pytest.approx(
                sum(timing["components_seconds"].values())
            )
            if variant == "sequential" and len(indices) == problem.size:
                np.testing.assert_array_equal(reference.restrict(indices)[1::2], 0)
    finally:
        solver.close()
