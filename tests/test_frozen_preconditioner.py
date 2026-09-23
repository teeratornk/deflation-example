"""The velocity-frozen preconditioner changes cost, never the answer."""

import numpy as np
import pytest

from deflation_example.coupled_derivatives import GaussNewtonOperator
from deflation_example.coupled_frozen_preconditioner import (
    FrozenSweepPreconditioner,
    frozen_preconditioner_factory,
)
from deflation_example.coupled_optimizer import minimize_coupled
from deflation_example.mesh_control import WeightedReducedOperator
from deflation_example.solvers import deflated_cg, independent_residual
from deflation_example.study_solvers import StudySolver
from test_coupled_derivatives import small_coupled_problem


def system(consistent=False):
    problem = small_coupled_problem([0.2, 0.35], consistent=consistent)
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
    diagonal = problem.preconditioning_diagonal(evaluation)
    rng = np.random.default_rng(20260921)
    b = rng.standard_normal(problem.size)
    return problem, evaluation, H, diagonal, b


def test_diagonal_path_is_untouched_when_no_preconditioner_is_given():
    _, _, H, diagonal, b = system()
    first = deflated_cg(H, b, diagonal=diagonal, rtol=1e-11)
    second = deflated_cg(H, b, diagonal=diagonal, rtol=1e-11, preconditioner=None)
    assert first.iterations == second.iterations
    assert np.array_equal(first.x, second.x)


def test_an_equivalent_preconditioner_reaches_the_same_solution():
    """Polak--Ribiere with the Jacobi map agrees with the Fletcher--Reeves path."""
    _, _, H, diagonal, b = system()
    reference = deflated_cg(H, b, diagonal=diagonal, rtol=1e-11)
    flexible = deflated_cg(
        H, b, diagonal=diagonal, rtol=1e-11, preconditioner=lambda r: r / diagonal
    )
    assert flexible.status == "converged"
    assert np.allclose(flexible.x, reference.x, rtol=1e-6, atol=1e-9)


def test_frozen_preconditioner_cuts_iterations_and_keeps_the_solution():
    problem, evaluation, H, diagonal, b = system()
    indices = np.arange(problem.size)
    factory = frozen_preconditioner_factory(problem, evaluation, sweeps=2)
    precondition = factory(indices)
    baseline = deflated_cg(H, b, diagonal=diagonal, rtol=1e-11)
    frozen = deflated_cg(H, b, diagonal=diagonal, rtol=1e-11, preconditioner=precondition)
    assert frozen.status == "converged"
    assert frozen.iterations < baseline.iterations
    assert independent_residual(H, frozen.x, b) <= 1e-11
    assert np.allclose(frozen.x, baseline.x, rtol=1e-6, atol=1e-9)
    assert precondition.applications >= frozen.iterations


def test_the_preconditioner_carries_the_consistent_source_normalisation():
    """The corrected model's frozen operator is the one the diagonal already uses."""
    problem, evaluation, H, diagonal, b = system(consistent=True)
    factory = frozen_preconditioner_factory(problem, evaluation, sweeps=2)
    frozen = deflated_cg(
        H, b, diagonal=diagonal, rtol=1e-11, preconditioner=factory(np.arange(problem.size))
    )
    baseline = deflated_cg(H, b, diagonal=diagonal, rtol=1e-11)
    assert frozen.status == "converged"
    assert frozen.iterations < baseline.iterations
    assert np.allclose(frozen.x, baseline.x, rtol=1e-6, atol=1e-9)


def test_the_factory_restricts_to_the_active_set():
    problem, evaluation, _, _, _ = system()
    inactive = np.arange(problem.size)[::2]
    precondition = frozen_preconditioner_factory(problem, evaluation, sweeps=3)(inactive)
    operator = WeightedReducedOperator(
        evaluation.frozen_operator, problem.weights, problem.alpha
    ).restrict(inactive)
    rng = np.random.default_rng(7)
    r = rng.standard_normal(len(inactive))
    z = precondition(r)
    assert z.shape == r.shape
    assert np.linalg.norm(r - operator @ z) < np.linalg.norm(r)
    assert precondition.storage()["slabs"] == problem.slabs


def test_more_sweeps_approximate_the_frozen_operator_better():
    problem, evaluation, _, _, _ = system()
    indices = np.arange(problem.size)
    operator = WeightedReducedOperator(
        evaluation.frozen_operator, problem.weights, problem.alpha
    ).restrict(indices)
    rng = np.random.default_rng(11)
    r = rng.standard_normal(problem.size)
    errors = []
    for sweeps in (1, 3):
        precondition = frozen_preconditioner_factory(problem, evaluation, sweeps=sweeps)(indices)
        errors.append(np.linalg.norm(r - operator @ precondition(r)))
    assert errors[1] < errors[0]


def test_damping_enters_the_preconditioner():
    problem, evaluation, _, _, _ = system()
    indices = np.arange(problem.size)
    plain = frozen_preconditioner_factory(problem, evaluation, damping=0.0, sweeps=1)(indices)
    damped = frozen_preconditioner_factory(problem, evaluation, damping=0.5, sweeps=1)(indices)
    rng = np.random.default_rng(3)
    r = rng.standard_normal(problem.size)
    assert not np.allclose(plain(r), damped(r))


def test_the_optimizer_reaches_the_same_optimum_with_either_preconditioner():
    problem = small_coupled_problem([0.2, 0.35])
    desired = np.full(problem.size, 0.08)
    lower, upper = np.full(problem.size, -1.0), np.full(problem.size, 1.0)
    results = {}
    for name in ("jacobi", "frozen"):
        solver = StudySolver("jacobi", rank=0, rtol=1e-11, maxiter=2000)
        results[name] = minimize_coupled(
            problem,
            desired,
            lower,
            upper,
            solver,
            tolerance=1e-8,
            max_iterations=6,
            inner_preconditioner=name,
            frozen_sweeps=2,
        )
        solver.close()
    assert results["frozen"].status == results["jacobi"].status
    assert np.allclose(
        results["frozen"].evaluation.state, results["jacobi"].evaluation.state, rtol=1e-6, atol=1e-9
    )


def test_unsupported_kernels_refuse_an_operator_preconditioner():
    problem, evaluation, H, diagonal, b = system()
    indices = np.arange(problem.size)
    B = H.restrict(indices)
    B.diagonal = lambda: diagonal.copy()
    B.preconditioner = frozen_preconditioner_factory(problem, evaluation)(indices)
    solver = StudySolver("jacobi", rank=0, rtol=1e-10, maxiter=10)
    solver.device = "cuda"
    with pytest.raises(ValueError, match="host kernel only"):
        solver.solve(B, b, indices)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"sweeps": 0}, "sweeps"),
        ({"correction": "not callable"}, "callable"),
    ],
)
def test_invalid_preconditioner_settings_are_refused(arguments, message):
    problem, evaluation, _, _, _ = system()
    operator = WeightedReducedOperator(
        evaluation.frozen_operator, problem.weights, problem.alpha
    ).restrict(np.arange(problem.size))
    slabs = np.arange(problem.size) // problem.spatial_size
    with pytest.raises(ValueError, match=message):
        FrozenSweepPreconditioner(operator, slabs, **arguments)


def test_a_non_callable_preconditioner_is_refused_by_the_kernel():
    _, _, H, diagonal, b = system()
    with pytest.raises(ValueError, match="callable"):
        deflated_cg(H, b, diagonal=diagonal, preconditioner=np.ones_like(b))


def test_the_optimizer_refuses_an_unknown_preconditioner():
    problem = small_coupled_problem([0.2])
    solver = StudySolver("jacobi", rank=0, rtol=1e-10, maxiter=10)
    with pytest.raises(ValueError, match="Jacobi diagonal or the velocity-frozen"):
        minimize_coupled(
            problem,
            np.full(problem.size, 0.08),
            np.full(problem.size, -1.0),
            np.full(problem.size, 1.0),
            solver,
            inner_preconditioner="nonsense",
        )
    solver.close()


def test_the_hybrid_wrapper_carries_the_preconditioner():
    """The hybrid solver rebuilds the operator, so it must carry what was attached.

    Without this the preconditioner is silently dropped and the run costs exactly
    what the Jacobi path costs, which is how the first screen attempt failed: its
    first quadratic took the baseline's 721 applications to the iteration.
    """
    from deflation_example.coupled_hybrid_solver import HybridCoupledSolver

    problem, evaluation, H, diagonal, b = system()
    indices = np.arange(problem.size)
    counted = {"calls": 0}
    precondition = frozen_preconditioner_factory(problem, evaluation, sweeps=2)(indices)

    def counting(residual):
        counted["calls"] += 1
        return precondition(residual)

    results = {}
    for name, attach in (("jacobi", False), ("frozen", True)):
        B = H.restrict(indices)
        B.diagonal = lambda: diagonal.copy()
        if attach:
            B.preconditioner = counting
        solver = HybridCoupledSolver("jacobi", rank=0, rtol=1e-11, maxiter=2000)
        results[name], _ = solver.solve(B, b, indices)
        solver.close()
    assert counted["calls"] >= results["frozen"].iterations
    assert results["frozen"].iterations < results["jacobi"].iterations
    assert results["frozen"].status == "converged"
    assert independent_residual(H, results["frozen"].x, b) <= 1e-11


def test_the_preconditioner_and_a_coarse_space_work_together():
    """The combination an ablation arm runs: operator preconditioner plus deflation.

    The coarse correction is applied to the preconditioned residual, so the two
    interact inside `precondition`. Nothing else covers that pairing.
    """
    problem, evaluation, H, diagonal, b = system()
    indices = np.arange(problem.size)
    rng = np.random.default_rng(5)
    basis = rng.standard_normal((problem.size, 4))
    precondition = frozen_preconditioner_factory(problem, evaluation, sweeps=2)(indices)
    plain = deflated_cg(H, b, basis=basis, diagonal=diagonal, rtol=1e-11)
    combined = deflated_cg(
        H, b, basis=basis, diagonal=diagonal, rtol=1e-11, preconditioner=precondition
    )
    assert plain.status == combined.status == "converged"
    assert plain.rank == combined.rank == 4
    assert combined.iterations < plain.iterations
    assert independent_residual(H, combined.x, b) <= 1e-11
    assert np.allclose(combined.x, plain.x, rtol=1e-6, atol=1e-9)


def test_a_coarse_space_with_a_preconditioner_survives_a_residual_restart():
    """Restarts recompute the residual and re-apply the coarse correction."""
    problem, evaluation, H, diagonal, b = system()
    indices = np.arange(problem.size)
    rng = np.random.default_rng(6)
    basis = rng.standard_normal((problem.size, 3))
    precondition = frozen_preconditioner_factory(problem, evaluation, sweeps=1)(indices)
    result = deflated_cg(
        H, b, basis=basis, diagonal=diagonal, rtol=1e-11, refresh=2, preconditioner=precondition
    )
    assert result.status == "converged"
    assert independent_residual(H, result.x, b) <= 1e-11


def test_the_preconditioner_is_built_once_per_active_set():
    """Restricting and factorizing it is not free, so an unchanged set reuses it."""
    from deflation_example.coupled_optimizer import box_quadratic

    problem, evaluation, H, diagonal, b = system()
    builds = {"count": 0}
    factory = frozen_preconditioner_factory(problem, evaluation, sweeps=1)

    def counting(indices):
        builds["count"] += 1
        return factory(indices)

    solver = StudySolver("jacobi", rank=0, rtol=1e-10, maxiter=500)
    result = box_quadratic(
        H,
        -b,
        diagonal,
        np.full(problem.size, -10.0),
        np.full(problem.size, 10.0),
        solver,
        tolerance=1e-8,
        max_steps=6,
        preconditioner_factory=counting,
    )
    solver.close()
    assert result.history
    steps = len(result.history)
    assert builds["count"] <= steps
    assert result.history[-1]["preconditioner_builds"] == builds["count"]
    assert result.history[-1]["preconditioner_seconds"] > 0.0


def test_a_superseded_device_coarse_space_releases_its_arrays():
    """One space per refinement attempt would otherwise stay resident for the solve."""
    from deflation_example.coupled_hybrid_coarse import CudaCoarseSpace

    class Stub:
        def __init__(self):
            self.Z, self.AZ, self.factor = object(), object(), object()
            self.retired = False

        def report(self):
            return {"rank": 3}

        retire = CudaCoarseSpace.retire

    first, second = Stub(), Stub()
    records = [first]
    if records and not isinstance(records[-1], dict):
        records[-1] = records[-1].retire()
    records.append(second)
    assert records[0] == {"rank": 3}
    assert first.Z is None and first.AZ is None and first.factor is None
    assert second.Z is not None
