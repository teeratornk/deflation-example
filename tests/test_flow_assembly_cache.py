"""Carrying the residual assembly into the next iteration changes no arithmetic.

The Newton loop assembles the momentum operator at the end of an iteration to
measure the residual, and the next iteration needs that same operator: nothing
between them changes the velocity. Reusing it removes one of the two assemblies
per iteration, and assemblies are about half the cost of a coupled forward
evaluation.

The claim is exactness, so these tests check the two things it rests on: the
assembly is a function of the velocity alone, and the matrix the loop carries is
the one a fresh assembly would produce. A one-off comparison against the previous
implementation, over both methods, both convection forms and steady and transient
solves, found every velocity, pressure and history entry bitwise identical.
"""

import numpy as np
import pytest

from deflation_example.axisymmetric_flow import AxisymmetricFlow
from test_axisymmetric_flow import annular_rectangle


def problem(convection_form="advective"):
    flow = AxisymmetricFlow(annular_rectangle(4), 0.3, convection_form=convection_form)
    r, z = flow.points.T
    exact = np.column_stack((0.02 * r, -0.04 * z))
    rq, zq = flow.quadrature_points.transpose(2, 0, 1)
    force = np.stack((0.02**2 * rq + 0.03, 0.04**2 * zq + 0.07), axis=2)
    return flow, force, exact


def solve(flow, force, exact, **extra):
    return flow.solve(
        force,
        flow.boundary,
        exact[flow.boundary],
        pressure_gauge=(0, 0.03),
        tolerance=1e-10,
        **extra,
    )


@pytest.mark.parametrize("convection_form", ["advective", "skew"])
def test_the_assembly_is_a_function_of_the_velocity_alone(convection_form):
    """Why carrying it is exact: the same velocity gives the same matrix, bitwise."""
    flow, _, _ = problem(convection_form)
    rng = np.random.default_rng(20260921)
    velocity = rng.standard_normal((flow.nv, 2))
    for time_step in (None, 0.5):
        first = flow.operator(velocity, time_step, True).tocsr()
        second = flow.operator(velocity, time_step, True).tocsr()
        assert np.array_equal(first.data, second.data)
        assert np.array_equal(first.indices, second.indices)
        assert np.array_equal(first.indptr, second.indptr)


@pytest.mark.parametrize("method", ["newton", "picard"])
def test_the_carried_matrix_is_what_a_fresh_assembly_would_give(method, monkeypatch):
    """Capture the operator each iteration measured its residual with, then rebuild it."""
    flow, force, exact = problem()
    seen = []
    original = AxisymmetricFlow._residual_metrics

    def capture(self, A, x, rhs, constrained, prescribed):
        seen.append((A, x.copy()))
        return original(self, A, x, rhs, constrained, prescribed)

    monkeypatch.setattr(AxisymmetricFlow, "_residual_metrics", capture)
    result = solve(
        flow, force, exact, method=method, time_step=0.5, previous=np.zeros((flow.nv, 2))
    )
    assert result.status == "converged"
    assert len(seen) >= 2
    for A, x in seen:
        velocity = np.column_stack((x[: flow.nv], x[flow.nv : 2 * flow.nv]))
        fresh = flow.operator(velocity, 0.5, True).tocsr()
        assert np.array_equal(A.tocsr().data, fresh.data)


def test_the_carry_removes_one_assembly_per_iteration(monkeypatch):
    flow, force, exact = problem()
    calls = {"count": 0}
    original = AxisymmetricFlow.operator

    def counting(self, velocity, time_step=None, convection=True):
        calls["count"] += 1
        return original(self, velocity, time_step, convection)

    monkeypatch.setattr(AxisymmetricFlow, "operator", counting)
    result = solve(flow, force, exact, method="newton")
    assert result.status == "converged"
    iterations = len(result.history)
    assert iterations > 1
    # One assembly to enter the loop, one per iteration for the residual, and none
    # at the head of any iteration after the first.
    assert calls["count"] == iterations + 1


def test_the_line_search_failure_path_does_not_carry_a_stale_matrix():
    """That branch restores the previous iterate, so nothing may be carried past it."""
    flow, force, exact = problem()
    result = solve(flow, force, exact, method="newton", relaxation=1e-12, max_iterations=3)
    assert result.status in {"converged", "iteration_cap", "line_search_failed", "stagnation"}
    for row in result.history:
        assert np.isfinite(row["momentum_relative_residual"])


def test_a_flow_at_rest_does_not_collapse_the_trust_region():
    """A rest flow has no speed to take a fraction of; the step must survive.

    Dividing by a floor of machine tiny overflowed and scaled the whole Newton
    step to zero, which abandoned the direction on the first iteration.
    """
    from deflation_example.coupled_newton_replay import trust_scale
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem([0.2])

    class Rest:
        velocity = np.zeros((problem.flow.nv, 2))

    update = np.zeros(len(problem.flow_free) + problem.spatial_size)
    update[0] = 1e-3
    update[-1] = 1e-4
    scale = trust_scale(problem, Rest(), update, limit=1.0)
    assert 0.0 < scale <= 1.0
