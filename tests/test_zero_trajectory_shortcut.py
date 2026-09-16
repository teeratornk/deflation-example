"""The cold start's shortcut must test the sensitivity it claims to have.

From a zero control the lumped model's thermal velocity sensitivity really is
exactly zero, so the evaluation keeps no momentum factors and the tangent never
traverses the momentum blocks. A consistent weighting breaks that: the background
source carries the streamline weight too, and that weight moves with the velocity,
so the sensitivity is non-zero from the very first iterate.

Taking the shortcut there left the transpose calling solve on a factor of None,
which is how the corrected model's screen arms died twelve minutes in.
"""

import numpy as np
import pytest

from test_coupled_derivatives import small_coupled_problem


def cold(problem):
    """The optimizer's own starting point: a zero temperature trajectory."""
    return np.zeros(problem.size)


def consistent(steps=None, inlet=0.3):
    problem = small_coupled_problem(steps, uniform_capacity=True, inlet=inlet)
    problem.consistent_stabilization = True
    problem.assembly = problem.assemble(problem.initial_flow.velocity)
    problem.reference_assembly = problem.assemble(
        problem.initial_flow.velocity, consistent=False
    )
    return problem


def test_the_lumped_cold_start_keeps_its_shortcut():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    problem.thermal_boundary = 0.0
    evaluation = problem.evaluate(cold(problem))
    jacobian = evaluation.jacobian
    assert all(factor is None for factor in jacobian.factors)
    assert jacobian.thermal_only
    # A tangent and its transpose both traverse the thermal blocks alone.
    rng = np.random.default_rng(17)
    direction, dual = rng.normal(size=(2, problem.size))
    assert np.isfinite(jacobian @ direction).all()
    assert np.isfinite(jacobian.T @ dual).all()


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_the_consistent_cold_start_keeps_its_momentum_factors(steps):
    problem = consistent(steps)
    problem.thermal_boundary = 0.0
    evaluation = problem.evaluate(cold(problem))
    jacobian = evaluation.jacobian
    # The background source's streamline weight moves with the velocity, so the
    # sensitivity is not zero and every slab needs its factor.
    assert not jacobian.thermal_only
    assert all(factor is not None for factor in jacobian.factors)
    rng = np.random.default_rng(23)
    direction, dual = rng.normal(size=(2, problem.size))
    forward = float(dual @ (jacobian @ direction))
    backward = float(direction @ (jacobian.T @ dual))
    assert abs(forward - backward) <= 1e-9 * max(abs(forward), abs(backward), 1e-30)


def test_a_mixed_evaluation_is_refused_rather_than_left_to_crash():
    """The traversal is all or nothing, so a partial set of factors is an error."""
    problem = consistent([0.2, 0.35])
    evaluation = problem.evaluate(cold(problem))
    jacobian = evaluation.jacobian
    jacobian.factors = (None,) + jacobian.factors[1:]
    with pytest.raises(AttributeError):
        # Without the guard in evaluate this is what the screen saw: the transpose
        # reaching a factor that was never built.
        jacobian.T @ np.ones(problem.size)
