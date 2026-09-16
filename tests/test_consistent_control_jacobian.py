"""The control is recovered through a factored action, so the transpose must match.

With the consistent stabilisation the source is weighted by the streamline test
function, so the control is no longer recovered by dividing by the lumped mass.
The thermal and velocity blocks are left unnormalised and the action is applied
inside the Jacobian, which is exact only if the transpose applies the transposed
factor first. These tests pin that, and pin that the tangent really is the
derivative of the recovered control.
"""

import numpy as np
import pytest

from test_coupled_derivatives import small_coupled_problem


def consistent_problem(steps):
    problem = small_coupled_problem(steps, uniform_capacity=True)
    problem.consistent_stabilization = True
    problem.assembly = problem.assemble(problem.initial_flow.velocity)
    return problem


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_the_consistent_control_jacobian_is_an_exact_transpose(steps):
    problem = consistent_problem(steps)
    assert problem.assembly.consistent
    state = np.linspace(0.03, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    J = evaluation.jacobian
    assert J.source_factors is not None and len(J.source_factors) == problem.slabs

    rng = np.random.default_rng(4113)
    direction, dual = rng.normal(size=(2, problem.size))
    direction /= np.linalg.norm(direction)
    dual /= np.linalg.norm(dual)
    forward = float(dual @ (J @ direction))
    backward = float(direction @ (J.T @ dual))
    assert abs(forward - backward) <= 1e-10 * max(abs(forward), abs(backward), 1e-30)


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_the_consistent_tangent_is_the_derivative_of_the_recovered_control(steps):
    problem = consistent_problem(steps)
    state = np.linspace(0.03, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    J = evaluation.jacobian
    rng = np.random.default_rng(991)
    direction = rng.normal(size=problem.size)
    direction /= np.linalg.norm(direction)
    tangent = J @ direction
    remainders = []
    for step in (0.01, 0.005, 0.0025):
        trial = problem.evaluate(state + step * direction)
        remainders.append(
            float(np.linalg.norm(trial.control - evaluation.control - step * tangent))
        )
    orders = np.log2(np.array(remainders[:-1]) / np.array(remainders[1:]))
    assert min(orders) > 1.9, f"orders {orders} from remainders {remainders}"


def test_the_lumped_default_is_untouched():
    """Without the consistent option nothing is factored and nothing changes."""
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    assert not problem.assembly.consistent
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    assert evaluation.jacobian.source_factors is None
