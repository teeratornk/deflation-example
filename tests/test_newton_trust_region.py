"""A restricted search must reach the same step, not a different one.

The trust region changes where each line search starts and nothing else, so on a
problem whose Newton steps are already small it must reproduce the unrestricted
solve, and on either model it must land on the same state to solver tolerance.
"""

import numpy as np
import pytest

from deflation_example.coupled_newton_replay import newton_step
from test_coupled_derivatives import small_coupled_problem


def solved(consistent, trust):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    if consistent:
        problem.consistent_stabilization = True
        problem.assembly = problem.assemble(problem.initial_flow.velocity)
    evaluation = problem.evaluate(np.linspace(0.04, 0.08, problem.size))
    source = np.zeros(len(problem.mesh.nodes))
    source[problem.free] = evaluation.control.reshape(problem.slabs, -1)[0]
    previous = problem.full_temperature(problem.initial)
    return newton_step(
        problem,
        source,
        previous,
        problem.initial_flow,
        0,
        tolerance=1e-12,
        max_iterations=40,
        line_search="fixed_scaled",
        trust_region=trust,
    )


@pytest.mark.parametrize("consistent", [False, True])
def test_a_restricted_search_reaches_the_same_step(consistent):
    free = solved(consistent, None)
    restricted = solved(consistent, 0.25)
    assert free.status == "converged" and restricted.status == "converged"
    difference = float(np.max(np.abs(free.state - restricted.state)))
    assert difference <= 1e-9, f"the two searches landed {difference} apart"
    assert all("trust_region_scale" in row for row in restricted.history[1:])
    assert all(row["trust_region_scale"] <= 1.0 for row in restricted.history[1:])


def test_the_unrestricted_search_records_no_restriction():
    free = solved(True, None)
    assert all(row["trust_region_scale"] == 1.0 for row in free.history[1:])
