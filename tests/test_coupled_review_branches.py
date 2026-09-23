import numpy as np

from deflation_example.coupled_review_branches import branch_distances
from test_coupled_derivatives import small_coupled_problem


def test_branch_distances_separate_stationary_and_moving_fields():
    problem = small_coupled_problem(consistent=True, uniform_capacity=True)
    zero = np.zeros_like(problem.initial_flow.velocity)
    moving = problem.initial_flow.velocity
    rows = branch_distances(problem, [zero, moving])
    assert rows[0]["limited_cells"] == 0
    assert all(np.isinf(r["row_limit_relative_distance"]) for r in rows[0]["nearest_switches"])
    assert all(0 <= r["row_limit_relative_distance"] <= 1 for r in rows[1]["nearest_switches"])
    assert all(len(r["nodes"]) == 3 for r in rows[1]["nearest_switches"])
