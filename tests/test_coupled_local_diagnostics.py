import numpy as np

from deflation_example.coupled_local_diagnostics import derivative_diagnostic, displaced
from deflation_example.coupled_newton_replay import newton_step
from test_coupled_derivatives import small_coupled_problem


def test_displacement_keeps_boundary_data_and_pressure_gauge():
    p = small_coupled_problem([0.2, 0.35], uniform_capacity=True, consistent=True)
    state = p.full_temperature(p.initial)
    direction = np.ones(len(p.flow_free) + p.spatial_size)
    candidate, flow = displaced(p, state, p.initial_flow, direction, 0.001)
    np.testing.assert_array_equal(candidate[p.mesh.dirichlet], state[p.mesh.dirichlet])
    np.testing.assert_array_equal(
        flow.velocity[p.boundary_indices], p.initial_flow.velocity[p.boundary_indices]
    )
    assert flow.pressure[0] == p.initial_flow.pressure[0]


def test_diagnostic_checks_original_jacobian_at_both_iterates():
    p = small_coupled_problem([0.2, 0.35], uniform_capacity=True, consistent=True)
    state = p.full_temperature(p.initial)
    source = np.zeros(len(p.mesh.nodes))
    source[p.free] = 0.05
    result = newton_step(
        p, source, state, p.initial_flow, 0, max_iterations=30, line_search="fixed_scaled"
    )
    assert result.status == "converged"
    for current, flow in ((state, p.initial_flow), (result.state, result.flow)):
        report = derivative_diagnostic(p, source, state, p.initial_flow, 0, current, flow)
        assert len(report["finite_differences"]) == 8
        random = [r for r in report["finite_differences"] if r["direction"] == "scaled_random"]
        assert min(r["scaled_relative_error"] for r in random) < 1e-6
        assert all(r["plus_branch_changes"] == r["minus_branch_changes"] == 0 for r in random)
