"""A quadratic scalar fold checks the seed sign and parameter units."""

import numpy as np
import pytest

from deflation_example.coupled_flow_branch import curvature_projection


@pytest.mark.parametrize("root", [0.2, -0.2, 1e-4])
def test_exact_scalar_quadratic_recovers_other_root(root):
    # F(x, mu) = x*x - root*root - mu; J=2*root and f=1.
    w = np.array([1 / (2 * root)])
    inverse_q = w**2 / (2 * root)
    result = curvature_projection(w, inverse_q, 1)
    assert result["predicted_turning_step_K"] == pytest.approx(-(root**2))
    parameter = result["nonzero_zero_load_seed_parameter_K"]
    assert root + parameter * w[0] == pytest.approx(-root)
    assert result["projection_relative_remainder"] < 1e-15


def test_pressure_scaling_cannot_change_velocity_projection():
    first = curvature_projection(np.array([1.0, 3e8]), np.array([2.0, -7e-9]), 1)
    second = curvature_projection(np.array([1.0, 0]), np.array([2.0, 0]), 1)
    assert first == second


@pytest.mark.parametrize(
    "w,q,n",
    [
        ([0.0], [1.0], 1),
        ([1.0], [0.0], 1),
        ([1.0], [float("nan")], 1),
        ([1.0], [2.0], 0),
        ([1.0], [2.0], True),
    ],
)
def test_invalid_curvature_projections_fail_explicitly(w, q, n):
    with pytest.raises(ValueError):
        curvature_projection(np.array(w), np.array(q), n)


def test_fresh_root_checks_detect_a_perturbed_momentum_field():
    from deflation_example.axisymmetric_flow import FlowResult
    from deflation_example.coupled_flow_branch import root_checks
    from deflation_example.coupled_flow_response import tangent_response, temperature_direction
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem([0.2, 0.35])
    state = np.linspace(0.04, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    retained, previous = evaluation.flows[1], evaluation.flows[0].velocity
    direction = temperature_direction(problem, state, np.zeros_like(state), 1)
    tangent, _ = tangent_response(problem, retained, previous, state, 1, direction)
    verified = root_checks(problem, retained, previous, state, 1, direction, tangent)
    assert max(verified["independent_residuals"].values()) < problem.flow_tolerance
    assert verified["newton_velocity_correction_norm_m_s"] < 1e-10
    assert verified["tangent_velocity_cosine_with_retained"] == pytest.approx(1)
    velocity = retained.velocity.copy()
    interior = np.setdiff1d(np.arange(problem.flow.nv), problem.boundary_indices)
    velocity[interior[0], 0] += 1e-4
    changed = FlowResult(velocity, retained.pressure.copy(), "diagnostic", [])
    failed = root_checks(problem, changed, previous, state, 1, direction, tangent)
    assert max(failed["independent_residuals"].values()) > problem.flow_tolerance
    assert failed["newton_velocity_correction_norm_m_s"] > 1e-6


def test_trajectory_evaluation_preserves_temperatures_and_never_claims_optimization(tmp_path):
    from deflation_example.coupled_flow_branch import trajectory_check
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    state = np.linspace(0.04, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    velocities = np.stack([f.velocity for f in evaluation.flows])
    pressures = np.stack([f.pressure for f in evaluation.flows])
    cfg = dict(
        nonlinear_tolerance=1e-8, equation_acceptance_tolerance=1e-9, conservation_tolerance=1e-6
    )
    report = {}
    trajectory_check(
        problem,
        state,
        velocities,
        pressures,
        evaluation.flows[1],
        1,
        np.full(problem.size, 0.12),
        (-0.5, 0.5),
        cfg,
        tmp_path,
        report,
    )
    assert len(report["trajectory_checks"]) == 2
    for row in report["trajectory_checks"]:
        assert row["status"] == "evaluated"
        assert row["equations_verified"] and row["adjoint_verified"]
        assert not row["optimization_performed"]
        assert not row["stationarity_criterion_met"]
    assert report["trajectory_checks"][1]["temperature_change_max"] == 0
    assert abs(report["trajectory_checks"][1]["normalized_objective_change"]) < 1e-12
