import numpy as np
import pytest

from deflation_example import coupled_control
from deflation_example.coupled_review_flow import capture_failures, momentum_diagnostics
from test_coupled_derivatives import small_coupled_problem


def test_momentum_diagnostic_checks_quadratic_identity_and_local_mass():
    problem = small_coupled_problem()
    flow = problem.flow
    initial = problem.initial_flow
    acceleration = np.zeros_like(flow.quadrature_points)
    acceleration[:, :, 1] = 0.1 * flow.quadrature_points[:, :, 0]
    result = flow.solve(
        acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        initial=initial,
        pressure_gauge=problem.pressure_gauge,
        max_iterations=1,
    )
    report = momentum_diagnostics(
        flow,
        acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        result,
        pressure_gauge=problem.pressure_gauge,
    )
    assert min(row["jacobian_relative_error"] for row in report["jacobian_differences"]) < 1e-8
    assert report["newton"]["linear_relative_residual"] < 1e-8
    assert report["newton"]["normalized_merit_directional_derivative"] == pytest.approx(
        -1, abs=1e-8
    )
    assert (
        max(row["quadratic_identity_relative_defect"] for row in report["newton"]["trials"]) < 1e-7
    )
    assert "cell_imbalance_over_total_face_flux" in report["mass"]


def test_failed_momentum_capture_preserves_return_and_restores_function(tmp_path):
    problem = small_coupled_problem()
    acceleration = np.zeros_like(problem.flow.quadrature_points)
    acceleration[:, :, 1] = 0.1 * problem.flow.quadrature_points[:, :, 0]
    original = coupled_control.solve_momentum
    settings = dict(
        initial=problem.initial_flow,
        pressure_gauge=problem.pressure_gauge,
        max_iterations=1,
        tolerance=1e-14,
        continuation=False,
    )
    expected = original(
        problem.flow, acceleration, problem.boundary_indices, problem.boundary_values, **settings
    )
    assert expected.status != "converged"
    reports = []
    with capture_failures(tmp_path, reports):
        actual = coupled_control.solve_momentum(
            problem.flow,
            acceleration,
            problem.boundary_indices,
            problem.boundary_values,
            **settings,
        )
    assert coupled_control.solve_momentum is original

    np.testing.assert_array_equal(actual.velocity, expected.velocity)
    np.testing.assert_array_equal(actual.pressure, expected.pressure)
    assert actual.status == expected.status
    assert len(reports) == 1
    assert (tmp_path / reports[0]["file"]).is_file()
    with pytest.raises(RuntimeError), capture_failures(tmp_path, []):
        raise RuntimeError("test restoration")
    assert coupled_control.solve_momentum is original


def test_diagnostic_failure_does_not_replace_momentum_outcome(tmp_path, monkeypatch):
    from deflation_example import coupled_review_flow

    problem = small_coupled_problem()
    acceleration = np.zeros_like(problem.flow.quadrature_points)
    acceleration[:, :, 1] = 0.1 * problem.flow.quadrature_points[:, :, 0]

    def failed_diagnostic(*args, **kwargs):
        raise np.linalg.LinAlgError("diagnostic test")

    monkeypatch.setattr(coupled_review_flow, "momentum_diagnostics", failed_diagnostic)
    reports = []
    with capture_failures(tmp_path, reports):
        result = coupled_control.solve_momentum(
            problem.flow,
            acceleration,
            problem.boundary_indices,
            problem.boundary_values,
            initial=problem.initial_flow,
            pressure_gauge=problem.pressure_gauge,
            max_iterations=1,
            tolerance=1e-14,
        )
    assert result.status != "converged"
    assert reports[0]["status"] == result.status
    assert reports[0]["diagnostic_status"] == "diagnostic_failed"
