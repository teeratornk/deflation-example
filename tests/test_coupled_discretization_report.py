"""Discretization selection requires all declared accuracy evidence."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example.coupled_discretization_report import local_rows, select
from deflation_example.coupled_discretization_replay import temperature_statistics
from deflation_example.reporting import write_report


def rows():
    return [
        {"transport_form": f, "slabs": n, "status": "all_checks_met", "passed": True}
        for f in ("advective", "skew")
        for n in (16, 32, 64)
    ]


def test_selection_uses_accuracy_and_grid_and_never_favorable_runtime():
    cases = rows()
    cases[0]["seconds"] = 10000
    cases[-1]["seconds"] = 0.01
    result = select(cases)
    assert result["selected"] == {"transport_form": "advective", "slabs": 16}
    cases[0].update(passed=False, status="replay_gate_failed")
    assert select(cases)["selected"] == {"transport_form": "skew", "slabs": 16}


@pytest.mark.parametrize("status", ["running", "missing", "verification_pending", "replay_pending"])
def test_an_early_passing_case_does_not_end_the_declared_population(status):
    cases = rows()
    cases[-1].update(status=status, passed=False)
    assert select(cases)["action"] == "wait_for_declared_outcomes"


def test_all_failed_cases_close_the_bounded_study_without_gpu_timings():
    cases = rows()
    for r in cases:
        r.update(status="trust_radius_exhausted", passed=False)
    assert select(cases)["action"] == "stop_bounded_study"


def test_missing_or_duplicate_configurations_cannot_be_dropped():
    with pytest.raises(ValueError):
        select(rows()[:-1])
    cases = rows()
    cases[-1] = copy.deepcopy(cases[0])
    with pytest.raises(ValueError):
        select(cases)


def test_all_time_comparison_includes_initial_condition_and_mass_weights():
    problem = SimpleNamespace(
        spatial_size=2,
        slabs=4,
        initial=np.array([0.2, 0.4]),
        physical_steps=np.ones(4),
        temperature_scale=20.0,
        temperature_offset=300.0,
        free=np.array([1, 2]),
        assembly=SimpleNamespace(mass=np.array([99.0, 1.0, 3.0])),
        mesh=SimpleNamespace(nodes=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])),
    )
    original = np.array([[0.4, 0.8], [0.8, 1.2]])
    refined = np.array([[0.3, 0.6], [0.4, 0.8], [0.6, 1.0], [0.8, 1.2]])
    refined[0, 1] += 0.004
    before = refined.copy()
    result = temperature_statistics(problem, original, refined, 2, 300.0, 320.0)
    assert result["maximum_trajectory_difference_K"] == pytest.approx(0.08)
    assert result["maximum_endpoint_difference_K"] == pytest.approx(0.0)
    assert result["mass_weighted_space_time_rms_K"] == pytest.approx(np.sqrt(3 * 0.08**2 / 16))
    assert result["difference_peak"]["time_s"] == 1
    assert not result["temperature_sensitivity_met"]
    assert result["temperature_bound_excess"]["upper"]["maximum_K"] == pytest.approx(4.0)
    np.testing.assert_array_equal(before, refined)


def test_local_population_keeps_failed_and_missing_perturbations(tmp_path):
    paths = [tmp_path / str(i) for i in range(27)]
    write_report(
        paths[11] / "record.json",
        {
            "status": "iteration_cap",
            "verified": False,
            "subdivision": 1,
            "perturbation_K": 1e-6,
            "steps": [{"status": "iteration_cap", "verified": False}],
        },
    )
    results = local_rows(paths)
    assert len(results) == 27
    assert results[11]["trajectory"] == "retained"
    assert results[11]["status"] == "iteration_cap"
    assert results[12]["status"] == "missing"
    assert not any(r["verified"] for r in results)


def test_initial_momentum_diagnostic_keeps_original_residual_and_fields():
    from test_coupled_derivatives import small_coupled_problem
    from deflation_example.coupled_initial_flow import diagnose

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    initial = problem.initial_flow.velocity.copy()
    result, report = diagnose(problem)
    assert result.status == "converged"
    assert report["float64_momentum_residual"] == pytest.approx(
        report["original_equations"]["momentum_relative_residual"],
        abs=1e-15,
    )
    assert report["extended_accumulation_momentum_residual"] < problem.flow_tolerance
    np.testing.assert_array_equal(problem.initial_flow.velocity, initial)
