"""Auxiliary continuation must end with the original fixed-source equation."""

import numpy as np
import pytest

from deflation_example.coupled_time_continuation import continued_step
from deflation_example.coupled_newton_replay import step_equations, criteria_met
from deflation_example.coupled_resolution import forward_model
from test_coupled_newton_replay import data


def test_auxiliary_stages_recover_original_step_without_changing_physical_history():
    problem, expected, _, source = data(consistent=True, streamline_rule="smooth_p8", inlet=0.2)
    previous = problem.full_temperature(problem.initial)
    before_source, before_state = source.copy(), previous.copy()
    before_steps, before_scaled = problem.physical_steps.copy(), problem.steps.copy()
    snapshots = []
    result = continued_step(
        problem,
        source,
        previous,
        problem.initial_flow,
        0,
        callback=lambda rows: snapshots.append(rows),
    )
    assert result.status == "converged", result.history
    assert snapshots[-1][-1]["auxiliary_step_fraction"] == 1
    np.testing.assert_allclose(result.state[problem.free], expected.reshape(2, -1)[0], atol=1e-10)
    _, checks = step_equations(
        problem,
        forward_model(problem),
        result.state,
        result.flow,
        source,
        previous,
        problem.initial_flow,
        0,
    )
    assert criteria_met(checks, 1e-12)
    assert all(result.history[-1][key] == value for key, value in checks.items())
    for actual, original in (
        (source, before_source),
        (previous, before_state),
        (problem.physical_steps, before_steps),
        (problem.steps, before_scaled),
    ):
        np.testing.assert_array_equal(actual, original)


def test_capped_continuation_never_reports_auxiliary_success_as_physical_success():
    problem, _, _, source = data(consistent=True, streamline_rule="smooth_p8", inlet=0.2)
    previous = problem.full_temperature(problem.initial)
    result = continued_step(problem, source, previous, problem.initial_flow, 0, stage_cap=1)
    assert result.history[0]["stages"][0]["auxiliary_status"] == "converged"
    assert result.status == "continuation_stage_cap"
    _, checks = step_equations(
        problem,
        forward_model(problem),
        result.state,
        result.flow,
        source,
        previous,
        problem.initial_flow,
        0,
    )
    assert not criteria_met(checks, 1e-12)
    assert all(result.history[-1][key] == value for key, value in checks.items())


def test_already_converged_candidate_skips_all_auxiliary_solves():
    problem, expected, evaluation, source = data(
        consistent=True, streamline_rule="smooth_p8", inlet=0.2
    )
    result = continued_step(
        problem,
        source,
        problem.full_temperature(problem.initial),
        problem.initial_flow,
        0,
        initial_state=problem.full_temperature(expected.reshape(2, -1)[0]),
        initial_flow=evaluation.flows[0],
        tolerance=1e-10,
    )
    assert result.status == "converged"
    assert result.history[0]["stages"] == []


def test_failed_stage_reduces_increment_and_keeps_matching_original_residual():
    problem, _, _, source = data(consistent=True, streamline_rule="smooth_p8", inlet=0.2)
    previous = problem.full_temperature(problem.initial)
    result = continued_step(problem, source, previous, problem.initial_flow, 0, max_iterations=0)
    assert result.status == "continuation_minimum_increment"
    stages = result.history[0]["stages"]
    assert len(stages) > 1
    assert stages[1]["auxiliary_step_fraction"] == stages[0]["auxiliary_step_fraction"] / 2
    np.testing.assert_array_equal(result.state, previous)


@pytest.mark.parametrize(
    "options",
    [
        {"tolerance": 1e-7},
        {"tolerance": 0},
        {"tolerance": np.nan},
        {"minimum_increment": 1},
        {"minimum_increment": 0},
        {"stage_cap": 0},
        {"max_iterations": -1},
    ],
)
def test_invalid_continuation_settings_rejected(options):
    problem, _, _, source = data()
    with pytest.raises(ValueError):
        continued_step(
            problem,
            source,
            problem.full_temperature(problem.initial),
            problem.initial_flow,
            0,
            **options,
        )
