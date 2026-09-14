"""Saved-state checks distinguish equation defects from propagated replay error."""

import numpy as np
import pytest

from deflation_example.coupled_replay_check import check_local_steps, check_saved_trajectory
from test_coupled_derivatives import small_coupled_problem


def saved_case():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    state = np.linspace(0.045, 0.09, problem.size)
    evaluation = problem.evaluate(state)
    return problem, [
        state,
        evaluation.control,
        np.stack([f.velocity for f in evaluation.flows]),
        np.stack([f.pressure for f in evaluation.flows]),
    ]


def test_all_saved_equations_and_exact_previous_recovery():
    problem, fields = saved_case()
    originals = [v.copy() for v in fields]
    rows = []
    result = check_saved_trajectory(problem, *fields, callback=rows.append)
    assert rows == result["rows"]
    assert len(rows) == 2
    np.testing.assert_allclose([r["time_s"] for r in rows], [0.2, 0.55])
    for row in rows:
        assert row["thermal_relative_residual"] < 1e-12
        assert row["momentum_relative_residual"] < 1e-10
        assert row["continuity_relative_residual"] < 1e-10
        assert row["exact_previous_frozen_velocity_error_K"] < 1e-12
        assert "observed_coupled_replay_error_K" not in row
    for original, field in zip(originals, fields, strict=True):
        np.testing.assert_array_equal(original, field)


def test_changed_control_is_visible_in_equation_residual_and_reconstruction():
    problem, fields = saved_case()
    fields[1] = fields[1] + 1
    rows = check_saved_trajectory(problem, *fields)["rows"]
    assert min(r["thermal_relative_residual"] for r in rows) > 1e-3
    assert min(r["exact_previous_frozen_velocity_error_K"] for r in rows) > 1e-3


def test_replay_comparison_distinguishes_previous_error_from_current_error():
    problem, fields = saved_case()
    replay = fields[0].reshape(problem.slabs, -1).copy()
    replay[0] += 0.01
    rows = check_saved_trajectory(problem, *fields, replay=replay)["rows"]
    assert rows[0]["replayed_previous_frozen_velocity_error_K"] < 1e-12
    assert rows[0]["observed_coupled_replay_error_K"] == pytest.approx(0.02)
    assert rows[1]["replayed_previous_frozen_velocity_error_K"] > 0
    assert rows[1]["observed_coupled_replay_error_K"] == 0


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("invalid", ["shape", "nan"])
def test_malformed_saved_fields_are_rejected(index, invalid):
    problem, fields = saved_case()
    if invalid == "shape":
        fields[index] = fields[index][:-1]
    else:
        fields[index].flat[0] = np.nan
    with pytest.raises(ValueError):
        check_saved_trajectory(problem, *fields)


@pytest.mark.parametrize("replay", [np.zeros((1, 1)), np.full((2, 9), np.nan)])
def test_incomplete_or_nonfinite_replay_is_rejected(replay):
    problem, fields = saved_case()
    with pytest.raises(ValueError, match="Replay"):
        check_saved_trajectory(problem, *fields, replay=replay)


def test_local_restarts_use_saved_previous_data_and_recover_both_steps():
    problem, fields = saved_case()
    originals = [v.copy() for v in fields]
    rows = check_local_steps(problem, *fields, [0, 1])
    assert len(rows) == 4
    assert {r["initial_guess"] for r in rows} == {"saved_current", "saved_previous"}
    for row in rows:
        assert row["status"] == "converged"
        assert row["temperature_difference_K"] < 1e-8
        assert row["velocity_difference_l2_m_s"] < 1e-8
    for original, field in zip(originals, fields, strict=True):
        np.testing.assert_array_equal(original, field)


@pytest.mark.parametrize("indices", [[-1], [2], [0, 0]])
def test_invalid_local_slab_selection_is_rejected(indices):
    problem, fields = saved_case()
    with pytest.raises(ValueError):
        check_local_steps(problem, *fields, indices)
