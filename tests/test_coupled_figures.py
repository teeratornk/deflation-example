"""Field figures retain physical units and actual stored time levels."""

import numpy as np
import pytest

from deflation_example.coupled_figures import plot_fields
from test_coupled_derivatives import small_coupled_problem


def test_coupled_field_figure_uses_saved_state_source_and_velocity(tmp_path):
    pytest.importorskip("matplotlib")
    problem = small_coupled_problem([0.2, 0.35])
    state = np.linspace(0.04, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    fields = {
        "state": state,
        "desired": state + 0.03,
        "control": evaluation.control,
        "velocity": np.stack([flow.velocity for flow in evaluation.flows]),
    }
    path = tmp_path / "fields.pdf"
    report = plot_fields(problem, fields, 3.0, 1.0, path)
    assert path.is_file()
    assert report["time_indices"] == [0, 1]
    np.testing.assert_allclose(report["times_s"], [0.2, 0.55])
    np.testing.assert_array_equal(fields["state"], state)
    assert report["source_scale_W_m3"] == 3.0
    with pytest.raises(ValueError, match="stored time"):
        plot_fields(problem, fields, 3.0, 1.0, tmp_path / "wrong.pdf", [2])


def test_constraint_panels_use_saved_values_and_actual_time(tmp_path):
    pytest.importorskip("matplotlib")
    problem = small_coupled_problem([0.2, 0.35])
    state = np.full(problem.size, 0.07)
    state[0], state[-1] = 0.04, 0.1
    evaluation = problem.evaluate(state)
    fields = {
        "state": state,
        "desired": state + 0.03,
        "control": evaluation.control,
        "velocity": np.stack([flow.velocity for flow in evaluation.flows]),
    }
    lower = problem.temperature_offset + problem.temperature_scale * 0.04
    upper = problem.temperature_offset + problem.temperature_scale * 0.1
    state[0] = (lower - problem.temperature_offset) / problem.temperature_scale
    state[-1] = (upper - problem.temperature_offset) / problem.temperature_scale
    description = plot_fields(
        problem, fields, 3.0, upper, tmp_path / "constraints.png", lower_K=lower
    )
    assert description["active_constraints"][0]["lower_active_dofs"] == 1
    assert description["active_constraints"][1]["upper_active_dofs"] == 1
    assert description["times_s"] == [0.2, 0.55]
    np.testing.assert_array_equal(fields["state"], state)
    with pytest.raises(ValueError, match="lower temperature"):
        plot_fields(problem, fields, 3.0, upper, tmp_path / "bad.png", lower_K=upper)
