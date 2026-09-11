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
