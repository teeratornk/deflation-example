"""Physical-unit and objective checks use the unchanged saved source field."""

from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example.coupled_confirmation_fields import field_metrics


def test_field_objective_and_bounds_use_consistent_weights_and_units():
    problem = SimpleNamespace(
        size=2,
        weights=np.array([1, 2]),
        alpha=0.5,
        objective_scale=3,
        temperature_offset=300,
        temperature_scale=20,
    )
    fields = {
        "state": np.array([1.0, 2.0]),
        "desired": np.array([0.5, 1.5]),
        "control": np.array([2.0, -2.0]),
    }
    result = field_metrics(problem, fields, fields["desired"], 305, 335, 10.125)
    assert result["objective_consistent"]
    assert result["objective_from_fields"] == pytest.approx(10.125)
    assert result["maximum_upper_violation_K"] == 5
    assert result["maximum_lower_violation_K"] == 0
    assert fields["control"][1] == -2.0


def test_changed_target_is_rejected_and_changed_objective_is_reported():
    problem = SimpleNamespace(
        size=2,
        weights=np.ones(2),
        alpha=1,
        objective_scale=1,
        temperature_offset=300,
        temperature_scale=20,
    )
    fields = {"state": np.zeros(2), "desired": np.ones(2), "control": np.zeros(2)}
    with pytest.raises(ValueError, match="desired"):
        field_metrics(problem, fields, np.zeros(2), 300, 340, 1)
    assert not field_metrics(problem, fields, fields["desired"], 300, 340, 2)[
        "objective_consistent"
    ]
