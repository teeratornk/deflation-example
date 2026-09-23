"""Confirmation cases are chosen without reference to performance."""

import pytest

from deflation_example.coupled_confirmation_targets import choose_targets


def test_selection_is_distinct_and_breaks_ties_by_identifier():
    rows = [
        {"target": 7, "peak_K": 400.0, "weighted_fraction_above_bound": 0.9},
        {"target": 3, "peak_K": 380.0, "weighted_fraction_above_bound": 0.2},
        {"target": 1, "peak_K": 380.0, "weighted_fraction_above_bound": 0.1},
        {"target": 2, "peak_K": 370.0, "weighted_fraction_above_bound": 0.8},
    ]
    assert choose_targets(rows, 7) == [7, 1, 2]
    for row in rows:
        row["solver_seconds"] = 12345.0 if row["target"] == 3 else 0.001
    assert choose_targets(list(reversed(rows)), 7) == [7, 1, 2]


def test_invalid_target_population_refused():
    with pytest.raises(ValueError, match="Three distinct"):
        choose_targets([], 7)
