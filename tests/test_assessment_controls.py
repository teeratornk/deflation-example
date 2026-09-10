"""Deterministic source selection and immutable temporal transfer."""

import numpy as np
import pytest

from deflation_example.assessment_controls import select_queries, source_at_time, array_digest


def test_selection_is_distinct_and_ties_use_query_order():
    assert select_queries(np.zeros(16), np.zeros(16)) == {"nominal": 8, "active": 0, "demanding": 1}
    active, control = np.zeros(16), np.zeros(16)
    active[5] = active[8] = 0.2
    control[5] = 100
    control[4] = control[11] = 3
    assert select_queries(active, control) == {"nominal": 8, "active": 5, "demanding": 4}


@pytest.mark.parametrize("value", [-1, 1.1, np.nan, np.inf])
def test_selection_rejects_invalid_active_fractions(value):
    active = np.zeros(16)
    active[2] = value
    with pytest.raises(ValueError):
        select_queries(active, np.zeros(16))


def test_selection_rejects_incomplete_population():
    with pytest.raises(ValueError):
        select_queries(np.zeros(15), np.zeros(16))


def test_temporal_source_is_right_slab_and_unchanged():
    source = np.array([[1.0, -2.0], [3.0, -4.0]])
    before = array_digest(source)
    for time, index in [(0, 0), (1, 0), (1 + 1e-9, 1), (2, 1)]:
        result = source_at_time(source, [1, 2], time)
        np.testing.assert_array_equal(result, source[index])
        result[:] = 0
    assert array_digest(source) == before


@pytest.mark.parametrize("time", [-1, 2.01, np.nan, np.inf])
def test_source_rejects_extrapolation(time):
    with pytest.raises(ValueError):
        source_at_time(np.ones((2, 3)), [1, 2], time)
