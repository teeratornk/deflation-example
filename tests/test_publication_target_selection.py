"""Target selection depends on physical desired fields, never solver outcomes."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from test_coupled_targets import problem

path = (
    Path(__file__).parents[1]
    / "examples/coupled_optimization/fixed_point/select_publication_targets.py"
)
spec = importlib.util.spec_from_file_location("publication_targets", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows(values):
    return [
        {"target": k, "weighted_rms_rise_K": v, "field_sha256": str(k)}
        for k, v in enumerate(values)
    ]


def test_selection_uses_nominal_median_then_maximum_without_duplicates():
    assert module.choose(rows([0, 1, 2, 3, 4, 5, 6, 7])) == [7, 3, 6]
    assert module.choose(rows([0, 1, 2, 3, 4, 5, 6, 7])[::-1]) == [7, 3, 6]


def test_equal_amplitudes_use_index_not_input_order():
    assert module.choose(rows([1] * 8)[::-1]) == [7, 0, 1]


@pytest.mark.parametrize(
    "mutation", ["duplicate_index", "duplicate_field", "negative", "nan", "absent"]
)
def test_invalid_population_rejected(mutation):
    cases = rows(list(range(8)))
    if mutation == "duplicate_index":
        cases[0]["target"] = 7
    elif mutation == "duplicate_field":
        cases[0]["field_sha256"] = cases[1]["field_sha256"]
    elif mutation == "negative":
        cases[0]["weighted_rms_rise_K"] = -1
    elif mutation == "nan":
        cases[0]["weighted_rms_rise_K"] = np.nan
    else:
        cases.pop()
    with pytest.raises(ValueError):
        module.choose(cases)


def physical_problem():
    p = problem()
    p.slabs = len(p.physical_steps)
    p.weights = np.repeat(p.physical_steps, len(p.free)) * np.tile([1, 2, 3], p.slabs)
    p.temperature_scale = 20.0
    return p


def test_amplitude_uses_initial_temperature_and_physical_weights():
    p = physical_problem()
    result = module.target_population(p, 16, 60)
    q = 7
    delta = module.desired_temperature(p, q, 16, 60) - np.tile(p.initial, p.slabs)
    expected = 20 * np.sqrt(np.sum(p.weights * delta**2) / p.weights.sum())
    assert result[q]["weighted_rms_rise_K"] == pytest.approx(expected)
    assert len({r["field_sha256"] for r in result}) == 16
    assert len(set(module.choose(result))) == 3


def test_uniform_weight_rescaling_preserves_population_amplitudes():
    p = physical_problem()
    first = module.target_population(p, 16, 60)
    p.weights *= 3
    second = module.target_population(p, 16, 60)
    np.testing.assert_allclose(
        [r["weighted_rms_rise_K"] for r in first],
        [r["weighted_rms_rise_K"] for r in second],
    )
    assert module.choose(first) == module.choose(second)


def test_nonpositive_tracking_weights_rejected():
    p = physical_problem()
    p.weights[0] = 0
    with pytest.raises(ValueError, match="positive"):
        module.target_population(p, 16, 60)
