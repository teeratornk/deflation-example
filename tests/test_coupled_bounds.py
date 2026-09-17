import numpy as np
import pytest

from deflation_example.coupled_bounds import resolution_margin, temperature_bounds


def test_margin_preserves_physical_bounds_and_tightens_both_design_bounds():
    cfg = {"lower_K": 337.3, "upper_K": 357.3}
    zero = temperature_bounds(cfg)
    assert zero["optimization_lower_K"] == cfg["lower_K"]
    assert zero["optimization_upper_K"] == cfg["upper_K"]
    cfg["temperature_margin_K"] = resolution_margin(0.015, 0.01)
    result = temperature_bounds(cfg)
    assert result["temperature_margin_K"] == pytest.approx(0.05)
    assert result["physical_upper_K"] == cfg["upper_K"]
    assert result["physical_lower_K"] == cfg["lower_K"]
    assert result["optimization_upper_K"] == pytest.approx(357.25)
    assert result["optimization_lower_K"] == pytest.approx(337.35)
    assert temperature_bounds(cfg, upper_K=353.3)["optimization_upper_K"] == pytest.approx(353.25)


def test_margin_is_not_clipped_when_resolution_is_insufficient():
    assert resolution_margin(0, 0) == 0.01
    with pytest.raises(ValueError, match="refine"):
        resolution_margin(0.04, 0.02)


@pytest.mark.parametrize("value", [-0.001, 0.1001, np.nan, np.inf, True])
def test_invalid_design_margin_rejected(value):
    with pytest.raises(ValueError):
        temperature_bounds({"lower_K": 337.3, "upper_K": 357.3, "temperature_margin_K": value})


def test_empty_interval_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        temperature_bounds({"lower_K": 337.3, "upper_K": 337.4, "temperature_margin_K": 0.1})
