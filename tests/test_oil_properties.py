import numpy as np
import pytest

from deflation_example.oil_properties import mineral_oil, momentum_reference


def test_published_correlations_at_75_c():
    p = mineral_oil(348.15)
    assert p["density_kg_m3"] == pytest.approx(849.208582207)
    assert p["dynamic_viscosity_Pa_s"] == pytest.approx(0.0030901356479932896)
    assert p["conductivity_W_m_K"] == pytest.approx(0.1271839325)
    assert p["specific_heat_J_kg_K"] == pytest.approx(2078.667430955)
    dt = 1e-3
    derivative = (
        mineral_oil(348.15 + dt)["density_kg_m3"] - mineral_oil(348.15 - dt)["density_kg_m3"]
    ) / (2 * dt)
    assert p["expansion_coefficient_K_inverse"] == pytest.approx(
        -derivative / p["density_kg_m3"], rel=1e-8
    )


@pytest.mark.parametrize("temperature", [273.14, 373.16, np.inf, np.nan, [], 300 + 1j])
def test_temperature_domain_is_enforced(temperature):
    with pytest.raises(ValueError):
        mineral_oil(temperature)


def test_viscosity_sensitivity_retains_density_and_expansion():
    low, base, high = [momentum_reference(t) for t in (357.3, 348.15, 341.3)]
    for key in ("density_kg_m3", "expansion_coefficient_K_inverse", "reference_temperature_K"):
        assert low[key] == base[key] == high[key]
    assert (
        low["dynamic_viscosity_Pa_s"]
        < base["dynamic_viscosity_Pa_s"]
        < high["dynamic_viscosity_Pa_s"]
    )
    for p in (low, base, high):
        assert p["kinematic_viscosity_m2_s"] * p["density_kg_m3"] == pytest.approx(
            p["dynamic_viscosity_Pa_s"]
        )
