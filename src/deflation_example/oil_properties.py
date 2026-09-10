"""Published mineral-oil correlations with an explicit temperature domain.

Y. Li et al., High Voltage 9 (2024), 230--240, equations (5)--(8).
https://doi.org/10.1049/hve2.12345
Expansion is derived from density: beta = -rho'(T)/rho(T).
These parameters define an assessment fluid; the original oil is unidentified.
"""

import numpy as np

from .validation import real_array

SOURCE = {
    "doi": "10.1049/hve2.12345",
    "url": "https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/hve2.12345",
    "title": "Thermofluidic investigations of oil natural transformer: Closed-loop modelling and experimental validation",
    "authors": "Yuan Li et al.",
    "journal": "High Voltage",
    "year": 2024,
    "volume": 9,
    "pages": "230-240",
    "equations": [5, 6, 7, 8],
    "temperature_domain_K": [273.15, 373.15],
    "scope": "Published assessment parameters; original optimization material data remain separate.",
}


def mineral_oil(temperature_K):
    """Return SI properties on 273.15--373.15 K, without extrapolation or clipping."""
    T = real_array(temperature_K, "Oil temperature")
    if not T.size or not np.isfinite(T).all() or np.any((T < 273.15) | (T > 373.15)):
        raise ValueError("The published oil correlations cover 273.15--373.15 K")
    rho = 1093 - 0.826 * T + 3.612e-4 * T * T
    mu = 7.863e-5 * np.exp(632 / (T - 176))
    return {
        "density_kg_m3": rho,
        "dynamic_viscosity_Pa_s": mu,
        "kinematic_viscosity_m2_s": mu / rho,
        "expansion_coefficient_K_inverse": (0.826 - 2 * 3.612e-4 * T) / rho,
        "specific_heat_J_kg_K": 455.9 + 5.191 * T - 1.522e-3 * T * T,
        "conductivity_W_m_K": 0.1538 - 7.645e-5 * T,
    }


def momentum_reference(viscosity_temperature_K=348.15, reference_temperature_K=348.15):
    """Freeze density and expansion at 75 C; vary viscosity alone by default.

    The viscosity temperatures are predeclared parameter choices, independent
    of computed temperatures. Thermal coefficients remain those of optimization.
    """
    if np.ndim(viscosity_temperature_K) or np.ndim(reference_temperature_K):
        raise ValueError("Frozen reference temperatures must be scalars")
    p = mineral_oil(reference_temperature_K)
    mu = mineral_oil(viscosity_temperature_K)["dynamic_viscosity_Pa_s"]
    return {
        "source": SOURCE.copy(),
        "reference_temperature_K": float(reference_temperature_K),
        "viscosity_temperature_K": float(viscosity_temperature_K),
        "density_kg_m3": float(p["density_kg_m3"]),
        "expansion_coefficient_K_inverse": float(p["expansion_coefficient_K_inverse"]),
        "dynamic_viscosity_Pa_s": float(mu),
        "kinematic_viscosity_m2_s": float(mu / p["density_kg_m3"]),
        "viscosity_policy": "constant within each forward solve; viscosity-only sensitivity",
    }
