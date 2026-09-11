"""Independent equation criteria must gate all coupled verification records."""

import copy
import numpy as np
import pytest

from deflation_example.coupled_optimize import equations_verified


def row():
    return {
        "momentum_relative_residual": 1e-10,
        "continuity_relative_residual": 1e-12,
        "thermal_relative_residual": 1e-12,
        "mass_relative_imbalance": 1e-10,
        "energy": {"relative_defect": 1e-10},
    }


@pytest.mark.parametrize(
    "field",
    [
        "momentum_relative_residual",
        "continuity_relative_residual",
        "thermal_relative_residual",
        "mass_relative_imbalance",
    ],
)
@pytest.mark.parametrize("value", [1e-4, np.nan, np.inf])
def test_failed_equation_or_mass_cannot_pass_derivative_verification(field, value):
    valid = row()
    assert equations_verified([valid])
    failed = copy.deepcopy(valid)
    failed[field] = value
    assert not equations_verified([valid, failed])


def test_empty_and_failed_energy_checks_do_not_pass():
    assert not equations_verified([])
    failed = row()
    failed["energy"]["relative_defect"] = np.nan
    assert not equations_verified([failed])
