"""Independent equation criteria must gate all coupled verification records."""

import copy
import numpy as np
import pytest

from deflation_example.coupled_optimize import equations_verified, equation_acceptance


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
@pytest.mark.parametrize("value", [1e-4, np.nan, np.inf, -1.0])
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


def test_declared_equation_acceptance_is_independent_of_legacy_default():
    checks = row()
    assert equation_acceptance([checks], {})
    config = {"equation_acceptance_tolerance": 1e-12, "conservation_tolerance": 1e-6}
    assert not equation_acceptance([checks], config)
    checks["momentum_relative_residual"] = 1e-12
    assert equation_acceptance([checks], config)
    checks["energy"]["relative_defect"] = 1.01e-6
    assert not equation_acceptance([checks], config)


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_invalid_acceptance_tolerances_are_rejected(value):
    with pytest.raises(ValueError):
        equations_verified([row()], equation_tolerance=value)
    with pytest.raises(ValueError):
        equations_verified([row()], conservation_tolerance=value)


def test_corrected_configuration_keeps_the_declared_formulation_and_pilot_population():
    from hydra import compose, initialize_config_module

    with initialize_config_module(config_module="deflation_example.conf", version_base=None):
        config = compose(config_name="coupled_corrected")
    assert config.consistent_stabilization
    assert config.reference_stabilization == "matched"
    assert config.transport_form == "advective"
    assert config.slabs == 64
    assert config.target_startup_s == 60
    assert config.horizon_s == 600
    assert config.queries == [{"target": 7, "upper_K": 357.3}]
    assert config.temperature_margin_K == 0
    assert config.flow_tolerance == config.equation_acceptance_tolerance == 1e-12
    assert config.conservation_tolerance == 1e-6
    assert config.local_mass_diagnostics
    assert config.nonlinear_tolerance == 1e-8
    assert config.inner_tolerance == 1e-10
    assert config.rank == config.recycle_window == 200
    assert config.secant_memory == 10
    assert config.stage.positions == [0]
