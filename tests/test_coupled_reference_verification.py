from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example.coupled_nominal_krylov import configured_krylov_reference
from test_coupled_derivatives import small_coupled_problem


def test_failed_reference_equations_are_recorded_before_termination(monkeypatch):
    import deflation_example.coupled_nominal_krylov as module

    monkeypatch.setattr(
        module,
        "temperature_bounds",
        lambda *a, **k: {"optimization_lower_K": -1, "optimization_upper_K": 1},
    )
    checks = [{"failure": "retained"}]
    problem = SimpleNamespace(
        size=2,
        temperature_offset=0,
        temperature_scale=1,
        evaluate=lambda state, initial: state,
        verify=lambda evaluation: checks,
    )
    monkeypatch.setattr(module, "equation_acceptance", lambda *a: False)
    saved = []
    with pytest.raises(ValueError, match="fail independent verification"):
        configured_krylov_reference(
            problem,
            {
                "device": "cpu",
                "inner_preconditioner": "frozen",
                "rank": 1,
                "queries": [{"upper_K": 1}],
            },
            verification_callback=saved.append,
        )
    assert saved == [checks]


def test_verification_reports_absolute_defects_and_normalization():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    evaluation = problem.evaluate(np.linspace(0.04, 0.1, problem.size))
    for row in problem.verify(evaluation):
        assert row["thermal_rhs_norm"] > 0
        assert row["thermal_relative_residual"] == pytest.approx(
            row["thermal_residual_norm"] / row["thermal_rhs_norm"]
        )
        assert row["mass_relative_imbalance"] == pytest.approx(
            row["mass_net_flux"] / row["mass_flux_normalization"]
        )
        assert row["energy"]["relative_defect"] == pytest.approx(
            abs(row["energy"]["defect"]) / row["energy"]["normalization"]
        )


@pytest.mark.parametrize("as_guess", [False, True])
def test_reference_evaluates_declared_initial_state_without_replacement(monkeypatch, as_guess):
    import deflation_example.coupled_nominal_krylov as module

    state = np.array([0.1, 0.2])
    guess = SimpleNamespace(state=state)
    seen = []

    def evaluate(value, initial):
        np.testing.assert_array_equal(value, state)
        assert value is not state
        assert initial is (guess if as_guess else None)
        seen.append(True)
        return value

    monkeypatch.setattr(
        module,
        "temperature_bounds",
        lambda *a, **k: {
            "optimization_lower_K": -1,
            "optimization_upper_K": 1,
        },
    )
    monkeypatch.setattr(module, "equation_acceptance", lambda *a: False)
    problem = SimpleNamespace(
        size=2,
        temperature_offset=0,
        temperature_scale=1,
        evaluate=evaluate,
        verify=lambda value: [{"retained": True}],
    )
    options = {"initial_guess": guess} if as_guess else {"initial_state": state}
    with pytest.raises(ValueError, match="fail independent verification"):
        configured_krylov_reference(
            problem,
            {
                "device": "cpu",
                "inner_preconditioner": "frozen",
                "rank": 1,
                "queries": [{"upper_K": 1}],
            },
            **options,
        )
    assert seen == [True]
