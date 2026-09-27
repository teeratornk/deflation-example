"""The repair campaign retains failures and never treats them as speedups."""

from copy import deepcopy

import numpy as np
import pytest

from deflation_example.coupled_backtrack_gate import decision
from test_coupled_trust_gate import records


def fixtures():
    adaptive = records()[1]
    repairs = [deepcopy(adaptive), deepcopy(adaptive)]
    for i, r in enumerate(repairs):
        r["configuration"].update(
            trial_policy="backtrack", capture_trials=True, flow_continuation=bool(i)
        )
    return adaptive, repairs


def converge(record, seconds=100):
    record.update(status="complete", all_problems_verified=True, cumulative_attempt_seconds=seconds)
    record["cases"] = [
        {
            "status": "converged",
            "verified": True,
            "objective": 1.0,
            "history": [],
            "kkt": {
                k: 0.0
                for k in [
                    "primal_absolute",
                    "stationarity",
                    "dual_feasibility",
                    "lower_complementarity",
                    "upper_complementarity",
                ]
            },
            "equations": [
                {
                    "momentum_relative_residual": 1e-13,
                    "continuity_relative_residual": 1e-13,
                    "thermal_relative_residual": 1e-13,
                    "mass_relative_imbalance": 1e-9,
                    "energy": {"relative_defect": 1e-9},
                }
            ],
            "adjoint": {
                "maximum_momentum_adjoint_relative_residual": 1e-13,
                "maximum_source_adjoint_relative_residual": 1e-13,
                "gradient_weight_normalized_difference": 1e-13,
            },
        }
    ]


def test_original_success_skips_repairs_and_running_records_wait():
    a, _ = fixtures()
    a["status"] = "running"
    assert decision(a)["action"] == "wait_for_adaptive"
    assert decision(a, adaptive_terminal=True)["action"] == "run_two_repair_diagnostics"
    converge(a)
    assert decision(a)["selected"] == "original_adaptive"


def test_two_unsuccessful_repairs_stop_expansion():
    a, r = fixtures()
    assert decision(a, r)["action"] == "stop_for_numerical_diagnosis"
    r[0]["status"] = "running"
    assert decision(a, r)["action"] == "wait_for_repairs"
    assert decision(a, r, repairs_terminal=(True, True))["action"] == "stop_for_numerical_diagnosis"
    assert r[0]["status"] == "running"


def test_partial_scheduler_termination_does_not_skip_a_live_repair():
    a, r = fixtures()
    for record in r:
        record["status"] = "running"
    assert decision(a, r, repairs_terminal=(True, False))["action"] == "wait_for_repairs"
    with pytest.raises(ValueError, match="termination flag"):
        decision(a, r, repairs_terminal=(True,))


def test_common_undeclared_change_does_not_pass_by_matching_both_repairs():
    a, r = fixtures()
    for record in r:
        record["configuration"]["transport_form"] = "different"
    with pytest.raises(ValueError, match="beyond the declared"):
        decision(a, r)


def test_one_success_selects_verified_policy_only():
    a, r = fixtures()
    converge(r[1])
    result = decision(a, r)
    assert result["action"] == "prepare_reference_screen"
    assert result["flow_continuation"] is True


def test_two_successes_require_agreement_then_use_total_cost():
    a, r = fixtures()
    converge(r[0], 100)
    converge(r[1], 90)
    with pytest.raises(ValueError, match="saved temperature"):
        decision(a, r)
    assert decision(a, r, [np.zeros(3), np.ones(3)])["action"] == "stop_solution_disagreement"
    assert decision(a, r, [np.zeros(3), np.zeros(3)])["flow_continuation"] is True


@pytest.mark.parametrize(
    "key,value",
    [("alpha", 1e-10), ("inner_tolerance", 1e-8), ("slabs", 32), ("target_startup_s", 0)],
)
def test_changed_physics_or_accuracy_is_refused(key, value):
    a, r = fixtures()
    r[1]["configuration"][key] = value
    with pytest.raises(ValueError):
        decision(a, r)


def test_mismatched_repair_sources_are_refused():
    a, r = fixtures()
    r[1]["environment"]["source_sha256"] = "different"
    with pytest.raises(ValueError, match="beyond flow"):
        decision(a, r)


@pytest.mark.parametrize("objective", [None, float("nan"), float("inf"), True])
def test_invalid_objective_cannot_select_a_policy(objective):
    a, r = fixtures()
    converge(r[0])
    r[0]["cases"][0]["objective"] = objective
    assert decision(a, r)["action"] == "stop_for_numerical_diagnosis"
