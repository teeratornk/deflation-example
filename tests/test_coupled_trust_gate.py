"""A diagnostic gate must retain failures and reject inconsistent evidence."""

import copy

import numpy as np
import pytest

from deflation_example.coupled_trust_gate import decision, inner_accuracy


def records():
    cfg = dict(
        flow_continuation=False,
        method="jacobi",
        inner_preconditioner="frozen",
        rank=0,
        queries=[{"target": 7, "upper_K": 357.3}],
        alpha=1e-11,
        slabs=64,
        horizon_s=600.0,
        target_startup_s=60.0,
        lower_K=337.3,
        inner_tolerance=1e-10,
        qp_tolerance=1e-10,
        nonlinear_tolerance=1e-8,
        equation_acceptance_tolerance=1e-12,
        conservation_tolerance=1e-6,
    )
    return [
        dict(
            schema="coupled-trust-development-v1",
            configuration={**cfg, "trust_accuracy": accuracy},
            environment={"source_sha256": "frozen-test"},
            policy={"identifier": "test"},
            baseline_sha256="same",
            status="budget_exhausted",
            all_problems_verified=False,
            cases=[],
            cumulative_attempt_seconds=100.0,
        )
        for accuracy in ["strict", "adaptive"]
    ]


def test_incomplete_or_capped_runs_do_not_trigger_fallback():
    result = decision(records(), [None, None])
    assert result["action"] == "stop_for_diagnosis"


def test_only_matched_flow_failures_trigger_fallback():
    rows = records()
    case = {
        "status": "trust_radius_exhausted",
        "history": [{"attempts": [{"trials": [{"status": "flow_stagnation"}]}]}],
    }
    for r in rows:
        r["cases"] = [copy.deepcopy(case)]
    assert decision(rows, [None, None])["action"] == "evaluate_continuation_fallback"
    rows[0]["cases"][0]["status"] = "budget_exhausted"
    assert decision(rows, [None, None])["action"] == "stop_for_diagnosis"


def test_configuration_mismatch_refuses_selection():
    rows = records()
    rows[1]["configuration"]["alpha"] = 1e-10
    with pytest.raises(ValueError):
        decision(rows, [None, None])


def test_inner_original_residuals_and_final_strict_phase():
    step = {"candidate_retained": True, "linear_status": "converged", "linear_residual": 1e-5}
    attempt = {"linear_tolerance": 1e-4, "strict_accuracy": False, "qp_history": [step]}
    case = {"history": [{"attempts": [attempt]}]}
    cfg = {"inner_tolerance": 1e-10}
    assert not inner_accuracy(case, cfg)
    strict = {
        "linear_tolerance": 1e-10,
        "strict_accuracy": True,
        "qp_history": [{**step, "linear_residual": 1e-11}],
    }
    case["history"][0]["attempts"].append(strict)
    assert inner_accuracy(case, cfg)
    strict["qp_history"][0]["linear_residual"] = np.nan
    assert not inner_accuracy(case, cfg)


def test_verified_selection_requires_solution_agreement_and_actual_checks():
    from deflation_example.coupled_trust_gate import verified

    rows = records()
    equation = dict(
        momentum_relative_residual=1e-13,
        continuity_relative_residual=1e-13,
        thermal_relative_residual=1e-13,
        mass_relative_imbalance=1e-9,
        energy={"relative_defect": 1e-9},
    )
    case = dict(
        status="converged",
        verified=True,
        objective=1.0,
        history=[],
        kkt={
            key: 0.0
            for key in (
                "primal_absolute",
                "stationarity",
                "dual_feasibility",
                "lower_complementarity",
                "upper_complementarity",
            )
        },
        equations=[equation],
        adjoint=dict(
            maximum_momentum_adjoint_relative_residual=1e-13,
            maximum_source_adjoint_relative_residual=1e-13,
            gradient_weight_normalized_difference=1e-13,
        ),
    )
    for r in rows:
        r.update(status="complete", all_problems_verified=True, cases=[copy.deepcopy(case)])
    rows[1]["cumulative_attempt_seconds"] = 90.0
    result = decision(rows, [np.ones(3), np.ones(3)])
    assert result["action"] == "evaluate_four_complete_sequences"
    assert result["accuracy"] == "adaptive"
    assert decision(rows, [np.ones(3), np.zeros(3)])["action"] == "stop_solution_disagreement"
    rows[0]["cases"][0]["equations"][0]["thermal_relative_residual"] = 1e-9
    assert not verified(rows[0])
    assert verified(rows[1])
