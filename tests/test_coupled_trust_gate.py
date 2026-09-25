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
