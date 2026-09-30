"""The paired summary keeps failures and refuses mismatched comparisons."""

import copy

import pytest

from deflation_example.coupled_branch_restart_report import summarize


@pytest.fixture
def records():
    base = dict(
        schema="coupled-radius-restart-v1",
        status="budget_exhausted",
        verified=False,
        gpu_gate_passed=False,
        source_configuration={"nonlinear_tolerance": 1e-8},
        record_sha256="source",
        field_sha256="field",
        position=1,
        target=8,
        restart={"budget_seconds": 7200},
        environment={"source_sha256": {"solver": "same"}},
        initial_branch={
            "selected": "retained",
            "assessment_sha256": "assessment",
            "trajectory_sha256": {},
        },
        history=[{"objective": 1.0, "kkt": {"stationarity": 0.9}}],
        objective=0.99,
        kkt={"stationarity": 0.8},
        seconds=7230,
        optimizer_seconds=7200,
    )
    alternative = copy.deepcopy(base)
    alternative["initial_branch"]["selected"] = "alternate_seed"
    alternative["status"] = "trust_radius_exhausted"
    return [base, alternative]


def test_both_failures_remain_in_summary(records):
    result = summarize(records)
    assert not result["alternative_gpu_gate_passed"]
    assert [r["status"] for r in result["branches"]] == [
        "budget_exhausted",
        "trust_radius_exhausted",
    ]
    assert len(result["branches"][0]["retained_states"]) == 2


@pytest.mark.parametrize(
    "change", ["population", "duplicate", "running", "settings", "source", "gate"]
)
def test_mismatches_and_unverified_gate_fail(records, change):
    if change == "population":
        records.pop()
    elif change == "duplicate":
        records[1]["initial_branch"]["selected"] = "retained"
    elif change == "running":
        records[1]["status"] = "running"
    elif change == "settings":
        records[1]["restart"]["budget_seconds"] = 1800
    elif change == "source":
        records[1]["environment"]["source_sha256"]["solver"] = "changed"
    else:
        records[1]["gpu_gate_passed"] = True
    with pytest.raises(ValueError):
        summarize(records)


def test_diagnostic_error_is_visible_without_optimization_metrics(records):
    records[1].update(status="diagnostic_error", error="branch changed")
    for key in ("history", "objective", "kkt"):
        records[1].pop(key)
    summary = summarize(records)
    assert summary["branches"][1]["retained_states"] == []
    assert summary["branches"][1]["error"] == "branch changed"
