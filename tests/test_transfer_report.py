"""Replay summaries preserve acceptance, timing and spectral definitions."""

from copy import deepcopy

import pytest

from deflation_example.benchmark_transfer_report import METHODS, summarize


def replay():
    sample = {
        "status": "converged",
        "original_residual": 1e-12,
        "transfer_seconds": 1,
        "kernel_seconds": 2,
        "total_seconds": 3,
        "deployed_rank": 2,
        "fallback_reason": None,
        "iterations": 4,
    }
    group = {
        "repetitions": [sample],
        "accepted_repetitions": 1,
        "all_repetitions_accepted": True,
        "accepted_median_seconds": 3,
        "diagnostics": {
            "status": "verified",
            "coarse_removed_energy_fraction": 0.8,
            "remaining_energy_fraction": 0.2,
            "original_preconditioned_condition": 10,
            "effective_condition": 2,
        },
    }
    return {
        "success": True,
        "complete": True,
        "specification": {
            "device": "cpu",
            "repetitions_per_kernel": 1,
            "source_protocol": {
                "controls": {
                    "problem": "steady",
                    "n": 3,
                    "rank": 2,
                    "rtol": 1e-10,
                }
            },
        },
        "rows": [
            {
                "query": 0,
                "one_step_identity_error": 0,
                "newly_active": 1,
                "newly_inactive": 2,
                "methods": {method: deepcopy(group) for method in METHODS},
            }
        ],
    }


def test_transfer_summary_checks_formulas_and_incomplete_replays():
    result = replay()
    rows = summarize(result)
    assert all(row["sum_kernel_median_seconds"] == 3 for row in rows)
    assert rows[0]["median_effective_condition"] == 2
    result["complete"] = False
    assert all(row["sum_kernel_median_seconds"] is None for row in summarize(result))
    result["rows"][0]["methods"]["full_reference"]["diagnostics"]["effective_condition"] = None
    row = summarize(result)[0]
    assert row["full_space_corrections"] == 1
    assert row["median_effective_condition"] is None


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("accepted_repetitions", 0, "count"),
        ("all_repetitions_accepted", False, "flag"),
        ("accepted_median_seconds", 4, "median"),
    ],
)
def test_transfer_summary_rejects_inconsistent_acceptance(field, value, match):
    result = replay()
    result["rows"][0]["methods"]["full_reference"][field] = value
    with pytest.raises(ValueError, match=match):
        summarize(result)


def test_failed_transfer_has_no_accepted_timing():
    result = replay()
    group = result["rows"][0]["methods"]["full_reference"]
    group.update(accepted_repetitions=0, all_repetitions_accepted=False)
    group["repetitions"][0]["status"] = "maxiter"
    with pytest.raises(ValueError, match="failed kernel"):
        summarize(result)
    group["accepted_median_seconds"] = None
    row = summarize(result)[0]
    assert row["statuses"] == {"maxiter": 1}
    assert row["sum_kernel_median_seconds"] is None


def test_transfer_summary_rejects_double_counted_time_and_energy():
    result = replay()
    group = result["rows"][0]["methods"]["full_reference"]
    group["repetitions"][0]["total_seconds"] = 4
    with pytest.raises(ValueError, match="timing"):
        summarize(result)
    group["repetitions"][0]["total_seconds"] = 3
    group["diagnostics"]["remaining_energy_fraction"] = 0.3
    with pytest.raises(ValueError, match="energy"):
        summarize(result)
