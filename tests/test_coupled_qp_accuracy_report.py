import copy

import numpy as np
import pytest

from deflation_example.coupled_qp_accuracy_report import TOLERANCES, summarize


def population():
    records, fields = {}, {}
    for t in TOLERANCES:
        records[t] = {
            "source_record_sha256": "r",
            "trace_sha256": "t",
            "system_sha256": "s",
            "quadratic_sha256": "q",
            "environment": {"git_head": "source", "cpu_model": "cpu"},
            "linear_tolerance": t,
            "status": "complete",
            "diagnostic": {"linear_verified": True, "original_residual": t * 0.1},
        }
        fields[t] = {
            "initial": np.zeros(2),
            "rhs": np.ones(2),
            "partition": np.zeros(2),
            "candidate": np.array([1 + t, 0.0]),
            "next_partition": np.array([0, 1]),
        }
    return records, fields


def test_complete_population_compares_candidate_and_partition_to_strict():
    records, fields = population()
    fields[1e-4]["next_partition"][0] = -1
    report = summarize(records, fields)
    assert report["all_linear_replays_verified"]
    assert report["rows"][0]["next_partition_difference_from_strict"] == 1
    assert report["rows"][-1]["candidate_infinity_difference_from_strict"] == 0


def test_missing_and_failed_replays_remain_visible():
    records, fields = population()
    del records[1e-6]
    records[1e-10]["status"] = "diagnostic_error"
    report = summarize(records, fields)
    assert not report["all_linear_replays_verified"]
    assert not report["strict_comparison_available"]
    assert report["rows"][1]["status"] == "missing"
    assert report["rows"][-1]["status"] == "diagnostic_error"


@pytest.mark.parametrize("what", ["input", "source", "accuracy", "initial", "rhs", "partition"])
def test_incompatible_records_are_rejected(what):
    records, fields = copy.deepcopy(population())
    if what == "input":
        records[1e-6]["trace_sha256"] = "changed"
    elif what == "source":
        records[1e-6]["environment"]["git_head"] = "changed"
    elif what == "accuracy":
        records[1e-6]["diagnostic"]["original_residual"] = 1e-3
    else:
        fields[1e-6][what][0] = 10
    with pytest.raises(ValueError):
        summarize(records, fields)
