import copy

import pytest

from deflation_example.coupled_qp_execution_report import summarize
from test_coupled_qp_cuda_report import gpu_record


def matched_record(device, rank):
    record = gpu_record(rank)
    record.update(device=device, solver_construction_seconds=2)
    return record


def test_failed_runs_have_no_completed_solve_speedup():
    rows = summarize([matched_record("cpu", 0), matched_record("hybrid", 16)])["rows"]
    assert rows[0]["construction_inclusive_seconds"] == 8
    assert rows[1]["construction_inclusive_seconds"] == 24
    # Synthetic base fixtures describe nonconverged quadratics.
    assert all(r["speedup"] is None for r in rows)
    missing = summarize([matched_record("cpu", 0), None])["rows"]
    assert missing[-1]["status"] == "missing"
    assert missing[-1]["speedup"] is None


@pytest.mark.parametrize(
    "key,value", [("quadratic_tolerance", 0.1), ("reference_initial", "optimizer")]
)
def test_execution_mismatch_rejected(key, value):
    other = matched_record("cuda", 16)
    other[key] = value
    with pytest.raises(ValueError, match="Match"):
        summarize([matched_record("cpu", 0), other])


def test_setup_cannot_be_omitted_and_inputs_are_unchanged():
    record = matched_record("cpu", 0)
    before = copy.deepcopy(record)
    summarize([record])
    assert record == before
    del record["solver_construction_seconds"]
    with pytest.raises(ValueError, match="construction timers"):
        summarize([record])
