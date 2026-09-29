"""Incomplete, unmatched and duplicate baselines cannot establish a speedup."""

import pytest

from deflation_example.coupled_small_report import summarize
from test_coupled_projected_report import record


def paired():
    baseline, reference = record(), record()
    for r, rank in ((baseline, 0), (reference, 16)):
        r["configuration"].update(alpha=1e-11, rank=rank, device="cpu")
        r["environment"] = {"source_sha256": {"module": "same"}}
        r["baseline_sha256"] = "same-baseline"
        r["initial_state"] = {"snapshot": "same-temperature"}
    baseline.update(attempt_seconds=24.0, cumulative_attempt_seconds=29.0)
    baseline["components_seconds"]["optimization"] += 12
    return baseline, reference


def test_complete_cost_includes_construction_and_all_intervals():
    baseline, reference = paired()
    result = summarize([baseline, reference, None])["runs"]
    assert result[1]["speedup"] == pytest.approx(29 / 17)
    assert result[2]["status"] == "missing" and result[2]["speedup"] is None


@pytest.mark.parametrize(
    "defect",
    [
        "failed",
        "source",
        "initial",
        "duplicate",
        "targets",
        "device",
        "repetition",
        "capture",
        "hardware",
        "boundary",
    ],
)
def test_unmatched_or_failed_comparison_has_no_ratio(defect):
    baseline, reference = paired()
    records = [baseline, reference]
    if defect == "failed":
        baseline["status"] = "budget_exhausted"
    elif defect == "source":
        reference["environment"]["source_sha256"] = {"module": "changed"}
    elif defect == "initial":
        reference["initial_state"] = {"snapshot": "different"}
    elif defect == "duplicate":
        records.append(baseline)
    elif defect == "device":
        reference["configuration"]["device"] = "cuda"
    elif defect == "repetition":
        reference["configuration"]["repetition"] = 1
    elif defect == "capture":
        reference["configuration"]["capture_linear_systems"] = True
    elif defect == "hardware":
        reference["environment"]["cpu_model"] = "different"
    elif defect == "boundary":
        reference["timing_boundary"] = "excludes construction"
    else:
        reference["configuration"]["queries"] = [{"target": 8}]
    assert summarize(records)["runs"][1]["speedup"] is None


def test_variants_remain_named_and_repetitions_pair_separately():
    import copy

    baseline, reference = paired()
    reference["configuration"].update(method="recycling", study_variant="recycling")
    second_base, second_ref = copy.deepcopy(baseline), copy.deepcopy(reference)
    for r in (second_base, second_ref):
        r["configuration"]["repetition"] = 1
    rows = summarize([baseline, reference, second_base, second_ref])["runs"]
    assert rows[1]["method"] == rows[1]["variant"] == "recycling"
    assert rows[1]["speedup"] == rows[3]["speedup"] == pytest.approx(29 / 17)
