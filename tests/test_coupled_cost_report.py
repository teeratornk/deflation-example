"""Exclusive cost accounting and all-outcome repeated-sequence summaries."""

import copy

import pytest

from deflation_example.coupled_cost_report import confirmation, profile_summary
from test_coupled_small_report import paired


def population():
    output = []
    for repetition in range(5):
        baseline, reference = paired()
        reference["configuration"]["rank"] = 8
        recycling = copy.deepcopy(reference)
        recycling["configuration"].update(method="recycling", study_variant="recycling")
        for r in (baseline, reference, recycling):
            r["configuration"].update(repetition=repetition, alpha=1e-14, slabs=16)
            r["configuration"]["queries"] = [{"target": t, "upper_K": 357.3} for t in (7, 8, 9)]
            original = r["cases"][0]
            r["cases"] = []
            for i, target in enumerate((7, 8, 9)):
                case = copy.deepcopy(original)
                case.update(position=i, target=target)
                case["equations"] = [{**case["equations"][0], "slab": s} for s in range(16)]
                r["cases"].append(case)
            r["prior_attempt_seconds"] = 0
            r["cumulative_attempt_seconds"] = r["attempt_seconds"]
            output.append(r)
    return output


def test_five_complete_repetitions_and_all_methods_are_required():
    result = confirmation(population())
    assert result["all_sequences_verified"]
    assert result["all_comparisons_matched"]
    assert result["methods"]["reference"]["verified_sequences"] == 5
    assert result["median_time_ratios"]["baseline_over_reference"] == pytest.approx(2)


@pytest.mark.parametrize(
    "defect", ["missing", "failed", "unmatched", "duplicate", "capture", "targets", "resumed"]
)
def test_incomplete_or_changed_comparisons_have_no_headline_ratio(defect):
    records = population()
    if defect == "missing":
        records[3] = None
    elif defect == "failed":
        records[3]["status"] = "budget_exhausted"
    elif defect == "unmatched":
        records[3]["environment"]["cpu_model"] = "different"
    elif defect == "duplicate":
        records[3] = records[0]
    elif defect == "capture":
        records[3]["configuration"]["capture_linear_systems"] = True
    elif defect == "targets":
        records[3]["configuration"]["queries"] = [{"target": 7}]
    else:
        records[3]["resumed"] = True
    if defect in {"duplicate", "capture", "targets", "resumed"}:
        with pytest.raises(ValueError):
            confirmation(records)
    else:
        assert confirmation(records)["median_time_ratios"] is None


def test_positive_outcome_is_not_required_and_iteration_counts_do_not_define_speedup():
    records = population()
    for r in records:
        if r["configuration"]["rank"] == 8:
            r["attempt_seconds"] *= 4
            r["cumulative_attempt_seconds"] *= 4
            r["components_seconds"] = {k: 4 * v for k, v in r["components_seconds"].items()}
    result = confirmation(records)
    assert result["median_time_ratios"]["baseline_over_reference"] == pytest.approx(0.5)


def test_profile_uses_exclusive_times_and_strips_machine_paths():
    stats = {
        ("/private/machine/solvers.py", 1, "solve"): (1, 1, 2.0, 50.0, {}),
        ("~", 0, "<method 'solve' of 'SuperLU' objects>"): (2, 2, 5.0, 5.0, {}),
    }
    result = profile_summary(stats)
    assert result["exclusive_total_seconds"] == 7
    assert sum(result["exclusive_categories_seconds"].values()) == 7
    assert result["largest_nested_calls"][0]["cumulative_seconds"] == 50
    assert all("/private/" not in row["file"] for row in result["largest_nested_calls"])


def test_all_slots_must_remain_visible():
    with pytest.raises(ValueError, match="slots"):
        confirmation(population()[:-1])


def test_matched_pairs_from_different_sources_cannot_form_one_population():
    records = population()
    for record in records[-3:]:
        record["environment"]["source_sha256"] = {"module": "later_source"}
    result = confirmation(records)
    assert result["all_sequences_verified"]
    assert not result["all_comparisons_matched"]
    assert result["median_time_ratios"] is None
    assert result["methods"]["reference"]["median_complete_seconds"] is None
