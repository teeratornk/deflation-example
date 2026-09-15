"""Reported comparisons preserve accuracy, timing boundaries and unsuccessful runs."""

from copy import deepcopy

import pytest

from deflation_example.coupled_report import summarize, validate_record
from test_coupled_protocol import row as equation_row


def record(method, repetition, seconds):
    return {
        "schema": "coupled-complete-sequence-v1",
        "status": "complete",
        "configuration": {
            "method": method,
            "repetition": repetition,
            "queries": [{"target": 7, "upper_K": 357.3}],
            "nonlinear_tolerance": 1e-8,
        },
        "numerical_policy": {"identifier": "test"},
        "baseline_sha256": "baseline",
        "environment": {
            "source_sha256": {"solver.py": "source"},
            "cpu_model": "test",
            "numpy": "test",
            "scipy": "test",
        },
        "device": {},
        "timing_boundary": "complete test sequence",
        "sequence_seconds": seconds,
        "components_seconds": {"setup": 0.2, "queries": seconds - 0.3, "cleanup": 0.1},
        "process_preparation_seconds": 0.5,
        "calibration_seconds": 1.0,
        "preparation_inclusive_seconds": seconds + 1.5,
        "cases": [
            {
                "position": 0,
                "target": 7,
                "upper_K": 357.3,
                "status": "converged",
                "verified": True,
                "kkt": {
                    "primal_absolute": 0.0,
                    "stationarity": 1e-9,
                    "dual_feasibility": 0.0,
                    "lower_complementarity": 0.0,
                    "upper_complementarity": 0.0,
                },
                "equations": [equation_row()],
                "adjoint": {"maximum_momentum_adjoint_relative_residual": 1e-12},
            }
        ],
        "verified_problems": 1,
        "all_problems_verified": True,
        "memory": {"complete": True, "peak_host_rss_bytes": 1000, "peak_gpu_process_bytes": 0},
    }


def population():
    return [
        record(method, repeat, base + repeat)
        for method, base in (("jacobi", 3), ("reference", 1), ("recycling", 2))
        for repeat in range(3)
    ]


def test_only_complete_matched_populations_produce_a_speedup():
    data = population()
    result = summarize(data, repetitions=3)
    assert result["all_declared_sequences_verified"]
    assert result["fastest_tested_alternative_over_reference"] == pytest.approx(1.5)
    assert result["methods"][1]["verified_sequence_range_seconds"] == [1, 3]
    assert result["methods"][1]["preparation_inclusive_median_seconds"] == 3.5
    missing = summarize(data[:-1], repetitions=3)
    assert not missing["all_declared_sequences_verified"]
    assert missing["fastest_tested_alternative_over_reference"] is None
    failed = deepcopy(data)
    failed[0]["cases"][0].update(status="nonlinear_iteration_cap", verified=False)
    failed[0].update(verified_problems=0, all_problems_verified=False)
    report = summarize(failed, repetitions=3)
    assert report["fastest_tested_alternative_over_reference"] is None
    assert report["methods"][0]["sequence_seconds_all_outcomes"] == [3, 4, 5]
    assert report["methods"][0]["outcomes"][0]["target_statuses"] == ["nonlinear_iteration_cap"]
    assert report["methods"][0]["verified_sequences"] == 2


def test_mismatched_sources_and_duplicate_repetitions_are_rejected():
    data = population()
    with pytest.raises(ValueError, match="Repetition"):
        summarize(data + [deepcopy(data[0])], repetitions=3)
    data[0]["environment"]["source_sha256"]["solver.py"] = "changed"
    with pytest.raises(ValueError, match="matched sources"):
        summarize(data, repetitions=3)


def test_false_verification_and_inconsistent_timing_are_rejected():
    original = record("reference", 0, 2)
    bad = deepcopy(original)
    bad["cases"][0]["kkt"]["stationarity"] = 1e-5
    with pytest.raises(ValueError, match="accuracy checks"):
        validate_record(bad)
    bad = deepcopy(original)
    bad["components_seconds"]["setup"] += 1
    with pytest.raises(ValueError, match="sum"):
        validate_record(bad)
    bad = deepcopy(original)
    bad["cases"] = []
    with pytest.raises(ValueError, match="Every declared target"):
        validate_record(bad)


def test_original_and_ramped_targets_cannot_be_pooled():
    data = population()
    for row in data:
        row["configuration"]["target_startup_s"] = 60.0
    assert summarize(data, repetitions=3)["all_declared_sequences_verified"]
    data[0]["configuration"]["target_startup_s"] = 0.0
    with pytest.raises(ValueError, match="matched sources"):
        summarize(data, repetitions=3)


def test_predeclared_ranks_allow_only_the_two_explicit_budget_differences():
    data = population()
    policy = {
        "jacobi": {"rank": 0, "recycle_window": 100},
        "reference": {"rank": 100, "recycle_window": 100},
        "recycling": {"rank": 200, "recycle_window": 200},
    }
    for row in data:
        row["configuration"].update(policy[row["configuration"]["method"]])
    with pytest.raises(ValueError, match="matched sources"):
        summarize(data, repetitions=3)
    result = summarize(data, repetitions=3, rank_policy=policy)
    assert result["fastest_tested_alternative_over_reference"] == pytest.approx(1.5)
    assert result["matched_protocol"]["method_specific_rank_policy"] == policy
    assert result["rank_comparison"] == "predeclared method-specific ranks"
    data[0]["configuration"]["rank"] = 1
    with pytest.raises(ValueError, match="predeclared"):
        summarize(data, repetitions=3, rank_policy=policy)
    data[0]["configuration"]["rank"] = 0
    data[0]["configuration"]["inner_tolerance"] = 1e-8
    with pytest.raises(ValueError, match="matched sources"):
        summarize(data, repetitions=3, rank_policy=policy)


@pytest.mark.parametrize(
    "failure", ["missing_method", "extra_setting", "nonzero_jacobi", "zero_reference"]
)
def test_rank_policy_cannot_silently_change_the_comparison(failure):
    policy = {
        method: {"rank": 0 if method == "jacobi" else 100, "recycle_window": 100}
        for method in ("jacobi", "reference", "recycling")
    }
    if failure == "missing_method":
        del policy["jacobi"]
    elif failure == "extra_setting":
        policy["reference"]["inner_tolerance"] = 1e-8
    elif failure == "nonzero_jacobi":
        policy["jacobi"]["rank"] = 100
    else:
        policy["reference"]["rank"] = 0
    with pytest.raises(ValueError):
        summarize(population(), repetitions=3, rank_policy=policy)


@pytest.mark.parametrize("value", [None, -1e-9, float("nan")])
def test_verified_labels_require_all_five_valid_kkt_components(value):
    data = record("reference", 0, 2)
    if value is None:
        del data["cases"][0]["kkt"]["dual_feasibility"]
    else:
        data["cases"][0]["kkt"]["dual_feasibility"] = value
    with pytest.raises(ValueError):
        validate_record(data)


def test_different_numerical_backends_cannot_be_pooled():
    data = population()
    for row in data:
        row["environment"]["blas"] = [{"internal_api": "openblas", "num_threads": 8}]
    data[0]["environment"]["blas"][0]["internal_api"] = "mkl"
    with pytest.raises(ValueError, match="matched sources"):
        summarize(data, repetitions=3)


@pytest.mark.parametrize("tolerance", [float("nan"), 0, -1e-8, 1e-6])
def test_invalid_or_relaxed_kkt_target_cannot_support_the_comparison(tolerance):
    data = record("reference", 0, 2)
    data["configuration"]["nonlinear_tolerance"] = tolerance
    with pytest.raises(ValueError):
        validate_record(data)


def test_an_empty_population_is_not_a_complete_sequence():
    data = record("reference", 0, 2)
    data["configuration"]["queries"] = []
    data.update(cases=[], verified_problems=0)
    with pytest.raises(ValueError, match="nonempty"):
        validate_record(data)
