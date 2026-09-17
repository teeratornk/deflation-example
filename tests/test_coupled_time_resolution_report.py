"""Nested-grid comparisons use fields, common inputs and declared physical times."""

import json
import numpy as np
import pytest

from deflation_example.coupled_time_resolution_report import summarize
from deflation_example.reporting import write_fields, write_report


def case(tmp_path, subdivision, status="converged"):
    directory = tmp_path / f"subdivision-{subdivision}"
    directory.mkdir()
    times = np.arange(1, 2 * subdivision + 1) / subdivision
    state = np.column_stack((times, times**2))
    if status != "converged":
        times, state = times[:1], state[:1]
    write_fields(directory / "states.npz", state=state, times_s=times)
    record = {
        "configuration": {"slabs": 2},
        "subdivision": subdivision,
        "optimization_field_sha256": "same-source",
        "baseline_sha256": "same-baseline",
        "status": status,
        "environment": {"git_head": "source"},
        "forward_solver": {"procedure": "monolithic_newton", "tolerance": 1e-12},
        "steps": [{"time_s": t, "status": status} for t in times],
        "seconds": 1.0,
        "tracking_integral_refined_K2_m3_s": 3.0,
    }
    write_report(directory / "record.json", record)
    return directory


def change(directory, key, value):
    path = directory / "record.json"
    record = json.loads(path.read_text())
    record[key] = value
    # Deliberately allow malformed external JSON values in rejection tests;
    # the normal writer converts nonfinite measurements to null.
    path.write_text(json.dumps(record))


def test_nested_comparison_samples_shared_times_and_recomputes_error(tmp_path):
    coarse, fine = case(tmp_path, 2), case(tmp_path, 4)
    with np.load(fine / "states.npz") as fields:
        state, times = fields["state"].copy(), fields["times_s"].copy()
    state[-1, 0] += 0.001
    write_fields(fine / "states.npz", state=state, times_s=times)
    change(coarse, "tracking_integral_refined_K2_m3_s", 2.98)
    result = summarize([fine, coarse], 20)
    pair = result["pairs"][0]
    assert pair["coarse_slabs"] == 4
    assert pair["fine_slabs"] == 8
    assert pair["maximum_shared_time_difference_K"] == pytest.approx(0.02)
    assert pair["tracking_relative_change_fine_denominator"] == pytest.approx(0.02 / 3)
    assert pair["shared_time_thresholds_met"]


def test_partial_and_missing_computations_remain_visible(tmp_path):
    complete = case(tmp_path, 1)
    partial = case(tmp_path, 2, "newton_iteration_cap")
    result = summarize([complete, partial, tmp_path / "missing"], 20)
    assert len(result["rows"]) == 3
    assert sum(row["complete"] for row in result["rows"]) == 1
    assert result["pairs"] == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("optimization_field_sha256", "different"),
        ("baseline_sha256", "different"),
        ("configuration", {"slabs": 3}),
        ("forward_solver", {"procedure": "monolithic_newton", "tolerance": 1e-8}),
        ("forward_formulation", {"consistent_stabilization": True}),
        ("environment", {"git_head": "another-source"}),
        ("tracking_integral_refined_K2_m3_s", -1),
        ("maximum_recorded_upper_violation_K", -1),
        ("maximum_recorded_lower_violation_K", float("nan")),
    ],
)
def test_mismatched_inputs_or_invalid_integral_rejected(tmp_path, key, value):
    coarse, fine = case(tmp_path, 1), case(tmp_path, 2)
    change(fine, key, value)
    with pytest.raises(ValueError):
        summarize([coarse, fine], 20)


def test_incomplete_fields_cannot_support_converged_status(tmp_path):
    directory = case(tmp_path, 2, "newton_iteration_cap")
    change(directory, "status", "converged")
    with pytest.raises(ValueError, match="every recorded state"):
        summarize([directory], 20)


def test_nonnested_or_duplicate_grids_are_rejected(tmp_path):
    a, b = case(tmp_path, 2), case(tmp_path, 3)
    with pytest.raises(ValueError, match="distinct and nested"):
        summarize([a, b], 20)
    with pytest.raises(ValueError, match="distinct forward"):
        summarize([a, a], 20)


@pytest.mark.parametrize("coarse_integral,relative", [(0.0, 0.0), (3.0, None)])
def test_zero_tracking_norm_has_explicit_relative_difference(tmp_path, coarse_integral, relative):
    a, b = case(tmp_path, 1), case(tmp_path, 2)
    change(a, "tracking_integral_refined_K2_m3_s", coarse_integral)
    change(b, "tracking_integral_refined_K2_m3_s", 0.0)
    result = summarize([a, b], 20)
    pair = result["pairs"][0]
    assert pair["tracking_relative_change_fine_denominator"] == relative
    assert pair["shared_time_thresholds_met"] == (relative == 0.0)
    write_report(tmp_path / "summary.json", result)


def test_cli_writes_pairwise_figure_and_summary(tmp_path, monkeypatch):
    import sys
    from deflation_example.coupled_time_resolution_report import main

    a, b = case(tmp_path, 1), case(tmp_path, 2)
    output = tmp_path / "summary"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summary",
            "--replays",
            str(a),
            str(b),
            "--temperature-scale",
            "20",
            "--output",
            str(output),
            "--plot",
        ],
    )
    main()
    report = json.loads((output / "summary.json").read_text())
    assert len(report["pairs"]) == 1
    assert (output / "time_resolution.pdf").stat().st_size > 0


def test_resolution_requires_three_complete_grids_and_retains_bound_excess(tmp_path):
    runs = [case(tmp_path, n) for n in (1, 2, 4)]
    change(runs[-1], "maximum_recorded_upper_violation_K", 0.23)
    report = summarize(runs, 20)
    assert report["resolution_assessment"]["discrete_time_resolution_met"]
    assert report["rows"][-1]["maximum_recorded_upper_violation_K"] == 0.23
    assert report["rows"][0]["maximum_recorded_upper_violation_K"] is None
    # Temperature resolution does not imply constraint feasibility.
    pair = summarize(runs[:2], 20)["resolution_assessment"]
    assert pair["status"] == "insufficient_refinements_for_trend"
    assert not pair["discrete_time_resolution_met"]
    missing = summarize([*runs, tmp_path / "missing"], 20)["resolution_assessment"]
    assert missing["status"] == "incomplete_declared_refinements"
    assert not missing["discrete_time_resolution_met"]


@pytest.mark.parametrize("quantity", ["temperature", "tracking"])
def test_increasing_refinement_changes_fail_even_below_threshold(tmp_path, quantity):
    runs = [case(tmp_path, n) for n in (1, 2, 4)]
    if quantity == "tracking":
        for directory, integral in zip(runs, (3.0, 3.001, 3.01), strict=True):
            change(directory, "tracking_integral_refined_K2_m3_s", integral)
    else:
        for directory, offset in zip(runs, (0.0, 0.0001, 0.001), strict=True):
            with np.load(directory / "states.npz") as fields:
                state, times = fields["state"].copy(), fields["times_s"].copy()
            write_fields(directory / "states.npz", state=state + offset, times_s=times)
    result = summarize(runs, 20)["resolution_assessment"]
    assert result["last_pair_thresholds_met"]
    assert result["status"] == "refinement_trend_not_decreasing"
    assert not result["discrete_time_resolution_met"]


def test_failed_finest_run_cannot_be_omitted_by_the_resolution_assessment(tmp_path):
    runs = [case(tmp_path, n) for n in (1, 2, 4)]
    runs.append(case(tmp_path, 8, "newton_iteration_cap"))
    result = summarize(runs, 20)["resolution_assessment"]
    assert result["last_pair_thresholds_met"]
    assert result["last_two_changes_nonincreasing"]
    assert not result["discrete_time_resolution_met"]


def test_intermediate_time_differences_can_fail_despite_identical_endpoints(tmp_path):
    runs = [case(tmp_path, n) for n in (1, 2, 4)]
    endpoints = summarize(runs, 1)
    assert endpoints["resolution_assessment"]["discrete_time_resolution_met"]
    whole = summarize(runs, 1, initial_value=0)
    pair = whole["pairs"][-1]
    assert pair["maximum_shared_time_difference_K"] == 0
    assert pair["maximum_all_refined_time_difference_K"] == pytest.approx(0.0625)
    assert not whole["resolution_assessment"]["discrete_time_resolution_met"]
    assert whole["resolution_assessment"]["temperature_comparison"] == (
        "all refined times versus linear coarse interpolation"
    )


def test_all_time_comparison_includes_initial_interval_and_has_decreasing_error(tmp_path):
    runs = [case(tmp_path, n) for n in (4, 8, 16)]
    result = summarize(runs, 1, initial_value=0)
    assert result["resolution_assessment"]["discrete_time_resolution_met"]
    pair = result["pairs"][-1]
    assert pair["maximum_all_refined_time_difference_K"] == pytest.approx(0.00390625)
    assert pair["maximum_interpolated_difference_K"][0] == pytest.approx(0.00390625)
    assert len(pair["refined_times_s"]) == 32
    changed_initial = summarize(runs, 1, initial_value=1)
    assert changed_initial["pairs"][-1]["maximum_all_refined_time_difference_K"] > 0.49
    assert not changed_initial["resolution_assessment"]["discrete_time_resolution_met"]


def test_one_passing_pair_does_not_satisfy_two_successive_comparisons(tmp_path):
    runs = [case(tmp_path, n) for n in (2, 4, 8)]
    gate = summarize(runs, 1, initial_value=0)["resolution_assessment"]
    assert gate["last_pair_thresholds_met"]
    assert not gate["last_two_pairs_thresholds_met"]
    assert not gate["discrete_time_resolution_met"]


def test_nonfinite_initial_value_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="initial"):
        summarize([case(tmp_path, 1)], 20, initial_value=float("nan"))


def test_peak_location_and_traces_preserve_signed_temperature_difference(tmp_path):
    coarse, fine = case(tmp_path, 2), case(tmp_path, 4)
    with np.load(fine / "states.npz") as fields:
        state, times = fields["state"].copy(), fields["times_s"].copy()
    state[3, 0] -= 2
    write_fields(fine / "states.npz", state=state, times_s=times)
    pair = summarize([coarse, fine], 20, initial_value=0)["pairs"][0]
    peak = pair["all_refined_time_peak"]
    assert peak == {
        "time_s": 1.0,
        "fine_time_index": 3,
        "free_node_index": 0,
        "signed_fine_minus_coarse_K": -40.0,
    }
    assert pair["shared_time_peak"]["free_node_index"] == 0
    trace = pair["peak_node_trace"]
    assert trace["fine_dimensionless_temperature"][4] == -1
    assert trace["coarse_dimensionless_temperature"][2] == 1
    assert trace["coarse_times_s"][0] == trace["fine_times_s"][0] == 0
