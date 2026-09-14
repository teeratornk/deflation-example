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
    write_report(path, record)


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
        ("tracking_integral_refined_K2_m3_s", -1),
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
