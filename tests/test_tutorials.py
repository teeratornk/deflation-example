"""Regression checks for tutorial formulas, populations, and output ownership."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from deflation_example.tutorial_costs import break_even, history_costs, rank_costs, resource_costs
from deflation_example.tutorial_spectral import angle_diagnostics, spectral_examples
from deflation_example.tutorial_verification import accuracy_examples, stopping_examples
from deflation_example.tutorials import Records, run

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/benchmarks"


@pytest.fixture(scope="module")
def spectral():
    return spectral_examples()


def test_rank_loss_has_no_equal_rank_angle_or_bound(spectral):
    row = spectral["rank_loss"]
    assert (row["requested_rank"], row["retained_rank"]) == (2, 1)
    assert row["largest_angle_degrees"] is None
    assert row["residual_angle_bound"] is None
    assert row["separation_status"] == "rank_loss"
    assert row["retained_rank_comparison_angle_degrees"] == 0


def test_repeated_cutoff_and_complete_cluster(spectral):
    assert spectral["repeated_cutoff"]["largest_angle_degrees"] == 90
    assert spectral["repeated_cutoff"]["separation"] == 0
    assert spectral["repeated_cutoff"]["residual_angle_bound"] is None
    assert spectral["complete_cluster"]["residual_angle_bound"] == 0


def test_crossing_invalidates_unconditional_reference_gap(spectral):
    row = spectral["eigenvalue_crossing"]
    assert row["actual_sine"] > row["unconditional_delta_over_gap"]
    assert row["residual_angle_bound"] is None
    assert not row["sufficient_no_crossing"]


def test_diffusion_reports_distinct_formulas_and_no_condition_certificate(spectral):
    rows = [r for r in spectral["diffusion"] if r["coupling_bound_status"] == "nontrivial"]
    assert [(r["n"], r["corner_width"]) for r in rows] == [(4, 1), (6, 1)]
    np.testing.assert_allclose(
        [r["coupling_angle_bound"] for r in rows], [0.737969, 0.896046], atol=5e-7
    )
    np.testing.assert_allclose(
        [r["residual_angle_bound"] for r in rows], [0.721523, 0.892952], atol=5e-7
    )
    for row in rows:
        assert row["actual_sine"] < row["residual_angle_bound"] < row["coupling_angle_bound"]
        assert row["deflated_condition_upper_bound"] > row["original_condition"]
    assert any(r["coupling_bound_status"] == "trivial_one" for r in spectral["diffusion"])
    assert any(r["coupling_bound_status"] == "unavailable" for r in spectral["diffusion"])


def test_controlled_spd_retains_transition(spectral):
    assert len(spectral["controlled_spd"]["rows"]) == 8
    for row in spectral["controlled_spd"]["rows"]:
        assert row["residual_identity_error"] < 1e-12
        if row["residual_angle_bound"] is not None:
            assert row["actual_sine"] <= row["residual_angle_bound"] + 1e-12


def test_invalid_spectral_dimensions_rejected():
    with pytest.raises(ValueError, match="requested rank"):
        angle_diagnostics(np.eye(2), np.eye(2))


def test_accuracy_checks_have_separate_objective_and_residual_acceptance():
    data = accuracy_examples()
    release = data["active_set_release"]
    assert release["status"] == "converged"
    assert sum(r["left"] for r in release["history"]) == 1
    np.testing.assert_allclose(release["solution"], [0.6, 1])
    assert data["independent_optimizer"]["objective_difference"] < 1e-10
    objective = data["objective_and_stationarity"]
    assert objective["objective_error"] < 1e-12 and objective["gradient_inf"] > 1e-4
    trace = data["residual_recomputation"]
    assert trace["recurrence_candidate"] and not trace["original_acceptance"]
    assert trace["production_status"] == "converged"
    assert trace["production_original_residual"] <= trace["relative_tolerance"]
    projected = data["projected_equation_without_coarse_correction"]
    assert projected["zero_guess_projected_residual"] == 0
    assert projected["zero_guess_original_relative_residual"] == 1
    assert projected["corrected_original_relative_residual"] <= 1e-10


def test_all_stopping_protocols_and_failed_attempts_retained():
    records = Records(DATA)
    groups = stopping_examples(records)
    assert len(groups) == 3 and len(records.manifest) == 186
    assert all(len(g["rows"]) == 12 and len(g["attempts"]) == 60 for g in groups)
    assert [sum(r["passed_sequences"] for r in g["rows"]) for g in groups] == [45, 45, 60]
    failed = [r for r in groups[1]["rows"] if r["n"] == 32 and r["method"] == "deflated_gpu_qr"]
    assert all(r["accepted_median_seconds"] is None for r in failed)


def test_resource_summary_keeps_every_method_and_repetition():
    data = resource_costs(Records(DATA).load("gpu.json"))
    assert len(data["rows"]) == 36
    assert sum(r["attempts"] for r in data["rows"]) == 132
    assert {r["method"] for r in data["rows"]} == {
        "deflated_gpu_qr",
        "deflated_cpu_svd_gpu",
        "amgx_fresh_resources",
        "amgx_persistent_resources",
        "jacobi_gpu",
        "deflated_cpu",
    }


def test_rank_summary_preserves_capped_attempts_and_rejects_invalid_residual():
    raw = Records(DATA).load("extended/scale.json")
    report = rank_costs(raw)
    assert len(report["rows"]) == 12
    assert [(r["n"], r["rank"]) for r in report["rows"] if r["status"] == "maxiter"] == [
        (102, 0),
        (128, 0),
    ]
    raw["rows"][0]["cpu_original_residual"] = float("nan")
    with pytest.raises(ValueError, match="original residual"):
        rank_costs(raw)


def test_history_summary_and_pool_example():
    raw = Records(DATA).load("extended/controls.json")
    result = history_costs(raw)
    assert len(result["rows"]) == 18
    assert all(len(r["targets"]) == 7 for r in result["rows"])
    assert result["span_example"]["current_coarse_projection_norm"] == 0
    corrupted = deepcopy(raw)
    corrupted["recycling"][0]["sequences"][0]["rows"].pop()
    with pytest.raises(ValueError, match="history target"):
        history_costs(corrupted)


@pytest.mark.parametrize("saving, expected", [(0.1, 12), (0, None), (-0.1, None)])
def test_constant_cost_break_even(saving, expected):
    assert break_even(1.2, saving) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_savings_rejected(value):
    with pytest.raises(ValueError):
        break_even(1.2, value)


@pytest.mark.parametrize("name", ["../records.json", "/records.json", "..\\records.json"])
def test_unsafe_input_paths_rejected(name):
    with pytest.raises(ValueError, match="Unsafe"):
        Records(DATA).load(name)


def test_tutorial_writes_protocol_results_and_csv_and_preserves_existing_output(tmp_path):
    folder = tmp_path / "spectral"
    report = run("spectral", DATA, folder)
    assert report["protocol"]["diffusion"]["grids"] == [4, 6, 12]
    assert len(report["input_sha256"]) == 2
    assert len((folder / "spectral.csv").read_text().splitlines()) == 22
    assert json.loads((folder / "results.json").read_text())["tutorial_version"] == 1
    with pytest.raises(FileExistsError):
        run("spectral", DATA, folder)
