"""A failed, missing or duplicated solve must never create a performance win."""

from copy import deepcopy

import pytest

from deflation_example.coupled_retention_report import plot, replay_label, summarize


def records():
    manifest = {
        "quadratics": [{"partition": "selection"}, {"partition": "held_out"}],
        "systems": [{"quadratic": 0}, {"quadratic": 1}],
    }
    rows = []
    for policy, rank, seconds in (("jacobi", 0, 10), ("thermal", 20, 5)):
        rows.append(
            {
                "policy": policy,
                "rank": rank,
                "width": 5,
                "feedback": None,
                "partition": "selection",
                "quadratic": 0,
                "trace_sha256": "t",
                "bank_sha256": "b",
                "construction_seconds_once": 0 if not rank else 2,
                "status": "complete",
                "cleanup_seconds": [0, 0, 0],
                "resource_creation_seconds": [0, 0, 0],
                "final_residual_tolerance": 1e-10,
                "rows": [
                    {
                        "system": 0,
                        "repetition": r,
                        "verified": True,
                        "status": "converged",
                        "original_residual": 1e-12,
                        "solve_seconds": seconds,
                        "iterations": 4,
                    }
                    for r in range(3)
                ],
            }
        )
    return manifest, rows


def test_construction_charged_once_and_repetition_totals_retained():
    manifest, rows = records()
    summary = summarize(manifest, rows)
    assert summary["reference_beats_jacobi"]
    assert summary["best_reference"]["replay_totals_seconds"] == [7, 7, 7]


@pytest.mark.parametrize("defect", ["missing", "failed", "duplicate", "incomplete_cleanup"])
def test_incomplete_evidence_is_ineligible(defect):
    manifest, rows = records()
    if defect == "missing":
        rows[1]["rows"].pop()
    elif defect == "failed":
        rows[1]["rows"][0]["verified"] = False
    elif defect == "duplicate":
        rows[1]["rows"].append(deepcopy(rows[1]["rows"][0]))
    else:
        rows[1]["cleanup_seconds"].pop()
    summary = summarize(manifest, rows)
    assert not summary["reference_beats_jacobi"]
    assert summary["best_reference"] is None


def test_different_bank_rejected():
    manifest, rows = records()
    rows[1]["bank_sha256"] = "other"
    with pytest.raises(ValueError, match="same trace"):
        summarize(manifest, rows)


def test_preconditioners_are_distinct_and_reference_competes_with_fastest_control():
    manifest, rows = records()
    frozen = deepcopy(rows[0])
    frozen.update(preconditioner="frozen", frozen_sweeps=3)
    for row in frozen["rows"]:
        row["solve_seconds"] = 1
    summary = summarize(manifest, [*rows, frozen])
    assert len(summary["rows"]) == 3
    assert not summary["reference_beats_jacobi"]
    assert summary["best_reference"]["median_replay_total_seconds"] == 7


def test_negative_residual_cannot_pass_verification():
    manifest, rows = records()
    rows[1]["rows"][0]["original_residual"] = -1
    assert not summarize(manifest, rows)["reference_beats_jacobi"]


def test_selection_records_do_not_supply_held_out_coverage():
    manifest, rows = records()
    summary = summarize(manifest, rows, "held_out")
    assert summary["rows"] == []
    assert not summary["reference_beats_jacobi"]


def test_cost_model_keeps_construction_and_online_work_separate():
    manifest, rows = records()
    rows[1]["construction_seconds_once"] = 12
    summary = summarize(manifest, rows)
    model = summary["amortization"]["rows"][0]
    assert not summary["reference_beats_jacobi"]
    assert model["online_saving_per_replay_block_seconds"] == 5
    assert model["constant_cost_break_even_replay_blocks"] == 3
    assert model["constant_cost_break_even_inner_solves"] == 3


@pytest.mark.parametrize("seconds", [10, 12])
def test_cost_model_has_no_break_even_without_online_savings(seconds):
    manifest, rows = records()
    for row in rows[1]["rows"]:
        row["solve_seconds"] = seconds
    model = summarize(manifest, rows)["amortization"]["rows"][0]
    assert model["constant_cost_break_even_replay_blocks"] is None
    assert model["online_saving_per_replay_block_seconds"] == 10 - seconds


def test_cost_model_excludes_incomplete_comparisons():
    manifest, rows = records()
    rows[0]["rows"].pop()
    assert summarize(manifest, rows)["amortization"]["rows"] == []


def test_plot_labels_name_the_actual_preconditioner(tmp_path):
    assert (
        replay_label({"policy": "jacobi", "rank": 0, "preconditioner": "frozen"})
        == "CG\nFrozen preconditioner\nr=0"
    )
    assert replay_label({"policy": "jacobi", "rank": 0}) == "CG\nJacobi preconditioner\nr=0"
    manifest, rows = records()
    rows[1].update(policy="krylov_coupled", rank=8, preconditioner="frozen")
    summary = summarize(manifest, rows)
    plot(summary, tmp_path)
    assert (tmp_path / "replay_cost.png").stat().st_size > 1000
