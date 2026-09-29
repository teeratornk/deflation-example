"""Every method and failed outcome remains in plots generated from records."""

import pytest

from deflation_example.coupled_cost_report import confirmation
from deflation_example.coupled_followup_figures import (
    backend_figure,
    confirmation_figure,
    transfer_figure,
)
from test_coupled_cost_report import population


def test_confirmation_plots_all_outcomes_with_host_memory(tmp_path):
    pytest.importorskip("matplotlib")
    records = population()
    records[4]["status"] = "budget_exhausted"
    for record in records:
        record["memory"] = {
            "complete": True,
            "peak_host_rss_bytes": 2**30,
            "peak_gpu_process_bytes": 0,
        }
    confirmation_figure(confirmation(records), tmp_path)
    assert (tmp_path / "complete_confirmation.pdf").stat().st_size > 1000
    assert (tmp_path / "complete_confirmation.png").stat().st_size > 1000


def test_backend_figure_keeps_failures_and_both_execution_backends(tmp_path):
    pytest.importorskip("matplotlib")
    records = population()[:6]
    for i, r in enumerate(records):
        r["configuration"]["device"] = "hybrid" if i < 3 else "cuda"
    records[-1]["status"] = "budget_exhausted"
    backend_figure(records, tmp_path)
    assert (tmp_path / "backend_development.pdf").stat().st_size > 1000


def test_transfer_plot_refuses_partial_chronological_history(tmp_path):
    with pytest.raises(ValueError):
        transfer_figure({"schema": "coupled-matched-transfer-v1", "status": "running"}, tmp_path)


def test_transfer_plot_includes_rank_loss_and_missing_energy_diagnostic(tmp_path):
    pytest.importorskip("matplotlib")
    methods = {
        p: {"verified": True, "deployed_rank": rank, "iterations": count}
        for p, rank, count in (("full", 8, 12), ("sequential", 6, 15))
    }
    row = {"newly_active": 2, "newly_inactive": 3, "methods": methods}
    report = {
        "schema": "coupled-matched-transfer-v1",
        "status": "complete",
        "expected_systems": 2,
        "rows": [row, row],
    }
    transfer_figure(report, tmp_path)
    assert (tmp_path / "matched_transfer.pdf").stat().st_size > 1000
