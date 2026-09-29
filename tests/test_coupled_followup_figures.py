"""Every method and failed outcome remains in plots generated from records."""

import pytest

from deflation_example.coupled_cost_report import confirmation
from deflation_example.coupled_followup_figures import (
    backend_figure,
    confirmation_figure,
    device_profile_figure,
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


def test_unsuccessful_sequence_duration_is_labeled_as_attempt_time(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    records = population()
    records[4]["status"] = "trust_radius_exhausted"
    labels = []
    monkeypatch.setattr(
        "deflation_example.coupled_followup_figures.save",
        lambda fig, *_: labels.append(fig.axes[0].get_ylabel()),
    )
    confirmation_figure(confirmation(records), tmp_path)
    assert labels == ["Elapsed attempt time (min)"]


def test_backend_figure_keeps_failures_and_both_execution_backends(tmp_path):
    pytest.importorskip("matplotlib")
    records = population()[:6]
    for i, r in enumerate(records):
        r["configuration"]["device"] = "hybrid" if i < 3 else "cuda"
    records[-1]["status"] = "budget_exhausted"
    backend_figure(records, tmp_path)
    assert (tmp_path / "backend_development.pdf").stat().st_size > 1000


def test_hybrid_confirmation_reports_host_and_gpu_memory(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    records = population()
    for record in records:
        record["configuration"]["device"] = "hybrid"
        record["memory"] = {
            "complete": True,
            "peak_host_rss_bytes": 8 * 2**30,
            "peak_gpu_process_bytes": 2**30,
        }
    captured = []
    monkeypatch.setattr(
        "deflation_example.coupled_followup_figures.save",
        lambda fig, *_: captured.extend(axis.get_ylabel() for axis in fig.axes),
    )
    confirmation_figure(confirmation(records), tmp_path)
    assert captured == [
        "Complete sequence time (min)",
        "Sampled host process RSS (GiB)",
        "Sampled GPU process allocation (GiB)",
    ]


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


def test_transfer_plot_distinguishes_identical_curves_and_uses_integer_ranks(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    methods = {
        policy: {"verified": True, "deployed_rank": 8, "iterations": 12}
        for policy in ("full", "sequential")
    }
    row = {"newly_active": 0, "newly_inactive": 0, "methods": methods}
    report = {
        "schema": "coupled-matched-transfer-v1",
        "status": "complete",
        "expected_systems": 2,
        "rows": [row, row],
    }
    captured = []
    monkeypatch.setattr(
        "deflation_example.coupled_followup_figures.save",
        lambda fig, *_: captured.append(fig),
    )
    transfer_figure(report, tmp_path)
    rank_axis = captured[0].axes[1]
    assert rank_axis.get_ylim() == (0, 9)
    assert all(float(tick).is_integer() for tick in rank_axis.get_yticks())
    assert [line.get_marker() for line in rank_axis.lines] == ["o", "s"]
    assert [line.get_linestyle() for line in rank_axis.lines] == ["-", "--"]
    assert rank_axis.lines[1].get_markerfacecolor() == "none"
    assert [text.get_text() for text in captured[0].axes[2].texts] == ["24", "24"]


def device_profile():
    return {
        "status": "verified",
        "repetitions": 5,
        "components": [
            {
                "action": action,
                "backend": backend,
                "verified": True,
                "timings": [{"wall_seconds": 0.1 + 0.001 * i} for i in range(5)],
            }
            for action, backends in (
                ("tangent", ("cpu", "cuda")),
                ("transpose", ("cpu", "cuda")),
                ("restricted_normal", ("cpu", "cuda")),
                ("frozen_inverse", ("cpu", "cuda_serial", "cuda_block_diagonal")),
            )
            for backend in backends
        ],
    }


def test_device_profile_keeps_all_actions_and_layouts(tmp_path):
    pytest.importorskip("matplotlib")
    device_profile_figure(device_profile(), tmp_path)
    assert (tmp_path / "operator_costs.pdf").stat().st_size > 1000


@pytest.mark.parametrize("change", ["failed", "missing", "partial", "nonfinite"])
def test_device_profile_rejects_unverified_or_partial_inputs(tmp_path, change):
    report = device_profile()
    if change == "failed":
        report["components"][0]["verified"] = False
    elif change == "missing":
        report["components"].pop()
    elif change == "partial":
        report["components"][0]["timings"].pop()
    else:
        report["components"][0]["timings"][0]["wall_seconds"] = float("nan")
    with pytest.raises(ValueError):
        device_profile_figure(report, tmp_path)
