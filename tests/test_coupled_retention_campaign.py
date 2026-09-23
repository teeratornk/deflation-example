"""Scheduler orchestration freezes settings and never resubmits a stage silently."""

import json
from pathlib import Path
import sys

import pytest

import deflation_example.coupled_retention_campaign as campaign
from deflation_example.reporting import file_sha256, write_report


def inputs(root):
    trace = root / "coupled-retention-capture-v8-10" / "linear-systems"
    trace.mkdir(parents=True)
    manifest = {
        "schema": "coupled-inactive-trace-v1",
        "status": "complete",
        "quadratics": [{"partition": "selection"}] * 3 + [{"partition": "held_out"}],
        "systems": [{"quadratic": i} for i in range(4)],
    }
    write_report(trace / "manifest.json", manifest)
    write_report(trace.parent / "record.json", {"all_problems_verified": True})
    bank = root / "coupled-retention-bank-v8-11"
    bank.mkdir()
    write_report(
        bank / "record.json",
        {"status": "complete", "trace_sha256": file_sha256(trace / "manifest.json")},
    )


def test_selection_dispatches_predeclared_grid_with_bounded_concurrency(tmp_path, monkeypatch):
    inputs(tmp_path)
    calls = []

    def submit(root, source, mode, arguments, dependency=None, gpu=False):
        assert root == tmp_path
        job = str(100 + len(calls))
        calls.append((mode, arguments, dependency, gpu))
        return job

    monkeypatch.setattr(campaign, "submit", submit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--run-root",
            str(tmp_path),
            "--source",
            "a" * 40,
            "--capture",
            "10",
            "--bank",
            "11",
            "--stage",
            "selection",
        ],
    )
    campaign.main()
    assert len(calls) == 28  # 3 quadratics x 9 policies, then the reporting controller.
    assert all(c[0] == "replay" and c[3] for c in calls[:-1])
    assert all(c[2] is None for c in calls[:4])
    assert all(len(c[2]) == 1 for c in calls[4:-1])
    assert calls[-1][0] == "dispatch" and len(calls[-1][2]) == 4 and not calls[-1][3]
    state_path = tmp_path / "coupled-retention-campaign-v8-11/selection-jobs.json"
    state = json.loads(state_path.read_text())
    assert len(state["jobs"]) == 27
    with pytest.raises(FileExistsError):
        campaign.main()
    assert len(calls) == 28


def test_scheduler_command_uses_argument_list_and_checks_identifier(tmp_path, monkeypatch):
    from types import SimpleNamespace

    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(stdout="123;cluster\n")

    monkeypatch.setattr(campaign.subprocess, "run", run)
    assert campaign.submit(tmp_path, "a" * 40, "replay", [1, 2], ["99"], True) == "123"
    assert "--gres=gpu:1" in seen[0] and "--dependency=afterany:99" in seen[0]
    assert Path(seen[0][-5]).name == "coupled-retention-v8.sbatch"
