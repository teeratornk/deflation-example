"""The campaign keeps all declared controls and freezes stages before dispatch."""

import json
import sys

import pytest

from deflation_example import coupled_completion_campaign as campaign
from deflation_example.reporting import file_sha256, write_report


def test_screen_keeps_strong_controls_and_all_ranks():
    choices = campaign.screening_choices()
    assert len(choices) == len(set(choices)) == 8
    assert ("jacobi", 0, "frozen") in choices
    assert ("jacobi", 0, "jacobi") in choices
    assert {rank for policy, rank, _ in choices if policy == "preconditioned_coupled"} == {
        20,
        50,
        100,
        200,
    }
    arms = campaign.arm_settings("preconditioned_coupled", 20)
    assert arms["reference"]["rank"] == arms["recycling"]["rank"]
    assert arms["frozen"]["rank"] == 0


def test_screen_dispatch_is_coverage_complete_bounded_and_not_repeatable(tmp_path, monkeypatch):
    trace = tmp_path / "trace"
    trace.mkdir()
    write_report(
        trace / "manifest.json",
        {
            "schema": "coupled-inactive-trace-v1",
            "status": "complete",
            "quadratics": [{"partition": "selection"}] * 3 + [{"partition": "held_out"}],
        },
    )
    bank = tmp_path / "bank"
    bank.mkdir()
    write_report(
        bank / "record.json",
        {"status": "complete", "trace_sha256": file_sha256(trace / "manifest.json")},
    )
    protocol = tmp_path / "protocol.json"
    write_report(
        protocol,
        {
            "run_root": str(tmp_path),
            "trace": "trace",
            "bank": "bank",
            "numerical_source": "a" * 40,
            "campaign": "campaign",
            "launcher": "launcher",
        },
    )
    calls = []

    def submit(launcher, source, mode, arguments, dependencies=(), gpu=False):
        calls.append((source, mode, arguments, dependencies, gpu))
        return str(100 + len(calls))

    monkeypatch.setattr(campaign, "submit", submit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--protocol",
            str(protocol),
            "--stage",
            "screen",
            "--controller-source",
            "b" * 40,
        ],
    )
    campaign.main()
    assert len(calls) == 25
    assert all(row[0] == "a" * 40 and row[4] for row in calls[:-1])
    assert calls[-1][0] == "b" * 40 and not calls[-1][4]
    assert len(calls[-1][3]) == 4
    assert not any(row[3] for row in calls[:4])
    assert all(len(row[3]) == 1 for row in calls[4:-1])
    state = json.loads((tmp_path / "campaign/screen-jobs.json").read_text())
    assert len(state["jobs"]) == 24
    with pytest.raises(FileExistsError):
        campaign.main()
    assert len(calls) == 25
