"""Hard scheduler stops must remain distinct from numerical convergence."""

import json
import sys

import pytest

from deflation_example.coupled_run_status import main, summarize


def manifest(state="TIMEOUT", seconds=172800):
    return {
        "schema": "coupled-run-status-manifest-v1",
        "runs": [
            {
                "label": "reference-repeat-0",
                "record": "reference/record.json",
                "scheduler": {
                    "state": state,
                    "elapsed_seconds": seconds,
                    "exit_code": "0:0",
                    "started": seconds > 0,
                },
            }
        ],
    }


def save_record(root, status="running", **extra):
    path = root / "reference" / "record.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"status": status, **extra}))
    return path


def test_timeout_overrides_stale_record_without_rewriting_it(tmp_path):
    path = save_record(tmp_path)
    original = path.read_bytes()
    result = summarize(manifest(), tmp_path)
    row = result["runs"][0]
    assert row["outcome"] == "scheduler_timeout"
    assert row["record_status"] == "running"
    assert row["stale_running_record"]
    assert row["scheduler"]["exit_code"] == "0:0"
    assert row["record_sha256"]
    assert result["all_processes_terminal"]
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "state,seconds,outcome",
    [
        ("CANCELLED", 0, "cancelled_before_start"),
        ("CANCELLED", 15, "scheduler_cancelled"),
        ("TIMEOUT", 172800, "scheduler_timeout"),
        ("OUT_OF_MEMORY", 90, "scheduler_out_of_memory"),
        ("PENDING", 0, "scheduler_pending"),
        ("RUNNING", 90, "scheduler_running"),
    ],
)
def test_missing_outputs_keep_every_scheduler_outcome(tmp_path, state, seconds, outcome):
    row = summarize(manifest(state, seconds), tmp_path)["runs"][0]
    assert row["outcome"] == outcome
    assert row["record_error"] == "missing_record"
    assert row["terminal"] == (state not in {"PENDING", "RUNNING"})


def test_numerical_budget_and_scheduler_failure_are_both_visible(tmp_path):
    save_record(tmp_path, "budget_exhausted", verified_problems=0)
    row = summarize(manifest("FAILED", 86600), tmp_path)["runs"][0]
    assert row["outcome"] == "scheduler_failed"
    assert row["record_status"] == "budget_exhausted"
    assert row["reported_verified_problems"] == 0


@pytest.mark.parametrize("text", ['{"status":', "[]", '{"residual": NaN}', "\xff"])
def test_invalid_records_remain_visible(tmp_path, text):
    path = save_record(tmp_path)
    path.write_bytes(text.encode("latin-1"))
    row = summarize(manifest(), tmp_path)["runs"][0]
    assert row["record_error"] == "invalid_record"
    assert row["outcome"] == "scheduler_timeout"


def test_process_success_is_not_accuracy_certification(tmp_path):
    save_record(tmp_path, "complete", all_problems_verified=True, verified_problems=3)
    result = summarize(manifest("COMPLETED", 10), tmp_path)
    row = result["runs"][0]
    assert row["outcome"] == "process_completed"
    assert row["reported_all_problems_verified"] is True
    assert "publication_gate_passed" not in result
    assert "speedup" not in result


def test_zero_elapsed_time_alone_does_not_establish_nonstart(tmp_path):
    data = manifest("CANCELLED", 0)
    del data["runs"][0]["scheduler"]["started"]
    row = summarize(data, tmp_path)["runs"][0]
    assert row["outcome"] == "scheduler_cancelled"


@pytest.mark.parametrize("started", [False, 1, "yes"])
def test_inconsistent_start_information_is_rejected(tmp_path, started):
    data = manifest()
    data["runs"][0]["scheduler"]["started"] = started
    with pytest.raises(ValueError):
        summarize(data, tmp_path)


def test_invalid_environment_is_reported(tmp_path):
    save_record(tmp_path, environment=[])
    row = summarize(manifest(), tmp_path)["runs"][0]
    assert row["record_error"] == "invalid_record"
    assert row["record_sha256"]


@pytest.mark.parametrize("defect", ["label", "path", "state", "negative", "bool", "float"])
def test_invalid_manifest_is_refused(tmp_path, defect):
    data = manifest()
    row = data["runs"][0]
    if defect == "label":
        data["runs"].append(dict(row))
    elif defect == "path":
        data["runs"].append({**row, "label": "other"})
    elif defect == "state":
        row["scheduler"]["state"] = "CANCELLED+"
    else:
        row["scheduler"]["elapsed_seconds"] = {"negative": -1, "bool": True, "float": 2.0}[defect]
    with pytest.raises(ValueError):
        summarize(data, tmp_path)


@pytest.mark.parametrize("path", ["../outside.json", "/outside.json", ""])
def test_paths_must_stay_within_manifest_directory(tmp_path, path):
    data = manifest()
    data["runs"][0]["record"] = path
    with pytest.raises(ValueError):
        summarize(data, tmp_path)


def test_cli_retains_population_and_refuses_overwrite(tmp_path, monkeypatch):
    data = manifest()
    data["runs"].append(
        {
            "label": "unstarted-repeat",
            "record": "unstarted/record.json",
            "scheduler": {"state": "CANCELLED", "elapsed_seconds": 0, "started": False},
        }
    )
    save_record(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    output = tmp_path / "outcomes"
    monkeypatch.setattr(sys, "argv", ["status", "--manifest", str(path), "--output", str(output)])
    main()
    result = json.loads((output / "summary.json").read_text())
    assert result["outcome_counts"] == {"cancelled_before_start": 1, "scheduler_timeout": 1}
    assert result["manifest_sha256"]
    with pytest.raises(FileExistsError):
        main()
