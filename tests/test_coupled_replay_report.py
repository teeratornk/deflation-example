"""Recomputed replay summaries preserve failures and check their physical input."""

import json
import numpy as np
import pytest

from deflation_example.coupled_replay_report import summarize
from deflation_example.coupled_saved import file_digest
from deflation_example.reporting import write_fields, write_report


def fixture(tmp_path, status="converged"):
    original = tmp_path / "original.npz"
    write_fields(original, state=np.arange(6.0))
    directory = tmp_path / "replay"
    directory.mkdir()
    values = np.arange(6.0).reshape(2, 3) + 0.001
    if status != "converged":
        values = values[:1]
    times = np.array([0.1, 0.2])[: len(values)]
    write_fields(directory / "states.npz", state=values, times_s=times)
    record = {
        "configuration": {"slabs": 2},
        "subdivision": 1,
        "optimization_field_sha256": file_digest(original),
        "status": status,
        "environment": {"git_head": "source"},
        "forward_solver": {"tolerance": 1e-10},
        "steps": [{"time_s": t, "status": status} for t in times],
        "maximum_endpoint_difference_K": 0.02,
        "seconds": 1.0,
    }
    write_report(directory / "record.json", record)
    return original, directory, record


def test_complete_summary_recomputes_differences_from_fields(tmp_path):
    original, directory, _ = fixture(tmp_path)
    report = summarize(original, [directory], 20)
    assert report["rows"][0]["complete"]
    assert report["rows"][0]["maximum_endpoint_difference_K"] == pytest.approx(0.02)


def test_incomplete_and_missing_attempts_remain_visible(tmp_path):
    original, directory, _ = fixture(tmp_path, "coupling_iteration_cap")
    report = summarize(original, [directory, tmp_path / "missing"], 20)
    assert not any(row["complete"] for row in report["rows"])
    assert report["rows"][0]["maximum_endpoint_difference_K"] is None
    assert report["rows"][0]["maximum_recorded_difference_K"] == pytest.approx(0.02)
    assert report["rows"][1]["status"] == "missing"


@pytest.mark.parametrize(
    "field,value",
    [
        ("subdivision", 2),
        ("optimization_field_sha256", "other"),
        ("maximum_endpoint_difference_K", 1.0),
    ],
)
def test_mismatched_input_or_summary_is_rejected(tmp_path, field, value):
    original, directory, record = fixture(tmp_path)
    record[field] = value
    write_report(directory / "record.json", record)
    with pytest.raises(ValueError):
        summarize(original, [directory], 20)


def test_converged_status_requires_all_slabs(tmp_path):
    original, directory, record = fixture(tmp_path, "coupling_iteration_cap")
    record["status"] = "converged"
    write_report(directory / "record.json", record)
    with pytest.raises(ValueError, match="every converged"):
        summarize(original, [directory], 20)


def test_duplicate_directory_is_rejected(tmp_path):
    original, directory, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        summarize(original, [directory, directory], 20)


def test_changed_replay_time_is_rejected(tmp_path):
    original, directory, _ = fixture(tmp_path)
    path = directory / "record.json"
    record = json.loads(path.read_text())
    record["steps"][0]["time_s"] = 0.12
    write_report(path, record)
    with pytest.raises(ValueError, match="field times"):
        summarize(original, [directory], 20)
