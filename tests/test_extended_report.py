"""Reject incomplete, inconsistent, or unaccepted sequence evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from deflation_example.benchmark_extended_report import (
    check_partition,
    checked_sequences,
    summarize,
)


RECORDS = Path(__file__).resolve().parents[1] / "examples/benchmarks/extended/cht_v1"


def test_recorded_initial_protocol_retains_failures():
    report, sequences = checked_sequences(RECORDS / "results.json", require_clean_source=True)
    summary = summarize(report, sequences)
    assert len(sequences) == 60
    assert len(summary["rows"]) == 12
    assert any(r["passed_sequences"] < 5 for r in summary["rows"])
    assert all(
        r["passed_sequences"] == 5 for r in summary["rows"] if r["warm_start"] == "outer_inner"
    )


def test_final_protocol_accepts_every_target_in_every_sequence():
    report, sequences = checked_sequences(
        RECORDS.parent / "cht_v3/results.json", require_clean_source=True
    )
    assert report["specification"]["protocol"] == "extended-cht-pdas-v3"
    assert len(sequences) == 60 and all(s["success"] for s in sequences)
    assert sum(len(s["cases"]) for s in sequences) == 960


def test_process_startup_replaces_gpu_initialization_without_double_counting():
    report, sequences = checked_sequences(RECORDS / "results.json")
    startup = {
        "complete": True,
        "success": True,
        "specification": {"repetitions": 2},
        "rows": [
            {"success": True, "total_seconds": t, "components_seconds": {"all": t}}
            for t in (4.0, 6.0)
        ],
    }
    summary = summarize(report, sequences, startup)
    assert summary["common_seconds"] == report["calibration_seconds"] + 5
    for row in summary["rows"]:
        assert row["with_common_seconds"] == row["median_seconds"] + summary["common_seconds"]


def test_failed_process_startup_cannot_supply_a_preparation_total():
    report, sequences = checked_sequences(RECORDS / "results.json")
    startup = {"complete": False, "success": False, "rows": []}
    with pytest.raises(ValueError, match="Incomplete process-startup"):
        summarize(report, sequences, startup)


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("incomplete", "unfinished"),
        ("missing", "Missing complete-sequence"),
        ("hash", "file hash"),
        ("unsafe", "Unsafe sequence path"),
        ("residual", "original residual"),
        ("coarse", "safeguard"),
        ("timing", "sum to the total"),
    ],
)
def test_record_validator_rejects_corruption(tmp_path, mutation, match):
    report = json.loads((RECORDS / "results.json").read_text())
    # Symlinks keep the complete fixture unchanged; the corrupted sequence is separate.
    (tmp_path / "sequences").mkdir()
    for record in report["sequences"]:
        (tmp_path / record["path"]).symlink_to(RECORDS / record["path"])
    if mutation == "incomplete":
        report["complete"] = False
    elif mutation == "missing":
        report["sequences"].pop()
    elif mutation == "hash":
        report["sequences"][0]["sha256"] = "0" * 64
    elif mutation == "unsafe":
        report["sequences"][0]["path"] = "../results.json"
    else:
        record = next(r for r in report["sequences"] if r["method"] == "deflated_gpu_qr")
        sequence = json.loads((RECORDS / record["path"]).read_text())
        inner = sequence["cases"][0]["inner"][0]
        if mutation == "residual":
            inner["original_residual"] = 1e-5
        elif mutation == "coarse":
            inner["coarse_condition"] = 1e19
        else:
            sequence["components_seconds"]["assembly"] += 1
        payload = json.dumps(sequence).encode()
        record["path"] = "sequences/corrupted.json"
        (tmp_path / record["path"]).write_bytes(payload)
        record["sha256"] = hashlib.sha256(payload).hexdigest()
    (tmp_path / "results.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        checked_sequences(tmp_path / "results.json")


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_timing_partition_rejects_invalid_component(value):
    with pytest.raises(ValueError, match="Invalid timing"):
        check_partition({"setup": value, "solve": 1}, 1)
