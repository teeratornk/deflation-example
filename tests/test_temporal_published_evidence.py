"""Check every released temporal outcome and its original source checksum."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1] / "examples/temporal_resolution/data"


def test_temporal_archive_keeps_every_case_and_refinement_stage():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum
    for form in ("original", "corrected"):
        design = json.loads((ROOT / form / "design.json").read_text())
        rows = json.loads((ROOT / form / "summary.json").read_text())["rows"]
        assert len(rows) == len(design["cases"]) == 10
        for i, case in enumerate(design["cases"]):
            record = json.loads((ROOT / form / f"case-{i:02d}/record.json").read_text())
            assert all(record[k] == v for k, v in case.items())
            assert record["verified"] == rows[i]["verified"]
            assert record["environment"]["git_head"] == manifest["sources"][form]
            assert record["environment"]["source_sha256"] == design["environment"]["source_sha256"]
            if record["verified"]:
                assert record["optimization"]["success"]
                assert record["optimization"]["maximum_original_residual"] <= 1e-10
                assert max(record["optimization"]["kkt"].values()) <= 1e-8
        assert sum(row["verified"] for row in rows) == (6 if form == "original" else 10)
    for stage in ("initial", "extension", "final"):
        for i in (0, 4, 5, 9):
            replay = json.loads((ROOT / f"replays/{stage}/case-{i:02d}.json").read_text())
            record = json.loads((ROOT / f"corrected/case-{i:02d}/record.json").read_text())
            assert replay["fields_sha256"] == record["fields_sha256"]
            assert replay["selected_cases"] == [0, 4, 5, 9]
            if stage == "final":
                assert replay["status"] == "within_temperature_change_scale"
                assert all(r["maximum_original_residual"] <= 1e-10 for r in replay["rows"])
                assert all(r["temperature_change_max_K"] <= 0.05 for r in replay["rows"][-2:])
                assert replay["rows"][-1]["maximum_bound_violation_K"] > 0
