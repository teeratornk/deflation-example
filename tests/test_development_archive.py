"""Preserve every relocated development outcome and its numerical scope."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1] / "examples/development_studies"


def test_archived_sources_and_all_cases_remain_available():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == entry["sha256"]
    data = json.loads((ROOT / "studies.json").read_text())
    assert len(data["construction_stopping"]) == 8
    assert len(data["finer_transformer_attempts"]) == 8
    assert sum(r["complete_success"] for r in data["finer_transformer_attempts"]) == 4
    text = (ROOT / "README.md").read_text()
    for index in range(1, 9):
        assert f"| P{index} |" in text
    assert "no replacement" in text


def test_temporal_snapshot_matches_the_published_population():
    data = json.loads((ROOT / "sources/temporal_resolution.json").read_text())
    original = json.loads(
        (ROOT.parent / "temporal_resolution/data/original/summary.json").read_text()
    )
    corrected = json.loads(
        (ROOT.parent / "temporal_resolution/data/corrected/summary.json").read_text()
    )
    assert data["original_rows"] == original["rows"]
    assert data["corrected_rows"] == corrected["rows"]
    for replay in data["replays"]:
        case = replay["case"]
        raw = json.loads(
            (
                ROOT.parent / f"temporal_resolution/data/replays/final/case-{case:02d}.json"
            ).read_text()
        )
        assert replay["rows"] == raw["rows"]
