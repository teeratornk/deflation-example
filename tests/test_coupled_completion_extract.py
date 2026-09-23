"""The portable extract preserves every measured outcome and deployment."""

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/coupled_completion"
summarize_extract = runpy.run_path(str(EXAMPLE / "reproduce.py"))["summarize_extract"]


def test_all_recorded_solves_remain_in_the_portable_summary():
    bundle = json.loads((EXAMPLE / "measurements.json").read_text())
    for group in bundle["groups"]:
        for record in group["records"]:
            assert record["final_residual_tolerance"] == 1e-10
            assert record["status"] == "complete"
            for row in record["rows"]:
                assert row["verified"] and row["status"] == "converged"
                assert 0 <= row["original_residual"] <= record["final_residual_tolerance"]
                assert row["deployed_rank"] == record["rank"] and row["fallback"] is None
    summary = summarize_extract(EXAMPLE / "measurements.json")
    assert [g["solve_count"] for g in summary["groups"]] == [72, 45]
    first, low_rank = summary["groups"]
    assert len(first["summary"]["groups"]) == 2
    assert len(low_rank["summary"]["groups"]) == 1
    assert all(
        not g["summary"]["reference_beats_jacobi"]
        for comparison in summary["groups"]
        for g in comparison["summary"]["groups"]
    )
    rows = low_rank["summary"]["groups"][0]["summary"]["rows"]
    assert len(rows) == 5 and all(row["eligible"] for row in rows)
    baseline = next(row for row in rows if row["rank"] == 0)
    rank8 = next(row for row in rows if row["rank"] == 8)
    assert baseline["total_iterations"] == [53] * 3
    assert rank8["total_iterations"] == [47] * 3
    assert baseline["median_replay_total_seconds"] == pytest.approx(204.5626606149599)
    assert rank8["construction_seconds_once"] == pytest.approx(431.32266974821687)
    assert rank8["median_replay_total_seconds"] == pytest.approx(656.7036090269685)


@pytest.mark.parametrize("defect", ["duplicate_record", "negative_index", "unsafe_name"])
def test_invalid_extract_is_rejected(tmp_path, defect):
    bundle = json.loads((EXAMPLE / "measurements.json").read_text())
    group = bundle["groups"][0]
    if defect == "duplicate_record":
        group["records"].append(deepcopy(group["records"][0]))
    elif defect == "negative_index":
        group["records"][0]["environment_index"] = -1
    else:
        group["name"] = "../elsewhere"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError):
        summarize_extract(path)
