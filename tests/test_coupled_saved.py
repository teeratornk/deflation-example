"""Fixed-source studies select verified fields and their actual baseline."""

import numpy as np
import pytest

from deflation_example.coupled_saved import load_saved_solution, require_matching_baseline
from deflation_example.reporting import write_fields, write_report


@pytest.mark.parametrize("position", [None, 2])
def test_verified_pilot_and_sequence_sources_are_loaded_unchanged(tmp_path, position):
    values = np.array([-2.0, 0.0, 3.0])
    cfg = {"method": "reference", "query": 7, "upper_K": 357.3}
    record = {"configuration": cfg, "baseline_sha256": "original"}
    if position is None:
        record["results"] = [{"method": "reference", "status": "converged"}]
        filename = "reference-fields.npz"
    else:
        record["cases"] = [{"position": 2, "target": 15, "upper_K": 353.3, "verified": True}]
        filename = "target-02.npz"
    write_report(tmp_path / "record.json", record)
    write_fields(tmp_path / filename, state=values + 1, control=values, desired=values + 2)
    original, selected, fields, digest = load_saved_solution(tmp_path, "reference", position)
    np.testing.assert_array_equal(fields["control"], values)
    assert len(digest) == 64
    assert selected["query"] == (7 if position is None else 15)
    assert selected["upper_K"] == (357.3 if position is None else 353.3)
    require_matching_baseline(original, {"baseline_sha256": "original"})
    with pytest.raises(ValueError, match="baseline differs"):
        require_matching_baseline(original, {"baseline_sha256": "different"})
    if position is None:
        record["results"][0]["status"] = "globalization_failed"
    else:
        record["cases"][0]["verified"] = False
    write_report(tmp_path / "record.json", record)
    with pytest.raises(ValueError, match="verified"):
        load_saved_solution(tmp_path, "reference", position)


def test_sequence_method_and_position_are_checked_before_loading_fields(tmp_path):
    write_report(tmp_path / "record.json", {"configuration": {"method": "jacobi"}, "cases": []})
    with pytest.raises(ValueError, match="method differs"):
        load_saved_solution(tmp_path, "reference", 0)
    with pytest.raises(ValueError, match="verified target"):
        load_saved_solution(tmp_path, "jacobi", 0)
