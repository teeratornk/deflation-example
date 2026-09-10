import json

import pytest
from threadpoolctl import threadpool_limits

from deflation_example.mesh_support import run
from deflation_example.validate_mesh import validate


@pytest.fixture(scope="module")
def validation():
    with threadpool_limits(2):
        return validate()


def test_physical_verification_covers_complete_temporal_coupling(validation):
    assert validation["success"]
    assert validation["settings"] == "physical"
    assert len(validation["cases"]) == 12
    transformer = [r for r in validation["cases"] if r["geometry"] == "transformer_2d"]
    assert {r["alpha"] for r in transformer} == {1e-14}
    assert all(sum(r["time_steps"]) < 5e-5 for r in transformer if r["time_steps"])
    assert max(r["state_relative_difference"] for r in validation["cases"]) < 1e-10


def test_support_checks_formulas_and_nonfinite_metrics(validation, tmp_path):
    source = tmp_path / "validation.json"
    source.write_text(json.dumps(validation))
    report = run(source, {}, {}, tmp_path / "good")
    assert report["validation"]["success"]
    invalid = json.loads(source.read_text())
    invalid["cases"][0]["state_relative_difference"] = float("nan")
    source.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="finite"):
        run(source, {}, {}, tmp_path / "bad")


def test_validation_rejects_unknown_settings():
    with pytest.raises(ValueError, match="settings"):
        validate("unknown")
