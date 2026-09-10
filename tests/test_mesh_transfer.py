import json

import numpy as np
from omegaconf import OmegaConf
import pytest
from threadpoolctl import threadpool_limits

from deflation_example.benchmark_mesh import controls, sequence
from deflation_example.mesh_transfer import run
from deflation_example.reporting import environment


@pytest.fixture
def source(tmp_path):
    c = controls(OmegaConf.create({"targets": 2, "rank": 5, "methods": ["reference"],
                                   "spatial_reference": "scaled_schur", "save_fields": False}))
    with threadpool_limits(2):
        record, _, _ = sequence(c, "reference")
    assert record["success"]
    record.update(controls=c, environment=environment())
    path = tmp_path / "source.json"
    path.write_text(json.dumps(record))
    return path


def test_mesh_transfer_replays_exact_systems_and_reports_disjoint_costs(source, tmp_path):
    with threadpool_limits(2):
        result = run(source, tmp_path / "replay", samples=3, repeats=2)
    assert result["success"]
    assert any(row["newly_inactive"] for row in result["rows"])
    for row in result["rows"]:
        assert row["transfer_identity_error"] <= 1e-12
        for method in row["methods"].values():
            assert method["correction"]["status"] == "verified"
            assert "effective_condition" in method["correction"]
            for rep in method["repetitions"]:
                np.testing.assert_allclose(rep["seconds"], rep["basis_transfer_seconds"] + rep["kernel_seconds"])
                assert rep["original_residual"] <= 1e-10
    first = result["rows"][0]["methods"]
    assert first["full_reference"]["correction"] == first["sequential_transfer"]["correction"]


def test_mesh_transfer_rejects_changed_numerical_source(source, tmp_path):
    record = json.loads(source.read_text())
    record["environment"]["source_sha256"]["mesh_reference.py"] = "wrong"
    source.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="recorded numerical source"):
        run(source, tmp_path / "replay")
    assert not (tmp_path / "replay").exists()
