import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from deflation_example.mesh_evidence import verify, memory_envelope


def exporter():
    directory = Path(__file__).parents[1] / "tools"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("mesh_export", directory / "export_mesh_records.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_export_and_verify_include_the_recorded_pilot_sources(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    (records / "record.json").write_text('{"success": false}\n')
    source = tmp_path / "source/deflation_example"
    source.mkdir(parents=True)
    (source / "benchmark_mesh.py").write_text('"""A source fixture."""\n')
    output = tmp_path / "evidence"
    manifest = exporter().run([f"pilot/coarse={records}"], [f"pilot-v1={source.parent}"], output)
    assert verify(output) == manifest
    assert "LICENSE" in manifest["files"]
    path = output / "sources/pilot-v1/deflation_example/benchmark_mesh.py"
    path.write_text('"""Changed source."""\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(output)


def test_manifest_rejects_unlisted_files_and_path_traversal(tmp_path):
    data = b'{}\n'
    path = tmp_path / "record.json"
    path.write_bytes(data)
    manifest = {"format": "mesh-cht-evidence-v1", "files": {
        "record.json": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert verify(tmp_path) == manifest
    (tmp_path / "unexpected.json").write_text('{}\n')
    with pytest.raises(ValueError, match="unlisted"):
        verify(tmp_path)
    manifest["files"]["../escape.json"] = manifest["files"]["record.json"]
    manifest["files"]["unexpected.json"] = manifest["files"]["record.json"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe"):
        verify(tmp_path)


def test_memory_envelope_uses_all_repetitions_and_keeps_rank_zero_fixed(tmp_path):
    studies = []
    for rank in (20, 100, 200):
        methods = [{"method": m, "accepted": 5, "requested": 5,
                    "complete_memory_measurements": 5, "median_seconds": 1000/rank,
                    "peak_gpu_bytes": [rank/100*2**30]*5}
                   for m in ("jacobi", "reference", "recycling", "amgx")]
        studies.append({"controls": {"geometry": "engine_3d", "level": 2,
                                     "transient": False, "rank": rank}, "methods": methods})
    rows = memory_envelope(studies, tmp_path)
    assert rows[1]["requested_rank"] == 100
    assert rows[5]["requested_rank"] == 200
    assert rows[4]["median_seconds"] == 10
    studies[-1]["methods"][1]["complete_memory_measurements"] = 4
    rows = memory_envelope(studies, tmp_path)
    assert rows[5]["requested_rank"] == 100
