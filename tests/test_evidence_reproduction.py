"""Frozen data are verified before publication summaries are regenerated."""

import hashlib
import json

import pytest

from deflation_example.benchmark_evidence import run, verify


def evidence(tmp_path):
    payload = b'{"success": true}\n'
    (tmp_path / "record.json").write_bytes(payload)
    manifest = {
        "format": "reference-policy-evidence-v1",
        "files": {
            "record.json": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_evidence_verifies_bytes_and_rejects_tampering(tmp_path):
    manifest = evidence(tmp_path)
    assert run(tmp_path) == manifest
    (tmp_path / "record.json").write_text('{"success": false}')
    with pytest.raises(ValueError, match="mismatch"):
        verify(tmp_path)


@pytest.mark.parametrize("name", ["../record.json", "/record.json", "bad\\record.json"])
def test_evidence_rejects_path_traversal(tmp_path, name):
    manifest = evidence(tmp_path)
    manifest["files"][name] = manifest["files"].pop("record.json")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe|every JSON"):
        verify(tmp_path)


def test_evidence_rejects_symlink_and_incomplete_campaign(tmp_path):
    evidence(tmp_path)
    (tmp_path / "record.json").rename(tmp_path / "original.json")
    (tmp_path / "record.json").symlink_to(tmp_path / "original.json")
    with pytest.raises(ValueError, match="symbolic|every JSON"):
        verify(tmp_path)
    (tmp_path / "record.json").unlink()
    (tmp_path / "original.json").rename(tmp_path / "record.json")
    (tmp_path / "final").mkdir()
    with pytest.raises(ValueError, match="every declared"):
        run(tmp_path, tmp_path / "output")


def test_evidence_rejects_unlisted_record(tmp_path):
    evidence(tmp_path)
    (tmp_path / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="every JSON"):
        verify(tmp_path)
