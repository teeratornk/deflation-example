"""A separate evidence checkout contains only explicitly selected, checked records."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "export_study_records.py"


def call(root, output, name="final/tiny"):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--records", f"{name}={root}", "--output", str(output)],
        capture_output=True,
        text=True,
    )


def test_export_preserves_bytes_and_excludes_unselected_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    content = b'{"status": "converged"}\n'
    (source / "results.json").write_bytes(content)
    (source / "scheduler.log").write_text("Private execution log")
    output = tmp_path / "evidence"
    assert call(source, output).returncode == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert list(manifest["files"]) == ["final/tiny/results.json"]
    assert (
        manifest["files"]["final/tiny/results.json"]["sha256"]
        == hashlib.sha256(content).hexdigest()
    )
    assert (output / "final/tiny/results.json").read_bytes() == content
    assert call(source, output).returncode != 0


def test_export_rejects_private_content_without_echoing_it(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    private = "/" + "home" + "/example/private"
    (source / "results.json").write_text(json.dumps({"path": private}))
    result = call(source, tmp_path / "evidence")
    assert result.returncode != 0
    assert private not in result.stdout + result.stderr
    assert not (tmp_path / "evidence").exists()
    assert call(source, tmp_path / "bad", "final/../escape").returncode != 0
