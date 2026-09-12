"""Preserve every relocated development outcome and its numerical scope."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1] / "examples/development_studies"
TOOL = ROOT.parents[1] / "tools/reproduce_development_studies.py"
spec = importlib.util.spec_from_file_location("development_reproduction", TOOL)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


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


def test_clean_cli_reproduces_exact_quantities_from_an_unrelated_directory(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "result with spaces"
    call = subprocess.run(
        [sys.executable, str(TOOL), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert call.returncode == 0, call.stdout + call.stderr
    record = json.loads((output / "reproduction.json").read_text())
    for name, checksum in record["output_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
    assert (output / "figures/temporal/resolution.pdf").read_bytes().startswith(b"%PDF-")
    assert (output / "studies.json").read_bytes() == (ROOT / "studies.json").read_bytes()
    assert (output / "revision/temporal_resolution.json").read_bytes() == (
        ROOT / "sources/temporal_resolution.json"
    ).read_bytes()
    before = (output / "reproduction.json").read_bytes()
    again = subprocess.run(
        [sys.executable, str(TOOL), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert again.returncode == 1 and "must be new" in again.stderr
    assert (output / "reproduction.json").read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        "script",
        "record",
        "missing",
        "extra",
        "path",
        "symlink",
        "manifest_symlink",
        "missing_comparison",
        "missing_replay",
        "summary",
    ],
)
def test_invalid_inputs_fail_before_creating_outputs(tmp_path, change):
    archive, temporal = tmp_path / "archive", tmp_path / "temporal"
    shutil.copytree(ROOT, archive)
    shutil.copytree(runner.TEMPORAL, temporal)
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    script = archive / "sources/import_temporal_study.py.txt"
    if change == "script":
        script.write_text("raise RuntimeError('must not execute changed source')")
    elif change == "record":
        (temporal / "original/summary.json").write_text("{}")
    elif change == "missing":
        script.unlink()
    elif change == "extra":
        (temporal / "replays/final/extra.json").write_text("{}")
    elif change == "path":
        manifest["files"]["../escape.py"] = manifest["files"].pop(
            "sources/import_temporal_study.py.txt"
        )
        manifest_path.write_text(json.dumps(manifest))
    elif change == "symlink":
        script.unlink()
        script.symlink_to(ROOT / "sources/import_temporal_study.py.txt")
    elif change == "manifest_symlink":
        manifest_path.unlink()
        manifest_path.symlink_to(ROOT / "manifest.json")
    elif change == "missing_comparison":
        path = archive / "studies.json"
        data = json.loads(path.read_text())
        data["construction_stopping"].pop(0)
        path.write_text(json.dumps(data))
        manifest["files"]["studies.json"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    elif change == "missing_replay":
        (temporal / "replays/final/case-09.json").unlink()
    else:
        (archive / "sources/temporal_resolution.json").write_text("{}")
    output = tmp_path / "output"
    with pytest.raises((ValueError, OSError)):
        runner.reproduce(output, archive, temporal)
    assert not output.exists()
    assert not list(tmp_path.glob(".development-*"))


def test_verify_only_requires_no_plotting_dependency(tmp_path):
    # -S omits site packages entirely. Byte validation uses only the standard library.
    result = subprocess.run(
        [sys.executable, "-S", str(TOOL), "--verify-only"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_failed_render_leaves_no_partial_result(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.SourceFileLoader,
        "exec_module",
        lambda self, module: setattr(
            module, "generate", lambda path: (_ for _ in ()).throw(RuntimeError("render failed"))
        ),
    )
    with pytest.raises(RuntimeError, match="render failed"):
        runner.reproduce(tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".development-*"))
