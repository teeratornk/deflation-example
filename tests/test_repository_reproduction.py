"""The reader workflow covers every relocated group and records unsuccessful commands."""

import importlib.util
import json
from pathlib import Path
import subprocess
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "repository_reproduction", ROOT / "tools/reproduce_repository_examples.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_default_commands_include_every_temporal_stage_and_all_selected_cases(tmp_path):
    tasks = dict(runner.commands(tmp_path))
    assert set(tasks) == {
        "supplementary-tables",
        "development",
        "temporal-initial",
        "temporal-extension",
        "temporal-final",
        "guarded-refinement",
    }
    for stage in ("initial", "extension", "final"):
        argv = tasks[f"temporal-{stage}"]
        assert [
            Path(s).name for s in argv[argv.index("--replays") + 1 : argv.index("--output")]
        ] == ["case-00.json", "case-04.json", "case-05.json", "case-09.json"]
    all_tasks = dict(runner.commands(tmp_path, tmp_path / "mesh", tmp_path / "cartesian"))
    assert set(all_tasks) - set(tasks) == {"body-fitted", "cartesian"}


def test_failed_and_timed_out_commands_remain_visible(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    outcomes = iter((0, 1, "timeout", 0, 0, 0))

    def execute(argv, **kwargs):
        value = next(outcomes)
        if value == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, value)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "test-source\n")
    output = tmp_path / "output"
    report = runner.run(output, timeout=1)
    assert not report["success"]
    assert [r["status"] for r in report["commands"]] == [
        "passed",
        "failed",
        "timeout",
        "passed",
        "passed",
        "passed",
    ]
    assert json.loads((output / "reproduction.json").read_text()) == report
    assert all((output / row["log"]).is_file() for row in report["commands"])
    with pytest.raises(ValueError, match="new reproduction"):
        runner.run(output)


def test_invalid_timeout_creates_no_output(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        runner.run(tmp_path / "output", timeout=0)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("change", ["content", "missing", "extra", "manifest_link"])
def test_guarded_inputs_are_verified_before_reporting(tmp_path, monkeypatch, change):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    root = tmp_path / "guarded"
    shutil.copytree(ROOT / "examples/guarded_refinement", root)
    path = root / "records/steady/reference-0/record.json"
    if change == "content":
        path.write_text("{}")
    elif change == "missing":
        path.unlink()
    elif change == "extra":
        (root / "records/extra.json").write_text("{}")
    else:
        (root / "manifest.json").unlink()
        (root / "manifest.json").symlink_to(ROOT / "examples/guarded_refinement/manifest.json")
    with pytest.raises(ValueError):
        runner.verify_guarded(root)
