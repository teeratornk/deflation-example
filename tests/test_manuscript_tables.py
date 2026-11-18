"""Relocated reader tables retain adverse outcomes and exact source inventories."""

import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("manuscript_tables", ROOT / "tools/manuscript_tables.py")
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)


def test_full_archive_regenerates_with_all_original_tables():
    manifest = tables.verify()
    assert [t["number"] for t in manifest["tables"]] == [str(n) for n in range(1, 7)] + [f"S{n}" for n in range(1, 20)]
    assert sum(t["rows"] for t in manifest["tables"]) == 214
    assert tables.DOCUMENT.read_text() == tables.render()


def test_failed_outcomes_and_stacked_memory_cells_survive():
    rendered = tables.render()
    assert "no method converges at every query" in rendered
    assert "RefDef 0/1" in rendered
    assert "0.77" in rendered and "0.78" in rendered
    assert "59.87" in rendered and "36.95" in rendered
    assert "142.336<br>[136.253, 143.655]<br>4.760" in rendered
    assert "162.978<br>[157.208, 174.287]<br>1.893" in rendered


@pytest.mark.parametrize("change", ["content", "missing", "extra", "path", "rows"])
def test_incomplete_or_modified_sources_fail(tmp_path, change):
    folder = tmp_path / "archive"
    shutil.copytree(tables.DATA, folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    entry = manifest["files"][0]
    path = folder / entry["path"]
    if change == "content":
        path.write_text(path.read_text() + "changed")
    elif change == "missing":
        path.unlink()
    elif change == "extra":
        (folder / "sources/extra.tex").write_text("unlisted")
    elif change == "path":
        entry["path"] = "../escape.tex"
    else:
        manifest["tables"][0]["rows"] -= 1
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        tables.verify(folder)
