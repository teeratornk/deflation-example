"""Detailed reader tables preserve all source rows and measurement definitions."""

import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "supplementary_tables", ROOT / "tools/supplementary_tables.py"
)
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)


def test_all_tables_and_rows_are_reproduced():
    manifest = tables.verify()
    assert len(manifest["tables"]) == 11
    assert sum(f["rows"] for t in manifest["tables"] for f in t["files"]) == 317
    assert tables.DOCUMENT.read_text() == tables.render()
    for table in manifest["tables"]:
        for item in table["files"]:
            raw = (tables.DATA / item["path"]).read_text()
            for row in tables.table_rows(raw, len(table["columns"])):
                assert "| " + " | ".join(map(tables.markdown_cell, row)) + " |" in tables.render()


def test_scalar_diagnostics_keep_every_breakdown():
    manifest = tables.verify()
    item = next(t for t in manifest["tables"] if t["id"] == "finer-transformer-diagnostics")
    rows = tables.table_rows((tables.DATA / item["files"][0]["path"]).read_text(), 9)
    assert len(rows) == 7
    assert all(float(row[-2]) < 0 and float(row[-1]) > 0 for row in rows)


@pytest.mark.parametrize("change", ["content", "missing", "extra", "path", "rows"])
def test_modified_or_incomplete_sources_fail(tmp_path, change):
    root = tmp_path / "tables"
    shutil.copytree(tables.DATA, root)
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["tables"][0]["files"][0]
    path = root / entry["path"]
    if change == "content":
        path.write_text(path.read_text().replace("1.26e-11", "9.99e-11", 1))
    elif change == "missing":
        path.unlink()
    elif change == "extra":
        (root / "sources/extra.tex").write_text("extra")
    elif change == "path":
        entry["path"] = "../escape.tex"
    else:
        entry["rows"] -= 1
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        tables.verify(root)


def test_numerical_sources_and_procedures_are_distinct():
    sources = tables.verify()["numerical_sources"]
    assert sources["body_fitted"] == "dc89ae8518ff6a975eadf2f6f3bae1acbd17b83b"
    assert sources["correction"] == "926b37c40935fbbe7134a3ab5f87ebdfddb68837"
    assert sources["guard"] == "0b3b3f06a535f515256932c0dbf27537c242744d"
    assert len(set(sources.values())) == 4
