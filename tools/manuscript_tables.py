"""Verify and regenerate the complete table archive using only the standard library."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/manuscript_tables"
DOCUMENT = ROOT / "docs/manuscript-tables.md"


def split_tex(text, delimiter):
    """Split only outside braced TeX arguments (e.g. stacked memory cells)."""
    parts, start, depth, i = [], 0, 0, 0
    while i < len(text):
        j = i - 1
        while j >= 0 and text[j] == "\\":
            j -= 1
        escaped = (i - j - 1) % 2 == 1
        if text[i] == "{" and not escaped:
            depth += 1
        elif text[i] == "}" and not escaped:
            depth -= 1
        if depth == 0 and text.startswith(delimiter, i):
            parts.append(text[start:i].strip())
            i += len(delimiter)
            if delimiter == r"\\":
                spacing = re.match(r"\[[^\]]*\]", text[i:])
                if spacing:
                    i += len(spacing[0])
            start = i
            continue
        i += 1
    if text[start:].strip():
        parts.append(text[start:].strip())
    if depth:
        raise ValueError("Unbalanced table arguments")
    return parts


def cells(text):
    result = []
    for cell in split_tex(text, "&"):
        span = re.fullmatch(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}", cell, re.S)
        if span:
            result.extend([span[2]] + [""] * (int(span[1]) - 1))
        else:
            result.append(cell)
    return result


def rows(text):
    return [cells(row) for row in split_tex(text.replace(r"\midrule", ""), r"\\")]


def markdown(text, references):
    text = re.sub(r"\\begin\{tabular\}(?:\[[^]]*\])?\{[^\n]*?\}(.*?)\\end\{tabular\}",
                  lambda m: m[1].replace(r"\\", " "), text, flags=re.S)
    text = re.sub(r"\\(?:Mref|Sref|ref|eqref)\{([^}]+)\}",
                  lambda m: references[m[1]], text)
    text = re.sub(r"\\shortstack\{(.*)\}", lambda m: m[1].replace(r"\\", "<br>"), text, flags=re.S)
    text = text.replace("{[", "[").replace("]}", "]").replace("{,}", ",")
    text = text.replace("$$", "").replace(r"\%", "%").replace("~", " ")
    text = text.replace("--", "–").replace(r"\,", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", r"\|")


def verify(folder=DATA):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["format"] != "manuscript-tables-v1":
        raise ValueError("Unknown archive format")
    expected = set()
    for entry in manifest["files"]:
        name = PurePosixPath(entry["path"])
        if name.is_absolute() or ".." in name.parts or name.parts[0] != "sources":
            raise ValueError("Unsafe archive path")
        path = folder / name
        if path.is_symlink() or path.parent.is_symlink() or entry["path"] in expected:
            raise ValueError("Duplicate or linked archive source")
        expected.add(entry["path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Changed source: {entry['path']}")
    actual = {p.relative_to(folder).as_posix() for p in (folder / "sources").rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Archive file inventory differs from the manifest")
    labels = set()
    for table in manifest["tables"]:
        if table["label"] in labels:
            raise ValueError("Duplicate table label")
        labels.add(table["label"])
        if table["source"] not in expected or table["rows_source"] not in expected:
            raise ValueError("Undeclared table source")
        data = rows((folder / table["rows_source"]).read_text())
        if len(data) != table["rows"] or any(len(row) != len(table["columns"]) for row in data):
            raise ValueError(f"Row or column count differs: {table['number']}")
    return manifest


def render(folder=DATA):
    folder = Path(folder)
    manifest = verify(folder)
    lines = (folder / "introduction.txt").read_text().rstrip().splitlines()
    lines += ["", "## Table index", "", "| Original table | Subject |", "| --- | --- |"]
    for table in manifest["tables"]:
        lines.append(f"| [{table['number']}](#table-{table['number'].lower()}) | {table['title']} |")
    for table in manifest["tables"]:
        lines += ["", f'<a id="table-{table["number"].lower()}"></a>', "",
                  f"## Table {table['number']}: {table['title']}", "",
                  markdown(table["caption"], manifest["references"]), "",
                  f"[Numerical reproduction guide]({table['guide']}). "
                  f"[Original table source](../examples/manuscript_tables/{table['source']}).", "",
                  "| " + " | ".join(markdown(c, manifest["references"]) for c in table["columns"]) + " |",
                  "| " + " | ".join("---" for _ in table["columns"]) + " |"]
        for row in rows((folder / table["rows_source"]).read_text()):
            lines.append("| " + " | ".join(markdown(c, manifest["references"]) for c in row) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = render()
    if args.write:
        DOCUMENT.write_text(output)
    elif not DOCUMENT.is_file() or DOCUMENT.read_text() != output:
        raise SystemExit("Displayed archive differs; run with --write")
    manifest = verify()
    print(f"Verified {len(manifest['tables'])} tables and {sum(t['rows'] for t in manifest['tables'])} rows.")


if __name__ == "__main__":
    main()
