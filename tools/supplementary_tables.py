"""Verify and render the detailed supplementary tables without a GPU or network."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/supplementary_tables"
DOCUMENT = ROOT / "docs/supplementary-tables.md"


def table_rows(text, columns):
    rows = []
    for row in text.split(r"\\"):
        if not row.strip():
            continue
        cells = [cell.strip() for cell in row.replace(r"\midrule", "").split("&")]
        if len(cells) != columns:
            raise ValueError("Table column count differs from its definition")
        rows.append(cells)
    return rows


def verify(folder=DATA):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["format"] != "supplementary-tables-v1":
        raise ValueError("Unknown supplementary-table format")
    expected = set()
    ids = set()
    for table in manifest["tables"]:
        if table["id"] in ids:
            raise ValueError("Duplicate table identifier")
        ids.add(table["id"])
        for item in table["files"]:
            name = PurePosixPath(item["path"])
            if name.is_absolute() or ".." in name.parts or name.parts[0] != "sources":
                raise ValueError("Unsafe table path")
            path = folder / name
            if path.is_symlink() or path.parent.is_symlink() or item["path"] in expected:
                raise ValueError("Linked or duplicate source table")
            expected.add(item["path"])
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Source table checksum mismatch")
            if len(table_rows(raw.decode(), len(table["columns"]))) != item["rows"]:
                raise ValueError("Source table row count mismatch")
    actual = {
        p.relative_to(folder).as_posix() for p in (folder / "sources").rglob("*") if p.is_file()
    }
    if actual != expected:
        raise ValueError("Supplementary source inventory differs from the manifest")
    return manifest


def markdown_cell(cell):
    # Keep numerical text unchanged. Join adjacent TeX math spans for GitHub.
    return cell.replace("$$", "").replace("|", r"\|")


def render(folder=DATA):
    folder = Path(folder)
    manifest = verify(folder)
    lines = [
        "# Detailed supplementary tables",
        "",
        "These tables support the prescribed-flow CHT optimization study. They give",
        "per-case accuracy and ranks, cost components, preparation-inclusive times,",
        "sampled memory, and the finer-transformer scalar diagnostics. Every row and",
        "compared method from these table groups is included. The manuscript and SI",
        "retain the principal complete comparisons, transfer tests, verification,",
        "stopping sensitivity, and conditional-bound example.",
        "",
        "The [body-fitted reproduction examples](mesh-showcases.md#choose-the-numerical-procedure)",
        "distinguish the primary projected recurrence, measured residual-correction",
        "study, and subsequent guarded solver. The guard is separate from the GPU timings.",
        "The [Cartesian examples](reference-policy-study.md#frozen-complete-optimization-campaign)",
        "provide the refinement and rank comparisons.",
        "",
        "## Read or regenerate the tables",
        "",
        "Read the tables below directly on GitHub. From a checkout of this revision:",
        "",
        "```bash",
        "uv run --locked python tools/supplementary_tables.py",
        "uv run --locked python tools/supplementary_tables.py --write",
        "```",
        "",
        "The first command verifies the source checksums, row counts and displayed tables.",
        "The second regenerates this page. Both use the bundled data and Python's standard",
        "library; table verification itself needs no accelerator, optional libraries,",
        "or network. `python3 tools/supplementary_tables.py` also works without installation.",
        "Each section links to commands for regenerating its numerical evidence.",
        "",
        "## Table index",
        "",
    ]
    lines.extend(f"- [{table['title']}](#{table['id']})" for table in manifest["tables"])
    lines.extend(
        [
            "",
            "## Data and measurement definitions",
            "",
            "The [manifest](../examples/supplementary_tables/manifest.json) identifies the",
            "numerical implementations, data versions and exact source-table checksums.",
            "The row files come from manuscript revision",
            f"[`{manifest['manuscript_source'][:7]}`]({manifest['repository']}/tree/{manifest['manuscript_source']}).",
            "Earlier SI table numbers in the manifest identify that fixed manuscript revision.",
            "They do not change when the current SI is renumbered.",
            "",
            "Bore 1 and Bore 2 denote the two bore-in-block mesh levels. Body-fitted",
            "comparisons use 16 targets and five complete repetitions unless the section",
            "specifies the separate four-target diagnostic study. Unmarked body-fitted",
            "reference ranks are 100. Warm starts apply to both PDAS and inner solves.",
            "All completed optimization sequences meet the independent original-residual",
            "and KKT criteria of 1e-10 and 1e-8. Diagnostic breakdowns are identified explicitly.",
            "Repeated timings cover the same target sequence. They do not represent additional",
            "distinct physical problems. Sampled process peaks include all numerical backends.",
            "",
        ]
    )
    for table in manifest["tables"]:
        lines.extend(
            [
                f"## {table['title']}",
                "",
                table["description"],
                "",
                f"[Reproduction example]({table['guide']}).",
                "",
                "| " + " | ".join(table["columns"]) + " |",
                "| " + " | ".join(["---"] * len(table["columns"])) + " |",
            ]
        )
        for item in table["files"]:
            for row in table_rows((folder / item["path"]).read_text(), len(table["columns"])):
                lines.append("| " + " | ".join(map(markdown_cell, row)) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate the GitHub table page")
    args = parser.parse_args()
    expected = render()
    if args.write:
        DOCUMENT.write_text(expected)
    elif not DOCUMENT.is_file() or DOCUMENT.read_text() != expected:
        raise SystemExit("Displayed tables differ from the source: use --write to regenerate")
    manifest = verify()
    count = sum(item["rows"] for table in manifest["tables"] for item in table["files"])
    print(f"Verified {len(manifest['tables'])} tables and all {count} rows.")


if __name__ == "__main__":
    main()
