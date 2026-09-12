"""Verify relocated evidence and regenerate it without modifying the archive."""

import argparse
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "examples/development_studies"
TEMPORAL = ROOT / "examples/temporal_resolution/data"
SOURCES = {
    "thermal_resolution_support.tex",
    "resolution_summary.tex",
    "pilot_protocol_rows.tex",
    "pilot_outcome_rows.tex",
    "support_finer_pilot_rows.tex",
    "finer_outcomes.tex",
    "import_temporal_study.py.txt",
    "temporal_resolution.json",
}


class FrozenSourceLoader(SourceFileLoader):
    """Compile the checked bytes directly, without reading or writing bytecode caches."""

    def get_code(self, fullname):
        return self.source_to_code(self.get_data(self.path), self.path)


def checked_file(root, name, checksum):
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or relative.as_posix() != name
        or not relative.parts
    ):
        raise ValueError(f"Unsafe archive path: {name}")
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"Missing or linked archive file: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"Archive checksum mismatch: {name}")


def verify(archive=ARCHIVE, temporal=TEMPORAL):
    """Validate manifests before loading the frozen plotting code or creating output."""
    archive, temporal = Path(archive), Path(temporal)
    manifests = []
    for root in (archive, temporal):
        path = root / "manifest.json"
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Linked archive manifest")
        manifest = json.loads(path.read_text())
        if not manifest.get("files"):
            raise ValueError("An archive manifest must enumerate its files")
        for name, entry in manifest["files"].items():
            checked_file(root, name, entry["sha256"] if isinstance(entry, dict) else entry)
        manifests.append(manifest)
    expected = {"studies.json", "temporal_resolution.svg"} | {f"sources/{s}" for s in SOURCES}
    observed = {
        p.relative_to(archive).as_posix()
        for p in (archive / "sources").rglob("*")
        if p.is_file() and "__pycache__" not in p.relative_to(archive).parts
    }
    if set(manifests[0]["files"]) != expected or observed != expected - {
        "studies.json",
        "temporal_resolution.svg",
    }:
        raise ValueError("Development source inventory differs from its manifest")
    observed = {
        p.relative_to(temporal).as_posix()
        for folder in ("original", "corrected", "replays")
        for p in (temporal / folder).rglob("*")
        if p.is_file()
    } | {"stability.json"}
    if observed != set(manifests[1]["files"]):
        raise ValueError("Temporal source inventory differs from its manifest")
    studies = json.loads((archive / "studies.json").read_text())
    if [p["identifier"] for p in studies["construction_stopping"]] != [
        f"P{i}" for i in range(1, 9)
    ] or len(studies["finer_transformer_attempts"]) != 8:
        raise ValueError("Keep all eight construction and eight refinement comparisons")
    return manifests


def reproduce(output, archive=ARCHIVE, temporal=TEMPORAL):
    archive, temporal, output = Path(archive), Path(temporal), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("The reproduction destination must be new")
    verify(archive, temporal)
    loader = FrozenSourceLoader(
        "archived_temporal_figure", str(archive / "sources/import_temporal_study.py.txt")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".development-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        (stage / "revision").mkdir(parents=True)
        module.ROOT = stage
        data = module.generate(temporal)
        if data != json.loads((archive / "sources/temporal_resolution.json").read_text()):
            raise ValueError("Regenerated quantities differ from the archived numerical summary")
        text = stage / "sections/generated/temporal/resolution_summary.tex"
        if text.read_bytes() != (archive / "sources/resolution_summary.tex").read_bytes():
            raise ValueError("Regenerated text differs from the archived summary")
        shutil.copyfile(archive / "studies.json", stage / "studies.json")
        for name in (
            "pilot_protocol_rows.tex",
            "pilot_outcome_rows.tex",
            "support_finer_pilot_rows.tex",
            "finer_outcomes.tex",
        ):
            shutil.copyfile(archive / "sources" / name, stage / name)
        hashes = {
            p.relative_to(stage).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        record = {
            "schema": "development-reproduction-v1",
            "scope": "Regeneration from frozen records; no new optimization or timing study.",
            "input_manifest_sha256": {
                key: hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
                for key, root in (("development", archive), ("temporal", temporal))
            },
            "output_sha256": hashes,
        }
        (stage / "reproduction.json").write_text(json.dumps(record, indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise ValueError("The reproduction destination must remain new")
        stage.rename(output)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify-only", action="store_true")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--temporal-data", type=Path, default=TEMPORAL)
    args = parser.parse_args()
    try:
        if args.verify_only:
            verify(args.archive, args.temporal_data)
            print("Verified all relocated development sources and temporal records.")
        else:
            record = reproduce(args.output, args.archive, args.temporal_data)
            print(f"Regenerated {len(record['output_sha256'])} files in {args.output}.")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Reproduction failed: {error}\n")


if __name__ == "__main__":
    main()
