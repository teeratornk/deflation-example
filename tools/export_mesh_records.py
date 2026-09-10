"""Export mesh records and numerical pilot sources into a versioned evidence tree."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

from check_release import MAX_FILE_BYTES, inspect_content
from export_study_records import export


def run(selections, sources, output):
    prepared, source_names = {}, set()
    for item in sources:
        name, separator, directory = item.partition("=")
        if not separator or not re.fullmatch(r"pilot-v[1-9][0-9]*", name) or name in source_names:
            raise ValueError("Pilot sources require distinct pilot-vN=directory entries")
        source_names.add(name)
        root = Path(directory)
        if not root.is_dir() or root.is_symlink():
            raise ValueError("A numerical source must be a regular directory")
        if not (root / "deflation_example/benchmark_mesh.py").is_file():
            raise ValueError("A pilot source must contain its mesh benchmark implementation")
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix not in {
                ".py",
                ".yaml",
                ".json",
                ".npz",
                ".md",
            }:
                continue
            if (
                path.is_symlink()
                or not path.is_file()
                or any((root / p).is_symlink() for p in relative.parents)
            ):
                raise ValueError("Pilot sources cannot contain symbolic links")
            target = (PurePosixPath("sources") / name / relative.as_posix()).as_posix()
            content = path.read_bytes()
            if len(content) > MAX_FILE_BYTES or inspect_content(target, content):
                raise ValueError("A pilot source failed publication checks")
            prepared[target] = (path, content)
    manifest = export(selections, output)
    output = Path(output)
    license_path = Path(__file__).resolve().parents[1] / "LICENSE"
    license_content = license_path.read_bytes()
    shutil.copyfile(license_path, output / "LICENSE")
    manifest["files"]["LICENSE"] = {
        "sha256": hashlib.sha256(license_content).hexdigest(),
        "bytes": len(license_content),
    }
    for target, (source, content) in prepared.items():
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if destination.read_bytes() != content:
            raise ValueError("A pilot source changed during export")
        manifest["files"][target] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    manifest.update(format="mesh-cht-evidence-v1", numerical_pilot_sources=sorted(source_names))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--pilot-sources", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.records, args.pilot_sources, args.output)
    print(f"Exported {len(result['files'])} checked mesh evidence files")


if __name__ == "__main__":
    main()
