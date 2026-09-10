"""Export explicitly selected JSON records into a separately versioned data checkout."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

from check_release import MAX_FILE_BYTES, inspect_content


def export(selections, output):
    output = Path(output)
    if output.exists():
        raise ValueError("The evidence destination must be new")
    files, roots = {}, {}
    for selection in selections:
        name, separator, source = selection.partition("=")
        if (
            not separator
            or not re.fullmatch(r"(final|pilot|diagnostic|summary)/[a-z0-9][a-z0-9-]*", name)
            or name in roots
        ):
            raise ValueError("Each selection requires a distinct category/name=directory")
        root = Path(source)
        if not root.is_dir() or root.is_symlink():
            raise ValueError(
                "Every selected record directory must exist and be a regular directory"
            )
        paths = sorted(root.rglob("*.json"))
        if not paths:
            raise ValueError("A selected study contains no JSON records")
        for path in paths:
            relative = path.relative_to(root)
            if path.is_symlink() or any((root / p).is_symlink() for p in relative.parents):
                raise ValueError("Evidence cannot contain symbolic links")
            target = (PurePosixPath(name) / relative.as_posix()).as_posix()
            content = path.read_bytes()
            if len(content) > MAX_FILE_BYTES or inspect_content(target, content):
                raise ValueError("Evidence failed the publication content or size checks")
            json.loads(content)
            files[target] = (path, content)
        roots[name] = {
            "files": len(paths),
            "scope": name.split("/")[0],
        }
    if not files:
        raise ValueError("Select at least one record directory")
    output.mkdir(parents=True)
    manifest = {"format": "reference-policy-evidence-v1", "studies": roots, "files": {}}
    for target, (source, content) in files.items():
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if destination.read_bytes() != content:
            raise ValueError("A source record changed during export")
        manifest["files"][target] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = export(args.records, args.output)
    print(f"Exported {len(result['files'])} checked records")


if __name__ == "__main__":
    main()
