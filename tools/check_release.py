"""Check publishable source and distributions without printing suspected secrets.

This is a release guard, not a guarantee that every possible secret is detected.
Review the file inventory as well. Uses only the Python standard library.
"""

import argparse
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import zipfile


PATTERNS = {
    "private key": r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----",
    "GitHub credential": r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "cloud credential": r"(?:AKIA|ASIA)[A-Z0-9]{16}",
    "credential URL": r"https?://[^\s/@:]+:[^\s/@]+@",
    "user directory": r"/(?:home|Users)/[A-Za-z0-9_.-]+/",
    "cluster directory": r"/(?:scratch|lustre|gpfs)/[A-Za-z0-9_.-]+/",
    "assigned credential": r"(?i)(?:api[_-]?key|password|secret[_-]?key)\s*[=:]\s*[\"'][A-Za-z0-9/+_=.-]{16,}[\"']",
}
FORBIDDEN = {".git", ".env", ".venv", "__pycache__", "runs", ".pytest_cache", ".ruff_cache"}
GENERATED_SOURCE = {"build", "dist", "runs", "__pycache__", ".pytest_cache", ".ruff_cache"}


def inspect_content(name, content):
    """Return rule names only; never return matched text."""
    path = PurePosixPath(name)
    findings = []
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        findings.append("unsafe path")
    if any(p in FORBIDDEN or p.startswith((".venv-", ".env.")) for p in path.parts):
        findings.append("private or generated file")
    if path.suffix.lower() in {".pem", ".key", ".p12", ".pyc", ".out", ".log"}:
        findings.append("unexpected file type")
    text = content.decode("utf-8", errors="replace")
    findings.extend(label for label, pattern in PATTERNS.items() if re.search(pattern, text))
    return findings


def members(path):
    def safe_name(name):
        entry = PurePosixPath(name)
        if entry.is_absolute() or ".." in entry.parts or "\\" in name:
            raise ValueError("Archive contains an unsafe path")

    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                safe_name(item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Archive contains a symlink")
                if not item.is_dir():
                    yield item.filename, archive.read(item)
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            for item in archive.getmembers():
                safe_name(item.name)
                if item.isdir():
                    continue
                if not item.isfile():
                    raise ValueError("Archive contains a link or special file")
                with archive.extractfile(item) as stream:
                    yield item.name, stream.read()
    else:
        raise ValueError("Expected a wheel or source distribution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Git checkout or unpacked source distribution")
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args()
    if args.source is None and not args.archives:
        parser.error("Specify --source and/or distribution archives")
    records = []
    if args.source:
        root = args.source.resolve()
        if not root.is_dir():
            parser.error("Source must be an existing directory")
        if (root / ".git").exists():
            names = (
                subprocess.check_output(
                    ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                    cwd=root,
                )
                .decode()
                .split("\0")
            )
            paths = [root / p for p in sorted(set(names)) if p]
        else:
            # An unpacked sdist has no Git ignore rules. Match the checkout's
            # generated-directory exclusions without hiding credential files.
            paths = []
            for directory, subdirs, files in os.walk(root):
                subdirs[:] = [
                    d for d in subdirs if d not in GENERATED_SOURCE and not d.startswith(".venv")
                ]
                paths.extend(Path(directory) / name for name in files)
                paths.extend(
                    Path(directory) / d for d in subdirs if (Path(directory) / d).is_symlink()
                )
            paths.sort()
        for path in paths:
            if path.is_symlink():
                raise ValueError("Source contains a symlink")
            records.append((path.relative_to(root).as_posix(), path.read_bytes()))
    for archive in args.archives:
        records.extend((f"{archive.name}/{name}", data) for name, data in members(archive))
    failed = False
    for name, content in records:
        for rule in inspect_content(name, content):
            print(f"{name}: {rule}")
            failed = True
    print(f"Checked {len(records)} files; findings: {'yes' if failed else 'none'}")
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
