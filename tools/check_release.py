"""Check publishable source and distributions without printing suspected secrets.

This is a release guard, not a guarantee that every possible secret is detected.
Review the file inventory as well. Uses only the Python standard library.
"""

import argparse
import ast
import io
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import subprocess
import tarfile
import zipfile


PATTERNS = {
    "private key": r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----",
    "GitHub credential": r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "cloud credential": r"(?:AKIA|ASIA)[A-Z0-9]{16}",
    "service credential": r"(?:\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|\bhf_[A-Za-z0-9]{24,}|\bglpat-[A-Za-z0-9_-]{20,}|\bxox[baprs]-[A-Za-z0-9-]{20,})",
    "credential URL": r"https?://[^\s/@:]+:[^\s/@]+@",
    "user directory": r"/(?:home|Users)/[A-Za-z0-9_.-]+/",
    "cluster directory": r"/(?:scratch|lustre|gpfs)/[A-Za-z0-9_.-]+/",
    "Windows user directory": r"[A-Za-z]:[/\\]+Users[/\\]+[^/\\\s]+",
    "private network address": r"(?i)(?:(?:https?|ssh)://|(?:host(?:name)?|server|address)\s*[\"']?\s*[:=]\s*[\"']?)(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b",
    "assigned credential": r"(?i)(?:api[_-]?key|password|(?:access|auth)[_-]?token|secret(?:[_-]?access)?[_-]?key)[\"']?\s*[=:]\s*[\"'][A-Za-z0-9/+_=.-]{16,}[\"']",
}
FORBIDDEN = {
    ".git",
    ".env",
    ".venv",
    "__pycache__",
    "runs",
    ".pytest_cache",
    ".ruff_cache",
    ".netrc",
    ".pypirc",
    ".npmrc",
    "id_rsa",
    "id_ed25519",
}
GENERATED_SOURCE = {"build", "dist", "runs", "__pycache__", ".pytest_cache", ".ruff_cache"}
MAX_FILE_BYTES = 5 * 1024 * 1024
# The source archive includes every repetition from three CHT protocols.
# The installed wheel excludes these records; individual-file limits still apply.
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 2000


def inspect_mesh_bundle(name, content):
    """Allow only the three reviewed numeric mesh bundles, without loading arrays."""
    suffix = str(PurePosixPath(name))
    expected = None
    if suffix.endswith(
        (
            "deflation_example/data/engine_3d/mesh.npz",
            "deflation_example/data/transformer_2d/mesh.npz",
        )
    ):
        expected = {"nodes", "cells", "materials", "dirichlet", "axisymmetric"}
    elif suffix.endswith("deflation_example/data/transformer_2d/inputs.npz"):
        expected = {"velocity_P2_m_s", "source_W_m3", "boundary_outlet", "boundary_wall"}
    if expected is None:
        return ["unexpected binary content"]
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as bundle:
            infos = bundle.infolist()
            if len(infos) != len(expected) or {i.filename for i in infos} != {
                k + ".npy" for k in expected
            }:
                return ["unexpected mesh arrays"]
            for info in infos:
                if info.file_size > MAX_FILE_BYTES:
                    return ["oversized mesh array"]
                array = bundle.read(info)
                if array[:6] != b"\x93NUMPY" or array[6:8] not in {b"\x01\x00", b"\x02\x00"}:
                    return ["invalid numeric mesh array"]
                width = 2 if array[6] == 1 else 4
                length = int.from_bytes(array[8 : 8 + width], "little")
                start = 8 + width
                header = ast.literal_eval(array[start : start + length].decode("ascii"))
                if set(header) != {"descr", "fortran_order", "shape"}:
                    return ["invalid numeric mesh header"]
                dtype, shape = header["descr"], header["shape"]
                if not isinstance(dtype, str) or not re.fullmatch(r"[<>=|][biuf][1248]", dtype):
                    return ["nonnumeric mesh array"]
                if (
                    not isinstance(shape, tuple)
                    or len(shape) > 4
                    or any(type(n) is not int or n < 0 for n in shape)
                ):
                    return ["invalid mesh array dimensions"]
                expected_size = math.prod(shape) * int(dtype[-1])
                if len(array) - start - length != expected_size:
                    return ["invalid mesh array length"]
    except (ValueError, SyntaxError, UnicodeError, KeyError, TypeError, zipfile.BadZipFile):
        return ["invalid mesh archive"]
    return []


def inspect_content(name, content):
    """Return rule names only; never return matched text."""
    path = PurePosixPath(name)
    findings = []
    if path.is_absolute() or PureWindowsPath(name).drive or ".." in path.parts or "\\" in name:
        findings.append("unsafe path")
    if any(p in FORBIDDEN or p.startswith((".venv", ".env.")) for p in path.parts):
        findings.append("private or generated file")
    if path.suffix.lower() in {".pem", ".key", ".p12", ".pyc", ".out", ".log", ".sbatch", ".slurm"}:
        findings.append("unexpected file type")
    if path.suffix.lower() == ".npz":
        return findings + inspect_mesh_bundle(name, content)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return findings + ["unexpected binary content"]
    if "\0" in text:
        findings.append("unexpected binary content")
    findings.extend(label for label, pattern in PATTERNS.items() if re.search(pattern, text))
    return findings


def members(path):
    seen, total = set(), 0

    def safe_entry(name, size):
        nonlocal total
        entry = PurePosixPath(name)
        if (
            entry.is_absolute()
            or PureWindowsPath(name).drive
            or ".." in entry.parts
            or "\\" in name
        ):
            raise ValueError("Archive contains an unsafe path")
        if entry in seen:
            raise ValueError("Archive contains duplicate paths")
        seen.add(entry)
        total += size
        if size > MAX_FILE_BYTES or total > MAX_ARCHIVE_BYTES or len(seen) > MAX_MEMBERS:
            raise ValueError("Archive exceeds this package's review size limits")

    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                safe_entry(item.filename, item.file_size)
                if stat.S_IFMT(item.external_attr >> 16) not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise ValueError("Archive contains a link or special file")
                if not item.is_dir():
                    yield item.filename, archive.read(item)
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            for item in archive.getmembers():
                safe_entry(item.name, item.size)
                if item.isdir():
                    continue
                if not item.isfile():
                    raise ValueError("Archive contains a link or special file")
                with archive.extractfile(item) as stream:
                    yield item.name, stream.read()
    else:
        raise ValueError("Expected a wheel or source distribution")


def history_records(root):
    """Inspect unique blobs reachable from all local refs; never execute old code."""
    if (
        subprocess.check_output(
            ["git", "rev-parse", "--is-shallow-repository"], cwd=root, text=True
        ).strip()
        != "false"
    ):
        raise ValueError("History audit requires a full clone, not a shallow checkout")
    objects = subprocess.check_output(
        ["git", "rev-list", "--objects", "--all"], cwd=root, text=True
    ).splitlines()
    for record in objects:
        oid, _, name = record.partition(" ")
        if not name:
            continue
        info = subprocess.check_output(
            ["git", "cat-file", "--batch-check"], input=oid + "\n", cwd=root, text=True
        ).split()
        if info[1] != "blob":
            continue
        if int(info[2]) > MAX_FILE_BYTES:
            raise ValueError("Historical blob exceeds the review size limit")
        data = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=root)
        yield f"history/{oid[:12]}/{name}", data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Git checkout or unpacked source distribution")
    parser.add_argument(
        "--history",
        action="store_true",
        help="Also scan reachable Git history (full clone required)",
    )
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args()
    if args.source is None and not args.archives:
        parser.error("Specify --source and/or distribution archives")
    if args.history and (args.source is None or not (args.source / ".git").exists()):
        parser.error("History audit requires --source pointing to a Git checkout")
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
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Source file exceeds the review size limit")
            records.append((path.relative_to(root).as_posix(), path.read_bytes()))
        if args.history:
            records.extend(history_records(root))
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
