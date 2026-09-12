"""Regenerate relocated tables and figures; retain a log and status for every command."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify_guarded(root=None):
    from reproduce_development_studies import checked_file

    root = Path(root) if root is not None else ROOT / "examples/guarded_refinement"
    path = root / "manifest.json"
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Linked guarded manifest")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "guarded-refinement-evidence-v1":
        raise ValueError("Unknown guarded evidence schema")
    expected = set(manifest["files"])
    observed = {p.relative_to(root).as_posix() for p in (root / "records").rglob("*")
                if p.is_file()}
    if not expected or observed != expected:
        raise ValueError("Guarded record inventory differs from its manifest")
    for name, checksum in manifest["files"].items():
        checked_file(root, name, checksum)
    return manifest


def commands(output, mesh_records=None, cartesian_records=None):
    """Use the same public entry points as the individual reproduction guides."""
    output = Path(output).resolve()
    temporal = ROOT / "examples/temporal_resolution/data"
    tasks = [
        ("supplementary-tables", [str(ROOT / "tools/supplementary_tables.py")]),
        (
            "development",
            [
                str(ROOT / "tools/reproduce_development_studies.py"),
                "--output",
                str(output / "development"),
            ],
        ),
    ]
    for stage in ("initial", "extension", "final"):
        tasks.append(
            (
                f"temporal-{stage}",
                [
                    str(ROOT / "tools/report_temporal_assessment.py"),
                    "--original",
                    str(temporal / "original/summary.json"),
                    "--corrected",
                    str(temporal / "corrected/summary.json"),
                    "--stability",
                    str(temporal / "stability.json"),
                    "--replays",
                    *[str(temporal / f"replays/{stage}/case-{c:02d}.json") for c in (0, 4, 5, 9)],
                    "--output",
                    str(output / f"temporal-{stage}"),
                ],
            )
        )
    tasks.append(
        (
            "guarded-refinement",
            [
                "-m",
                "deflation_example.mesh_report",
                "--studies",
                str(ROOT / "examples/guarded_refinement/records/steady"),
                str(ROOT / "examples/guarded_refinement/records/transient"),
                "--output",
                str(output / "guarded-refinement"),
            ],
        )
    )
    for name, module, records in (
        ("body-fitted", "mesh_evidence", mesh_records),
        ("cartesian", "benchmark_evidence", cartesian_records),
    ):
        if records is not None:
            tasks.append(
                (
                    name,
                    [
                        "-m",
                        f"deflation_example.{module}",
                        "--records",
                        str(Path(records).resolve()),
                        "--output",
                        str(output / name),
                    ],
                )
            )
    return tasks


def run(output, mesh_records=None, cartesian_records=None, timeout=1800):
    output = Path(output).resolve()
    if timeout <= 0:
        raise ValueError("The per-command timeout must be positive")
    if output.exists():
        raise ValueError("Use a new reproduction output directory")
    # Check the byte inventories before any report can load archived inputs.
    from reproduce_development_studies import verify

    verify()
    verify_guarded()
    output.mkdir(parents=True, exist_ok=False)
    tasks = commands(output, mesh_records, cartesian_records)
    report = {
        "schema": "repository-reproduction-v1",
        "scope": "CPU regeneration of archived evidence; no new solver timing measurements.",
        "commands": [],
        "generator_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "tools").glob("*.py"))
        },
        "optional_evidence": {
            "body_fitted": mesh_records is not None,
            "cartesian": cartesian_records is not None,
        },
    }
    try:
        report["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        report["git_head"] = None
    for name, arguments in tasks:
        log = output / f"{name}.log"
        with log.open("w") as stream:
            try:
                result = subprocess.run(
                    [sys.executable, *arguments],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
                status, code = ("passed" if result.returncode == 0 else "failed"), result.returncode
            except subprocess.TimeoutExpired:
                status, code = "timeout", None
        report["commands"].append(
            {
                "name": name,
                "status": status,
                "returncode": code,
                "log": log.name,
                "arguments": arguments,
            }
        )
        print(f"{name}: {status}", flush=True)
        report["success"] = all(row["status"] == "passed" for row in report["commands"])
        (output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    report["output_sha256"] = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p != output / "reproduction.json"
    }
    (output / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mesh-records", type=Path)
    parser.add_argument("--cartesian-records", type=Path)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per command")
    args = parser.parse_args()
    try:
        report = run(**vars(args))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Reproduction failed: {error}\n")
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
