"""Verify a frozen evidence checkout and regenerate the CHT submission summaries."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


PILOTS = (
    "steady48-spatial-pilot-v1",
    "steady48-all-active-pilot-v2",
    "steady48-refresh-pilot-v2",
    "steady48-stopping-pilot-v2",
    "transient24-scale-pilot-v2",
    "transient24-cached-pilot-v3",
    "mode-dependent-construction-pilot-v2",
    "tensor-construction-pilot-v2",
)
TRANSFERS = (
    "steady-transfer-small-final-v1",
    "transient-transfer-small-final-v1",
    "steady-transfer-moderate-final-v1",
    "transient-transfer-moderate-final-v1",
    "transient-transfer-time4-final-v1",
)


def verify(root):
    """Check every declared byte hash before any records are interpreted."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "reference-policy-evidence-v1" or not manifest.get("files"):
        raise ValueError("Unsupported or empty evidence manifest")
    observed = {path.relative_to(root).as_posix() for path in root.rglob("*.json")}
    if observed - {"manifest.json"} != set(manifest["files"]):
        raise ValueError("Evidence manifest must cover every JSON record")
    for name, record in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Unsafe evidence path")
        path = root / name
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Evidence files must exist without symbolic links")
        content = path.read_bytes()
        if (
            len(content) != record["bytes"]
            or hashlib.sha256(content).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Evidence hash or size mismatch: {name}")
    return manifest


def run(records, output=None):
    root = Path(records)
    manifest = verify(root)
    if output is None:
        return manifest
    from . import benchmark_cht_presentation, benchmark_cht_support, benchmark_transfer_report
    from .benchmark_cht_campaign import campaign_specification

    destination = Path(output)
    if destination.exists():
        raise ValueError("The regeneration destination must be new")
    final = sorted(path for path in (root / "final").iterdir() if path.is_dir())
    declared = set(campaign_specification()["studies"])
    if {path.name for path in final} != declared:
        raise ValueError("The frozen campaign requires every declared final population")
    destination.mkdir(parents=True)
    benchmark_cht_presentation.run(
        final,
        destination / "presentation",
        ["steady32", "steady48", "time4", "transient24"],
        ["steady32", "transient24"],
        ["steady48", "time4"],
    )
    benchmark_transfer_report.run(
        [root / "diagnostic" / name for name in TRANSFERS],
        destination / "transfer",
        ["steady-transfer-moderate-final-v1", "transient-transfer-time4-final-v1"],
    )
    benchmark_cht_support.run(
        root / "diagnostic/independent-validation-final-v1",
        root / "diagnostic/cht-conditioning-final-v1",
        destination / "support",
        [root / "pilot" / name for name in PILOTS],
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument(
        "--output", help="Regenerate submission figures and tables after verification"
    )
    args = parser.parse_args()
    result = run(**vars(args))
    print(f"Verified {len(result['files'])} frozen records")


if __name__ == "__main__":
    main()
