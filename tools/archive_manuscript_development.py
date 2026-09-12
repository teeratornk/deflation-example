"""Preserve development studies relocated from the paper, without selecting outcomes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def archive(manuscript, output):
    manuscript, output = Path(manuscript), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "sources"
    sources.mkdir()
    manifest = {
        "manuscript_source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=manuscript, text=True
        ).strip(),
        "scope": "All construction/stopping pilots, original finer-transformer attempts, and paired temporal-resolution pilot; separate from final comparisons.",
        "files": {},
    }
    files = [
        "sections/thermal_resolution_support.tex",
        "sections/generated/temporal/resolution_summary.tex",
        "sections/generated/cht/pilot_protocol_rows.tex",
        "sections/generated/cht/pilot_outcome_rows.tex",
        "sections/generated/mesh/support_finer_pilot_rows.tex",
        "sections/generated/si/finer_outcomes.tex",
        "tools/import_temporal_study.py",
        "revision/temporal_resolution.json",
    ]
    for relative in files:
        source = manuscript / relative
        target = sources / (source.name + ".txt" if source.suffix == ".py" else source.name)
        shutil.copyfile(source, target)
        manifest["files"][target.relative_to(output).as_posix()] = {
            "manuscript_path": relative,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    cartesian = json.loads((manuscript / "revision/reference_policy_artifacts.json").read_text())
    mesh = json.loads((manuscript / "revision/mesh_artifacts.json").read_text())
    data = {
        "construction_stopping": cartesian["summaries"]["support"]["pilots"],
        "finer_transformer_attempts": mesh["summaries"]["support"]["finer_pilots"],
    }
    (output / "studies.json").write_text(json.dumps(data, indent=2) + "\n")
    manifest["files"]["studies.json"] = {
        "sha256": hashlib.sha256((output / "studies.json").read_bytes()).hexdigest()
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    archive(args.manuscript, args.output)
