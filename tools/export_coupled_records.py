"""Export coupled final-study records and presentation artifacts as versioned evidence.

Records (assembled chains, stage records, screen stages, resolution assessments)
are exported through the shared study-record exporter, which checks content and
size. The presentation artifacts (LaTeX rows, figures, summary) are copied with
their checksums into an ``artifacts`` tree, and the evidence manifest names the
frozen source tag and plan checksum so the manuscript importer can verify them.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from export_study_records import export

FORMAT = "coupled-final-evidence-v1"
ARTIFACT_SUFFIXES = {".tex", ".pdf", ".png", ".json"}


def run(selections, artifacts, plan, output):
    manifest = export(selections, output)
    output = Path(output)
    plan = json.loads(Path(plan).read_text())
    if plan.get("frozen_selection") is None:
        raise ValueError("Export the evidence of a frozen plan only")
    copied = {}
    artifacts = Path(artifacts)
    for path in sorted(artifacts.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix not in ARTIFACT_SUFFIXES:
            continue
        relative = path.relative_to(artifacts).as_posix()
        target = output / "artifacts" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        content = target.read_bytes()
        copied["artifacts/" + relative] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    if "artifacts/summary.json" not in copied:
        raise ValueError("The presentation directory must contain summary.json")
    summary = json.loads((artifacts / "summary.json").read_text())
    if summary.get("generator_environment", {}).get("source_tree_clean") is not True:
        raise ValueError("Artifact generation requires a clean source checkout")
    manifest["files"].update(copied)
    manifest.update(
        format=FORMAT,
        source={"tag": plan["source"]["tag"], "sha256": plan["source"]["sha256"]},
        plan_sha256=plan["plan_sha256"],
        protocol_sha256=plan["protocol"]["sha256"],
        frozen_selection=plan["frozen_selection"],
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Exported {len(manifest['files'])} evidence files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="append", default=[], help="category/name=directory")
    parser.add_argument("--artifacts", required=True, help="coupled_presentation output directory")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.select, args.artifacts, args.plan, args.output)
