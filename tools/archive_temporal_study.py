"""Freeze the paired temporal checks, including every unsuccessful attempt."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from deflation_example.reporting import write_report

SOURCES = {
    "original": "91f05ae68f0a58a2b1627bc744b9c97c6ca05bc5",
    "corrected": "e4c11c8b7051e41e958bbb4cffc2e110191d9b81",
}
STAGES = {
    "initial": ("prescribed-replay-v1-case-", [256, 512, 1024, 2048, 4096]),
    "extension": ("prescribed-replay-extension-v1-case-", [4096, 8192, 16384, 32768]),
    "final": ("prescribed-replay-resolution-v2-case-", [16384, 32768, 65536]),
}
SELECTED = [0, 4, 5, 9]


def verify_optimization(record, design, case, source):
    if any(record.get(k) != v for k, v in case.items()):
        raise ValueError("An optimization differs from its preselected case")
    if record.get("status") in ("running", "missing"):
        raise ValueError("The archive requires a terminated attempt for every case")
    if record["protocol_sha256"] != design["protocol_sha256"]:
        raise ValueError("The optimization protocol changed")
    environment = record["environment"]
    # One original attempt could not obtain Git status. Its complete numerical
    # file hashes are still checked against the clean target-selection source.
    if environment["git_head"] != source or environment["source_tree_clean"] not in (True, None):
        raise ValueError("The optimization uses a different numerical source")
    if environment["source_sha256"] != design["environment"]["source_sha256"]:
        raise ValueError("The optimization source differs from target selection")


def verify_replay(record, corrected, case, slabs):
    if record["case"] != case or record["replay_slabs"] != slabs:
        raise ValueError("The replay differs from the declared refinement stage")
    if record["selected_cases"] != SELECTED:
        raise ValueError("The replay selection changed")
    if record["protocol_sha256"] != corrected["protocol_sha256"]:
        raise ValueError("The replay uses a different optimization protocol")
    if record["optimization_source"]["source_sha256"] != corrected["environment"]["source_sha256"]:
        raise ValueError("The replay uses a different optimized source")
    if record["status"] in ("running", "missing"):
        raise ValueError("The replay is incomplete")


def archive(study_root, output):
    from report_temporal_assessment import generate

    study_root, output = Path(study_root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "prescribed-temporal-evidence-v1", "sources": SOURCES, "files": {}}

    def copy(source, name):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest["files"][name] = hashlib.sha256(target.read_bytes()).hexdigest()
        return target

    designs = {}
    for form, source in SOURCES.items():
        stem = "prescribed-temporal-v1" if form == "original" else "prescribed-temporal-skew-v1"
        root = study_root / f"{stem}-data"
        design = json.loads((root / "design.json").read_text())
        designs[form] = design
        if len(design["cases"]) != 10 or design["queries"] != [7, 15]:
            raise ValueError("Keep both preselected targets and all five optimization grids")
        copy(root / "design.json", f"{form}/design.json")
        for index, case in enumerate(design["cases"]):
            path = root / f"case-{index:02d}/record.json"
            record = json.loads(path.read_text())
            verify_optimization(record, design, case, source)
            copy(path, f"{form}/case-{index:02d}/record.json")
        copy(study_root / f"{stem}-summary/summary.json", f"{form}/summary.json")
    copy(study_root / "prescribed-skew-v1-stability/stability.json", "stability.json")
    for stage, (prefix, slabs) in STAGES.items():
        for case in SELECTED:
            path = study_root / f"{prefix}{case}/summary.json"
            replay = json.loads(path.read_text())
            verify_replay(replay, designs["corrected"], case, slabs)
            record = json.loads((output / f"corrected/case-{case:02d}/record.json").read_text())
            if replay["fields_sha256"] != record.get("fields_sha256"):
                raise ValueError("A replay uses a different saved control field")
            copy(path, f"replays/{stage}/case-{case:02d}.json")
    for stage in STAGES:
        generate(
            output / "original/summary.json",
            output / "corrected/summary.json",
            output / "stability.json",
            [output / f"replays/{stage}/case-{c:02d}.json" for c in SELECTED],
            output / f"summaries/{stage}",
            plots=False,
        )
    write_report(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    archive(args.study_root, args.output)
