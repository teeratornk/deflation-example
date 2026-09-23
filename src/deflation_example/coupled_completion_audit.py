"""Rebuild evidence locations and preserve externally terminated study outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_retention_report import summarize
from .coupled_trace import read_manifest
from .reporting import file_sha256, write_report


def read(path):
    return json.loads(Path(path).read_text())


def retention_audit(root, campaign):
    """Identify summaries by the evidence partition, not the dispatch-stage name."""
    results = []
    mapping = (
        ("widths-summary", ["selection"], "selection"),
        ("heldout-summary", ["selection", "widths"], "selection"),
        ("assessment-summary", ["heldout"], "held_out"),
    )
    declared = read(campaign / "selection-jobs.json")
    trace = root / f"coupled-retention-capture-v8-{declared['capture']}" / "linear-systems"
    manifest = read_manifest(trace)
    digest = file_sha256(trace / "manifest.json")
    for old_directory, stages, partition in mapping:
        records, sources = [], []
        for stage in stages:
            for job in read(campaign / (stage + "-jobs.json"))["jobs"]:
                if job["role"] != "replay":
                    continue
                path = root / f"coupled-retention-replay-v8-{job['job']}" / "record.json"
                record = read(path)
                if record["trace_sha256"] != digest:
                    raise ValueError("A replay refers to a different captured trajectory")
                records.append(record)
                sources.append(
                    {"file": path.parent.name + "/record.json", "sha256": file_sha256(path)}
                )
        actual = summarize(manifest, records, partition)
        old_path = campaign / old_directory / "summary.json"
        old = read(old_path)
        # Compare the numerical entries retained by both report versions.
        values = (
            "policy",
            "rank",
            "width",
            "eligible",
            "median_replay_total_seconds",
            "total_iterations",
        )
        before = [{k: r[k] for k in values} for r in old["rows"]]
        after = [{k: r[k] for k in values} for r in actual["rows"]]
        matches = old["partition"] == partition and before == after
        results.append(
            {
                "partition": partition,
                "dispatch_stage_directory": old_directory,
                "source_summary_sha256": file_sha256(old_path),
                "matches_saved_summary": matches,
                "records": sources,
                "recomputed": actual,
            }
        )
    return {
        "schema": "coupled-retention-evidence-audit-v1",
        "all_summaries_reproduced": all(r["matches_saved_summary"] for r in results),
        "summaries": results,
        "scope": "Dispatch stages and evidence partitions differ. Previously examined replays are development evidence; fresh targets are required for confirmation.",
    }


def forward_archive(root, closure):
    """Keep missing/cancelled attempts visible without inventing worker records."""
    outcomes = []
    for job in closure["jobs"]:
        item = {**job, "complete_verified": False, "complete_seconds": None}
        if "folder" not in job:
            outcomes.append(item)
            continue
        path = root / job["folder"] / "record.json"
        if not path.is_file():
            item["record_status"] = "missing"
            outcomes.append(item)
            continue
        record = read(path)
        steps = record.get("steps", [])
        audit = record.get("independent_audit", {})
        equations = audit.get("maximum_recomputed_equations", {})
        names = (
            "momentum_relative_residual",
            "continuity_relative_residual",
            "thermal_relative_residual",
            "mass_relative_imbalance",
            "energy_relative_defect",
        )
        values = np.asarray([equations.get(k, np.nan) for k in names], dtype=float)
        passed = bool(
            job["status"] == "completed_verified"
            and record.get("status") == "converged"
            and record.get("complete_trajectory_verified")
            and audit.get("complete_trajectory_verified")
            and len(steps) == record.get("forward_slabs")
            and len(steps) > 0
            and np.isfinite(values).all()
            and np.all(values >= 0)
            and np.all(values[:3] <= 1e-12)
            and np.all(values[3:] <= 1e-6)
        )
        seconds = record.get("seconds")
        passed = (
            passed and isinstance(seconds, (float, int)) and np.isfinite(seconds) and seconds > 0
        )
        item.update(
            record_sha256=file_sha256(path),
            record_status=record.get("status"),
            completed_steps=len(steps),
            declared_steps=record.get("forward_slabs"),
            saved_step_seconds=sum(r["seconds"] for r in steps),
            complete_verified=bool(passed),
            complete_seconds=seconds if passed else None,
        )
        outcomes.append(item)
    return {
        "schema": "coupled-forward-closure-summary-v1",
        "outcomes": outcomes,
        "complete_pimple_speedup": None,
        "scope": "Incomplete costs describe attempts, not completed solutions. This forward assessment establishes no optimization speedup or temporal-resolution claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("retention", "forward"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = (
        retention_audit(args.run_root, args.input)
        if args.mode == "retention"
        else forward_archive(args.run_root, read(args.input))
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
