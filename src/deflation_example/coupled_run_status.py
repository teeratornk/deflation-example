"""Report scheduler termination separately from unfinished numerical records."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .reporting import write_report


TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
ACTIVE_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
    "SIGNALING",
    "STAGE_OUT",
    "STOPPED",
}


def parse_object(content):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    result = json.loads(content, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def contained_record(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Record paths must be nonempty relative paths")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Record paths must remain within the manifest directory")
    return path


def scheduler_outcome(state, started):
    if state == "CANCELLED" and started is False:
        return "cancelled_before_start"
    if state == "COMPLETED":
        return "process_completed"
    return "scheduler_" + state.lower()


def summarize(manifest, root):
    """Retain every declared run without inferring accuracy or speed from job status."""
    if manifest.get("schema") != "coupled-run-status-manifest-v1":
        raise ValueError("Use a coupled-run-status-manifest-v1 manifest")
    entries = manifest.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Declare every expected run in a nonempty list")
    rows, labels, paths = [], set(), set()
    for entry in entries:
        label = entry["label"]
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError("Run labels must be nonempty and unique")
        labels.add(label)
        scheduler = entry["scheduler"]
        state = scheduler["state"]
        elapsed = scheduler["elapsed_seconds"]
        started = scheduler.get("started")
        if state not in TERMINAL_STATES | ACTIVE_STATES:
            raise ValueError("Use an exact scheduler state without truncation or annotations")
        if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
            raise ValueError("Scheduler elapsed seconds must be a nonnegative integer")
        if started is not None and not isinstance(started, bool):
            raise ValueError("Scheduler started must be a Boolean or null")
        if started is False and elapsed > 0:
            raise ValueError("An unstarted run cannot have positive elapsed time")
        path = contained_record(Path(root), entry["record"])
        if path in paths:
            raise ValueError("Each run must identify a different record")
        paths.add(path)
        row = {
            "label": label,
            "record": entry["record"],
            "scheduler": scheduler,
            "terminal": state in TERMINAL_STATES,
            "outcome": scheduler_outcome(state, started),
            "record_status": None,
            "record_sha256": None,
            "record_error": None,
        }
        try:
            content = path.read_bytes()
            row["record_sha256"] = hashlib.sha256(content).hexdigest()
            record = parse_object(content)
            if not isinstance(record.get("environment", {}), dict):
                raise ValueError("Expected an environment object")
        except FileNotFoundError:
            row["record_error"] = "missing_record"
        except (ValueError, UnicodeError):
            row["record_error"] = "invalid_record"
        else:
            row.update(
                record_status=record.get("status"),
                numerical_source=record.get("environment", {}).get("git_head"),
                reported_verified_problems=record.get("verified_problems"),
                reported_all_problems_verified=record.get("all_problems_verified"),
                stale_running_record=state in TERMINAL_STATES and record.get("status") == "running",
            )
        rows.append(row)
    return {
        "schema": "coupled-run-status-summary-v1",
        "runs": rows,
        "outcome_counts": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        "all_processes_terminal": all(row["terminal"] for row in rows),
        "scope": (
            "Scheduler elapsed time is allocation time, not complete optimization time. "
            "Process completion and reported verification flags require a separate numerical "
            "accuracy audit. Missing, interrupted and cancelled runs establish no speedup. "
            "Original records remain unchanged."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.manifest.read_bytes()
    result = summarize(parse_object(content), args.manifest.parent)
    result["manifest_sha256"] = hashlib.sha256(content).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", result)


if __name__ == "__main__":
    main()
