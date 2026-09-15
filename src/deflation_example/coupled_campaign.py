"""Assemble staged coupled sequences into complete records and summarize them.

A chain is one method and repetition executed as consecutive stage jobs. The
assembler verifies that the stages are contiguous, share source, baseline,
configuration and device, and that every restored warm start carries the
checksum of the preceding stage's final arrays. The assembled record follows
the complete-sequence schema with stage timers summed, so the standard
reporter validates and compares it; its assembly policy is part of the
matched identity, so staged and single-process records never pool silently.
"""

import argparse
import json
from pathlib import Path
import re

from .coupled_report import summarize
from .coupled_saved import file_digest
from .reporting import write_report
from .validation import integer

STAGE_SCHEMA = "coupled-sequence-stage-v1"
COMPLETE_SCHEMA = "coupled-complete-sequence-v1"
ASSEMBLY_POLICY = "staged-v1"
STAGE_PATTERN = re.compile(r"stage-(\d+)(?:-resume-(\d+))?$")


def load_stage(directory):
    directory = Path(directory)
    path = directory / "record.json"
    record = json.loads(path.read_text())
    if record.get("schema") != STAGE_SCHEMA:
        raise ValueError(f"{directory} is not a stage record")
    return {"directory": str(directory), "record": record, "record_sha256": file_digest(path)}


def _identity(record):
    configuration = {k: v for k, v in record["configuration"].items() if k != "stage"}
    return json.dumps(
        {
            "configuration": configuration,
            "baseline_sha256": record["baseline_sha256"],
            "source_sha256": record["environment"]["source_sha256"],
            "device": record["device"],
            "numerical_policy": record["numerical_policy"],
            "timing_boundary": record["timing_boundary"],
            "chain": record["stage"]["chain"],
        },
        sort_keys=True,
    )


def assemble(stage_directories, interrupted_attempts=(), execution_status=None):
    """Combine completed stage records of one chain into a complete-sequence record."""
    stages = [load_stage(directory) for directory in stage_directories]
    if not stages:
        raise ValueError("Assemble at least one stage")
    first = stages[0]["record"]
    identity = _identity(first)
    positions, previous = [], None
    for entry in stages:
        record = entry["record"]
        if _identity(record) != identity:
            raise ValueError(
                "Stages of one chain must share configuration, source, baseline, device and chain"
            )
        if record["status"] != "complete":
            raise ValueError(
                "Only complete stage records assemble; failed attempts are retained separately"
            )
        stage = record["stage"]
        if previous is not None:
            expected_restore = previous["stage"]["final"]["warm_start_valid"]
            restore = stage["restore"]
            if stage["positions"][0] != previous["stage"]["positions"][-1] + 1:
                raise ValueError("Stages must be contiguous in declared order")
            if expected_restore:
                if (
                    restore is None
                    or restore["arrays_sha256"] != previous["stage"]["final"]["sha256"]
                ):
                    raise ValueError("A stage must restore the preceding stage's final arrays")
            elif restore is not None and restore["warm_start_valid"]:
                raise ValueError(
                    "A stage after an unverified problem must not restore a warm start"
                )
        elif stage["positions"][0] != 0 or stage["restore"] is not None:
            raise ValueError(
                "The first stage starts at the first declared problem without a restore"
            )
        positions.extend(stage["positions"])
        previous = record
    declared = len(first["configuration"]["queries"])
    complete_chain = positions == list(range(declared))
    # Restore provenance: each restored record checksum must match the stored predecessor.
    for index in range(1, len(stages)):
        restore = stages[index]["record"]["stage"]["restore"]
        if restore is not None and restore["record_sha256"] != stages[index - 1]["record_sha256"]:
            raise ValueError(
                "A restore must cite the checksum of the preceding stage record as assembled"
            )
    cases = [row for entry in stages for row in entry["record"]["cases"]]
    components = {}
    for entry in stages:
        for key, value in entry["record"]["components_seconds"].items():
            components[key] = components.get(key, 0.0) + float(value)
    total = float(sum(entry["record"]["sequence_seconds"] for entry in stages))
    preparation = float(sum(entry["record"]["process_preparation_seconds"] for entry in stages))
    calibration = float(first["calibration_seconds"])
    memories = [entry["record"]["memory"] for entry in stages]
    memory = {
        "complete": all(bool(m.get("complete")) for m in memories),
        "peak_host_rss_bytes": max((m.get("peak_host_rss_bytes") or 0) for m in memories),
        "peak_gpu_process_bytes": max((m.get("peak_gpu_process_bytes") or 0) for m in memories),
        "stages": memories,
        "scope": "Maximum sampled peak over stage processes; each stage sampled its own process.",
    }
    restart_overhead = preparation - float(first["process_preparation_seconds"])
    resumes = []
    for entry in stages:
        stage = entry["record"]["stage"]
        if stage["restore"] is not None:
            restart_overhead += float(stage["restore"]["seconds"])
        if stage["resume"] is not None:
            resumes.append(stage["resume"])
            restart_overhead += float(stage["resume"]["seconds"]) + float(
                stage["resume"].get("prior_process_preparation_seconds", 0.0)
            )
    discarded = None
    attempts = [load_attempt(directory) for directory in interrupted_attempts]
    if execution_status is not None:
        discarded = discarded_seconds(attempts, execution_status)
    completed_cost = [entry["record"].get("attempt_process_seconds") for entry in stages]
    scheduler_cost = {
        r["directory"]: r["elapsed_seconds"] for r in (execution_status or {}).get("attempts", [])
    }
    all_attempt_cost = None
    if all(v is not None for v in completed_cost) and all(
        a["directory"] in scheduler_cost for a in attempts
    ):
        all_attempt_cost = sum(completed_cost) + sum(
            float(scheduler_cost[a["directory"]]) for a in attempts
        )
        all_attempt_cost += sum(
            entry["record"].get("attempt_restore_seconds", 0) for entry in stages
        )
    verified = sum(bool(row["verified"]) for row in cases)
    all_verified = complete_chain and verified == len(cases) and len(cases) == declared
    started = min(entry["record"]["stage"]["started_utc"] for entry in stages)
    finished = max(entry["record"]["stage"]["finished_utc"] for entry in stages)
    record = {
        "schema": COMPLETE_SCHEMA,
        "numerical_policy": first["numerical_policy"],
        "environment": first["environment"],
        "configuration": {k: v for k, v in first["configuration"].items() if k != "stage"},
        "timing_boundary": first["timing_boundary"]
        + " Staged execution: stage timers are summed over the completed algorithmic path; restore loading and process preparation of every stage are reported outside the timers.",
        "scope": first["scope"],
        "baseline_sha256": first["baseline_sha256"],
        "baseline_configuration": first.get("baseline_configuration"),
        "calibration_seconds": calibration,
        "input_sha256": first.get("input_sha256"),
        "state_dofs_per_problem": first.get("state_dofs_per_problem"),
        "status": "complete" if complete_chain else "incomplete_chain",
        "error_type": None,
        "device": first["device"],
        "cases": cases,
        "verified_problems": verified,
        "all_problems_verified": bool(all_verified),
        "sequence_seconds": total,
        "components_seconds": components,
        "process_preparation_seconds": preparation,
        "preparation_inclusive_seconds": calibration + preparation + total,
        "all_attempt_process_seconds": all_attempt_cost,
        "all_attempt_preparation_inclusive_seconds": None
        if all_attempt_cost is None
        else calibration + all_attempt_cost,
        "memory": memory,
        "reference_storage": first.get("reference_storage", {}),
        "assembly": {
            "policy": ASSEMBLY_POLICY,
            "stages": [
                {
                    "directory": entry["directory"],
                    "positions": entry["record"]["stage"]["positions"],
                    "record_sha256": entry["record_sha256"],
                    "sequence_seconds": entry["record"]["sequence_seconds"],
                    "process_preparation_seconds": entry["record"]["process_preparation_seconds"],
                    "restore": entry["record"]["stage"]["restore"],
                    "resume": entry["record"]["stage"]["resume"],
                    "checkpoints": entry["record"]["stage"]["checkpoints"],
                    "job": entry["record"]["stage"]["job"],
                    "node": entry["record"]["stage"]["node"],
                    "started_utc": entry["record"]["stage"]["started_utc"],
                    "finished_utc": entry["record"]["stage"]["finished_utc"],
                }
                for entry in stages
            ],
            "interrupted_attempts": attempts,
            "restart_count": len(stages) - 1 + len(resumes),
            "restart_overhead_seconds": restart_overhead,
            "discarded_seconds": discarded,
            "started_utc": started,
            "finished_utc": finished,
            "claim_rule": "The speedup must hold under sequence_seconds, sequence_seconds plus restart_overhead_seconds, and wall elapsed time.",
        },
    }
    if "reference_description" in first:
        record["reference_description"] = first["reference_description"]
    return record


def load_attempt(directory):
    """An interrupted attempt contributes its checkpoint bookkeeping only."""
    directory = Path(directory)
    meta_path = directory / "checkpoint-latest.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    record_path = directory / "record.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    return {
        "directory": str(directory),
        "last_checkpoint": None
        if meta is None
        else {
            "iteration": meta["iteration"],
            "stage_elapsed_seconds": meta["stage_elapsed_seconds"],
            "attempt_process_seconds": meta.get("attempt_process_seconds"),
            "written_utc": meta["written_utc"],
            "arrays_sha256": meta["arrays_sha256"],
        },
        "record_status": None if record is None else record.get("status"),
        "job": None if record is None or "stage" not in record else record["stage"]["job"],
    }


def discarded_seconds(attempts, execution_status):
    """Scheduler elapsed time of interrupted attempts beyond their last checkpoint."""
    total = 0.0
    rows = {row["directory"]: row for row in execution_status.get("attempts", [])}
    previous_cumulative = 0.0
    for attempt in attempts:
        row = rows.get(attempt["directory"])
        if row is None or attempt["last_checkpoint"] is None:
            return None
        elapsed = float(row["elapsed_seconds"])
        checkpoint = attempt["last_checkpoint"]
        cumulative = float(checkpoint["stage_elapsed_seconds"])
        local = checkpoint.get("attempt_process_seconds")
        if local is None:
            # Legacy chains supply chronological cumulative retained work.
            local = (
                cumulative - previous_cumulative
                if cumulative >= previous_cumulative
                else cumulative
            )
        total += max(0.0, elapsed - float(local))
        previous_cumulative = cumulative
    return total


def discover_chains(root):
    """Map (method, repetition) to ordered completed stage directories and attempts."""
    root = Path(root)
    chains = {}
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for rep_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            match = re.fullmatch(r"rep-(\d+)", rep_dir.name)
            if match is None:
                continue
            key = (method_dir.name, int(match.group(1)))
            candidates = {}
            for stage_dir in sorted(p for p in rep_dir.iterdir() if p.is_dir()):
                found = STAGE_PATTERN.fullmatch(stage_dir.name)
                if found is None:
                    continue
                record_path = stage_dir / "record.json"
                record = json.loads(record_path.read_text()) if record_path.exists() else {}
                candidates.setdefault(int(found.group(1)), []).append(
                    (
                        stage_dir,
                        record.get("status") == "complete",
                        bool(record.get("all_problems_verified", False)),
                    )
                )
            stages, attempts = {}, []
            for index in sorted(candidates):
                # The last verified attempt of a stage is its record; every other
                # attempt (interrupted, unverified or superseded) stays visible.
                verified = [d for d, complete, ok in candidates[index] if complete and ok]
                complete = [d for d, complete, _ in candidates[index] if complete]
                chosen = verified[-1] if verified else complete[-1] if complete else None
                if chosen is not None:
                    stages[index] = chosen
                attempts.extend(d for d, _, _ in candidates[index] if d != chosen)
            ordered = [stages[k] for k in sorted(stages)]
            chains[key] = {"stages": ordered, "attempts": attempts}
    return chains


def hydra_value(value):
    """Render a JSON value in Hydra override syntax."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        if any(character in value for character in " ,=[]{}\"'"):
            raise ValueError(f"Cannot render {value!r} as an unquoted Hydra value")
        return value
    if isinstance(value, list):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{hydra_value(item)}" for key, item in value.items()) + "}"
    raise TypeError(f"Unsupported override value {value!r}")


def stage_overrides(
    protocol,
    plan,
    method,
    repetition,
    positions,
    output,
    restore=None,
    resume=None,
    rank=None,
    window=None,
    refresh=None,
    slabs=None,
):
    """Hydra overrides for one stage: protocol common settings plus the frozen selection.

    Before the freeze (``frozen_selection`` null) the rank, window and refresh
    interval must be supplied explicitly; afterwards they come from the plan
    and explicit values are rejected so a screen cannot silently drift. A
    declared off-grid resolution run (``slabs`` differing from the plan grid)
    is never a timed-population stage and must declare its settings explicitly.
    """
    if method not in protocol["methods"]:
        raise ValueError("Unknown method for this protocol")
    integer(repetition, "Repetition", 0)
    positions = [integer(p, "Stage position", 0) for p in positions]
    settings = dict(protocol["common"])
    grid = integer(plan["grid"]["slabs"], "Frozen slabs", 1)
    slabs = grid if slabs is None else integer(slabs, "Declared slabs", 1)
    frozen = plan.get("frozen_selection")
    if frozen is None or slabs != grid:
        stage_kind = "screen stages" if frozen is None else "off-grid resolution stages"
        if method != "jacobi" and (rank is None or window is None):
            raise ValueError(f"Rank and window must be declared explicitly for {stage_kind}")
        if refresh is None:
            raise ValueError(f"inner_refresh must be declared explicitly for {stage_kind}")
        settings.update(rank=0 if method == "jacobi" else rank, recycle_window=window or 1)
        settings["inner_refresh"] = refresh
    else:
        if rank is not None or window is not None or refresh is not None:
            raise ValueError("A frozen plan fixes rank, window and refresh; do not override them")
        policy = frozen["ranks"][method]
        settings.update(rank=policy["rank"], recycle_window=policy["recycle_window"])
        settings["inner_refresh"] = frozen["inner_refresh"]
    settings.update(
        slabs=slabs,
        method=method,
        repetition=repetition,
        baseline_directory=plan["baseline"]["directory"],
        output=str(output),
    )
    if "slabs" in protocol["common"]:
        raise ValueError("The grid is fixed by the frozen plan, not the protocol common block")
    overrides = [f"{key}={hydra_value(value)}" for key, value in settings.items()]
    overrides.append(f"stage.positions={hydra_value(positions)}")
    overrides.append(f"stage.restore={hydra_value(None if restore is None else str(restore))}")
    overrides.append(f"stage.resume={hydra_value(None if resume is None else str(resume))}")
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    stage = sub.add_parser("overrides", help="Print the Hydra overrides of one stage")
    stage.add_argument("--protocol", type=Path, required=True)
    stage.add_argument("--plan", type=Path, required=True)
    stage.add_argument("--method", required=True)
    stage.add_argument("--repetition", type=int, required=True)
    stage.add_argument("--positions", type=int, nargs="+", required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--restore", type=Path)
    stage.add_argument("--resume", type=Path)
    stage.add_argument("--rank", type=int)
    stage.add_argument("--window", type=int)
    stage.add_argument("--slabs", type=int)
    stage.add_argument("--refresh", type=int)
    one = sub.add_parser("assemble", help="Assemble the stage directories of one chain")
    one.add_argument("stages", type=Path, nargs="+")
    one.add_argument("--attempt", type=Path, action="append", default=[])
    one.add_argument("--execution-status", type=Path)
    one.add_argument("--output", type=Path, required=True)
    many = sub.add_parser("report", help="Assemble every chain under a root and summarize")
    many.add_argument("--root", type=Path, required=True)
    many.add_argument("--repetitions", type=int, default=5)
    many.add_argument("--rank-policy", type=Path)
    many.add_argument("--population", default="jacobi,reference")
    many.add_argument("--execution-status", type=Path)
    many.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "overrides":
        protocol = json.loads(args.protocol.read_text())
        plan = json.loads(args.plan.read_text())
        print(
            " ".join(
                stage_overrides(
                    protocol,
                    plan,
                    args.method,
                    args.repetition,
                    args.positions,
                    args.output,
                    args.restore,
                    args.resume,
                    args.rank,
                    args.window,
                    args.refresh,
                    args.slabs,
                )
            )
        )
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    status = json.loads(args.execution_status.read_text()) if args.execution_status else None
    if args.action == "assemble":
        record = assemble(args.stages, args.attempt, status)
        args.output.mkdir(parents=True, exist_ok=False)
        write_report(args.output / "record.json", record)
        return
    integer(args.repetitions, "Declared repetitions", 1)
    chains = discover_chains(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    records, assembled = [], []
    for (method, repetition), chain in sorted(chains.items()):
        try:
            record = assemble(chain["stages"], chain["attempts"], status)
        except ValueError as failure:
            assembled.append(
                {
                    "method": method,
                    "repetition": repetition,
                    "status": "not_assembled",
                    "reason": str(failure),
                }
            )
            continue
        target = args.output / "assembled" / f"{method}-rep-{repetition}"
        target.mkdir(parents=True, exist_ok=False)
        write_report(target / "record.json", record)
        assembled.append(
            {
                "method": method,
                "repetition": repetition,
                "status": record["status"],
                "all_problems_verified": record["all_problems_verified"],
                "sequence_seconds": record["sequence_seconds"],
                "restart_count": record["assembly"]["restart_count"],
            }
        )
        if record["status"] == "complete":
            records.append(record)
    rank_policy = json.loads(args.rank_policy.read_text()) if args.rank_policy else None
    summary = {"schema": "coupled-campaign-report-v1", "chains": assembled, "summary": None}
    if records:
        summary["summary"] = summarize(
            records, args.repetitions, rank_policy, args.population.split(",")
        )
        if args.rank_policy:
            summary["rank_policy_sha256"] = file_digest(args.rank_policy)
    write_report(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
