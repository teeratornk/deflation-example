"""Execute one frozen study task, select policies, or summarize all outcomes.

Use fresh processes on compute nodes. No command submits jobs or changes an
existing campaign. Run --help for the task indices in each phase.
"""

import argparse
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from deflation_example.reporting import environment, file_sha256, write_report
from deflation_example.fixed_point_study import POLICIES

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def screen_tasks(protocol):
    return list(itertools.product(protocol["families"], protocol["cases"], protocol["policies"]))


def selection(protocol, root):
    """All declared outcomes precede selection; compare time only on common cases."""
    result = {
        "schema": "fixed-point-selection-v1",
        "protocol_sha256": file_sha256(HERE / "protocol.json"),
        "environment": environment(),
        "families": {},
    }
    source = result["environment"]["source_sha256"]
    for family in protocol["families"]:
        outcomes = {}
        for case, policy in itertools.product(protocol["cases"], protocol["policies"]):
            folder = root / "screen" / family / case["name"] / policy
            record_path = folder / "record.json"
            if not record_path.exists():
                raise ValueError(f"Missing declared record: {family}/{case['name']}/{policy}")
            record = read(record_path)
            if record.get("status") == "running":
                raise ValueError("A declared screen task is still running")
            if record["environment"]["source_sha256"] != source:
                raise ValueError("Screen numerical sources differ")
            if record["optimization_field_sha256"] != protocol["optimization_field_sha256"]:
                raise ValueError("Screen saved controls differ")
            outcomes[(case["name"], policy)] = {
                **record.get("row", {}),
                "record_sha256": file_sha256(record_path),
            }
        names = [case["name"] for case in protocol["cases"]]
        candidates = [p for p in protocol["policies"] if p != "newton"]
        counts = {
            p: sum(outcomes[(c, p)].get("verified", False) for c in names)
            for p in protocol["policies"]
        }
        best_count = max(counts[p] for p in candidates)
        tied = [p for p in candidates if counts[p] == best_count]
        common = [c for c in names if all(outcomes[(c, p)].get("verified", False) for p in tied)]
        times = {
            p: statistics.median(outcomes[(c, p)]["seconds"] for c in common) if common else None
            for p in tied
        }
        chosen = (
            min(
                tied,
                key=lambda p: (
                    times[p] if times[p] is not None else float("inf"),
                    POLICIES[p].get("depth", 0),
                    candidates.index(p),
                ),
            )
            if best_count
            else None
        )
        result["families"][family] = {
            "selected": chosen,
            "converged_counts": counts,
            "common_timing_cases": common,
            "tie_break_median_seconds": times,
            "records": [{"case": c, "policy": p, **row} for (c, p), row in outcomes.items()],
        }
    destination = root / "selection.json"
    if destination.exists():
        raise FileExistsError("Selection is frozen; use a new study root for a new protocol")
    write_report(destination, result)
    return result


def solver_overrides(policy):
    settings = POLICIES[policy]
    method = "picard" if settings["policy"] == "relaxed" else settings["policy"]
    return [
        f"flow_method={method}",
        f"flow_relaxation={settings.get('relaxation', 0.5)}",
        f"flow_history={settings.get('depth', 3)}",
        "flow_cap=100",
    ]


def optimization_command(args, protocol, policy, *, method=None, repetition=0):
    # Keep all physical and optimizer settings in the existing Hydra interface.
    final = read(HERE.parent / "final_study" / "protocol.json")["common"]
    output = (
        args.output
        / args.phase
        / policy
        / ("derivatives" if method is None else f"{method}/rep-{repetition}")
    )
    values = {
        **final,
        "flow_cap": 100,
        "slabs": 64,
        "rank": 200,
        "recycle_window": 200,
        "inner_refresh": 1000,
        "threads": args.threads,
        "output": str(output),
        "baseline_directory": str(args.data_root / protocol["baseline"]),
        "reference_baseline_directory": str(args.data_root / protocol["baseline"]),
        "device": "cpu" if method is None else "hybrid",
        "mode": "derivatives" if method is None else "optimize",
        "evaluation_progress": True,
        "linear_progress": True,
    }
    if method is None:
        for key in ("queries", "warm_start", "memory_interval"):
            values.pop(key, None)
        module = "deflation_example.coupled_optimize"
    else:
        module = "deflation_example.coupled_sequence"
        values.update(
            method=method,
            repetition=repetition,
            queries=[{"target": 7, "upper_K": 357.3}],
            stage={"positions": [0], "restore": None, "resume": None},
        )
    # JSON is a valid Hydra value for these scalars/lists/mappings; use the
    # repository's canonical formatter for dictionaries and path strings.
    from deflation_example.coupled_campaign import hydra_value

    return [
        sys.executable,
        "-m",
        module,
        *[f"{k}={hydra_value(v)}" for k, v in values.items()],
        *solver_overrides(policy),
    ], output


def command(args, protocol):
    common = [sys.executable, "-m", "deflation_example.fixed_point_study"]
    inputs = [
        "--baseline",
        str(args.data_root / protocol["baseline"]),
        "--optimization",
        str(args.data_root / protocol["optimization"]),
        "--threads",
        str(args.threads),
    ]
    if args.phase == "screen":
        family, case, policy = screen_tasks(protocol)[args.task]
        output = args.output / "screen" / family / case["name"] / policy
        cmd = [
            *common,
            "step",
            *inputs,
            "--fine-baseline",
            str(args.data_root / protocol["fine_baseline"]),
            "--family",
            family,
            "--replay",
            str(args.data_root / case["replay"]),
            "--policy",
            policy,
            "--initial",
            case["initial"],
            "--output",
            str(output),
        ]
        if "slab" in case:
            cmd += ["--slab", str(case["slab"])]
        return cmd, output
    frozen = read(args.output / "selection.json")
    if frozen["protocol_sha256"] != file_sha256(HERE / "protocol.json"):
        raise ValueError("Frozen selection belongs to a different protocol")
    if frozen["environment"]["source_sha256"] != environment()["source_sha256"]:
        raise ValueError("Frozen selection belongs to different numerical source")
    family = "forward" if args.phase == "trajectory" else "momentum"
    selected = frozen["families"][family]["selected"]
    if selected is None:
        raise ValueError("No fixed-point policy passed the screen")
    if args.phase == "trajectory":
        # Twelve tasks: three repetitions, two grids, two solver policies.
        tasks = list(itertools.product(range(3), ("original", "fine"), ("newton", selected)))
        rep, mesh, policy = tasks[args.task]
        output = args.output / "trajectory" / mesh / policy / f"rep-{rep}"
        cmd = [
            *common,
            "trajectory",
            *inputs,
            "--policy",
            policy,
            "--repetition",
            str(rep),
            "--subdivision",
            "1" if mesh == "original" else "2",
            "--output",
            str(output),
        ]
        if mesh == "fine":
            cmd += ["--fine-baseline", str(args.data_root / protocol["fine_baseline"])]
        return cmd, output
    if args.phase == "derivatives":
        return optimization_command(args, protocol, ("newton", selected)[args.task])
    for policy in ("newton", selected):
        gate = read(args.output / "derivatives" / policy / "derivatives" / "record.json")
        if (
            gate["status"] != "verified"
            or gate["environment"]["source_sha256"] != environment()["source_sha256"]
        ):
            raise ValueError(
                "Both matching numerical derivative gates must pass before optimization"
            )
    tasks = list(
        itertools.product(range(3), ("jacobi", "reference", "recycling"), ("newton", selected))
    )
    rep, method, policy = tasks[args.task]
    return optimization_command(args, protocol, policy, method=method, repetition=rep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("screen", "select", "trajectory", "derivatives", "optimize")
    )
    parser.add_argument(
        "--task",
        type=int,
        default=0,
        help="screen 0..59; trajectory 0..11; derivatives 0..1; optimize 0..17",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.task < 0 or args.threads < 1:
        parser.error("Task must be nonnegative and threads positive")
    protocol = read(HERE / "protocol.json")
    if args.phase == "select":
        if args.dry_run:
            parser.error("Selection writes a frozen record; omit --dry-run")
        selection(protocol, args.output)
        return
    cmd, output = command(args, protocol)
    if args.dry_run:
        import shlex

        print(shlex.join(cmd))
        return
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt = output.parent / (output.name + "-attempt.json")
    if attempt.exists():
        raise FileExistsError(attempt)
    meta = {
        "phase": args.phase,
        "task": args.task,
        "environment": environment(),
        "protocol_sha256": file_sha256(HERE / "protocol.json"),
        "status": "running",
    }
    write_report(attempt, meta)
    began = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    write_report(
        attempt,
        {
            **meta,
            "status": "complete",
            "returncode": result.returncode,
            "elapsed_process_seconds": time.perf_counter() - began,
        },
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
