"""Gated execution of the fixed-physics coupled completion study.

The scheduler launcher is supplied by the deployment. Numerical jobs use the
frozen numerical source; controller changes do not relabel that source.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess

from .coupled_confirmation_report import summarize as confirmation_summary
from .coupled_retention_report import summarize, plot
from .coupled_trace import read_manifest
from .reporting import write_report, file_sha256


def screening_choices():
    return [
        ("jacobi", 0, "jacobi"),
        ("jacobi", 0, "frozen"),
        ("thermal", 20, "frozen"),
        ("thermal", 200, "frozen"),
        *[("preconditioned_coupled", r, "frozen") for r in (20, 50, 100, 200)],
    ]


def arm_settings(policy, rank):
    return {
        "jacobi": dict(
            method="jacobi",
            rank=0,
            recycle_window=1,
            inner_preconditioner="jacobi",
            reference_selection="thermal",
        ),
        "frozen": dict(
            method="jacobi",
            rank=0,
            recycle_window=1,
            inner_preconditioner="frozen",
            reference_selection="thermal",
        ),
        "recycling": dict(
            method="recycling",
            rank=rank,
            recycle_window=rank,
            inner_preconditioner="frozen",
            reference_selection="thermal",
        ),
        "reference": dict(
            method="reference",
            rank=rank,
            recycle_window=rank,
            inner_preconditioner="frozen",
            reference_selection=policy,
        ),
    }


def submit(launcher, source, mode, arguments, dependencies=(), gpu=False):
    command = ["sbatch", "--parsable"]
    if gpu:
        command.append("--gres=gpu:1")
    if dependencies:
        command.append("--dependency=afterany:" + ":".join(dependencies))
    command += [str(launcher), source, mode, *map(str, arguments)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("Unexpected scheduler identifier")
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("screen", "select", "assess", "confirm", "report"), required=True
    )
    parser.add_argument("--controller-source", required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    source = protocol["numerical_source"]
    if not all(re.fullmatch(r"[0-9a-f]{40}", s) for s in (source, args.controller_source)):
        raise ValueError("Full numerical and controller source identifiers are required")
    root = Path(protocol["run_root"])
    output = root / protocol["campaign"]
    output.mkdir(exist_ok=True)
    stage_path = output / (args.stage + "-jobs.json")
    if stage_path.exists():
        raise FileExistsError(
            "The stage already has retained submissions; inspect before restarting"
        )
    trace = root / protocol["trace"]
    manifest = read_manifest(trace)
    bank = json.loads((root / protocol["bank"] / "record.json").read_text())
    if (
        manifest["status"] != "complete"
        or bank["status"] != "complete"
        or bank["trace_sha256"] != file_sha256(trace / "manifest.json")
    ):
        raise ValueError("A complete matching trace and reference bank are required")
    state = {
        "stage": args.stage,
        "source": source,
        "protocol_sha256": file_sha256(args.protocol),
        "jobs": [],
        "status": "dispatching",
    }
    write_report(stage_path, state)
    lanes = [None] * 4

    def register(job, mode):
        registry = protocol.get("monitor_registry")
        if registry is None:
            return
        import fcntl

        path = Path(registry)
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(path.read_text())
            if job not in state["jobs"]:
                state["jobs"].append(job)
            prefix = {
                "replay": "coupled-completion-replay-v9-",
                "run": "coupled-completion-sequence-v9-",
            }.get(mode)
            if prefix is not None and prefix + job not in state["folders"]:
                state["folders"].append(prefix + job)
            write_report(path, state)

    def launch(mode, parameters, **description):
        lane = len(state["jobs"]) % 4
        job = submit(
            protocol["launcher"],
            source,
            mode,
            parameters,
            [lanes[lane]] if lanes[lane] else (),
            gpu=True,
        )
        lanes[lane] = job
        state["jobs"].append({"job": job, "mode": mode, **description})
        write_report(stage_path, state)
        register(job, mode)

    def follow(stage):
        state["next_job"] = submit(
            protocol["launcher"],
            args.controller_source,
            "campaign",
            [args.protocol, stage],
            [j for j in lanes if j],
        )
        register(state["next_job"], "campaign")

    def replays(partition, choices):
        for q, quadratic in enumerate(manifest["quadratics"]):
            if quadratic["partition"] == partition:
                for policy, rank, preconditioner in choices:
                    launch(
                        "replay",
                        [q, policy, rank, preconditioner, protocol["bank"]],
                        quadratic=q,
                        policy=policy,
                        rank=rank,
                        preconditioner=preconditioner,
                    )

    def replay_summary(stage, partition):
        jobs = json.loads((output / (stage + "-jobs.json")).read_text())["jobs"]
        reports, sources = [], []
        for job in jobs:
            path = root / f"coupled-completion-replay-v9-{job['job']}" / "record.json"
            if not path.exists():
                state.update(
                    status="incomplete", decision="missing_replay_record", missing_job=job["job"]
                )
                write_report(stage_path, state)
                return None
            reports.append(json.loads(path.read_text()))
            sources.append({"file": path.parent.name + "/record.json", "sha256": file_sha256(path)})
        result = summarize(manifest, reports, partition)
        result["records"] = sources
        folder = output / (partition + "-summary")
        folder.mkdir(exist_ok=False)
        write_report(folder / "summary.json", result)
        plot(result, folder)
        return result

    def complete_runs(phase, repetitions, selected):
        target_record = json.loads((root / protocol["targets"] / "targets.json").read_text())
        settings = {
            "phase": phase,
            "repetitions": repetitions,
            "targets": target_record["targets"],
            "arms": arm_settings(selected["policy"], selected["rank"]),
            "target_selection_sha256": file_sha256(root / protocol["targets"] / "targets.json"),
        }
        write_report(output / (phase + "-settings.json"), settings)
        for repeat in range(repetitions):
            names = list(settings["arms"])
            names = names[repeat % 4 :] + names[: repeat % 4]
            for name in names:
                arm = settings["arms"][name]
                policy = selected["policy"] if name == "reference" else arm["method"]
                launch(
                    "run",
                    [
                        policy,
                        arm["rank"],
                        arm["inner_preconditioner"],
                        repeat,
                        *settings["targets"],
                    ],
                    arm=name,
                    repetition=repeat,
                    phase=phase,
                )

    if args.stage == "screen":
        replays("selection", screening_choices())
        follow("select")
    elif args.stage == "select":
        result = replay_summary("screen", "selection")
        if result is not None and result["reference_beats_jacobi"]:
            selected = result["best_reference"]
            write_report(output / "frozen-choice.json", selected)
            replays(
                "held_out",
                [
                    (selected["policy"], selected["rank"], "frozen"),
                    ("jacobi", 0, "jacobi"),
                    ("jacobi", 0, "frozen"),
                ],
            )
            follow("assess")
        elif result is not None:
            state["decision"] = (
                "no_selection_cost_gain; retain all records and diagnose before further development"
            )
    elif args.stage == "assess":
        result = replay_summary("select", "held_out")
        if result is not None and result["reference_beats_jacobi"]:
            selected = json.loads((output / "frozen-choice.json").read_text())
            complete_runs("development", 3, selected)
            follow("confirm")
        elif result is not None:
            state["decision"] = "no_heldout_cost_gain; no complete speedup established"
    else:
        import numpy as np

        phase = "development" if args.stage == "confirm" else "confirmation"
        previous = "assess" if args.stage == "confirm" else "confirm"
        settings = json.loads((output / (phase + "-settings.json")).read_text())
        jobs = json.loads((output / (previous + "-jobs.json")).read_text())["jobs"]
        reports, arrays = [], []
        for job in jobs:
            folder = root / f"coupled-completion-sequence-v9-{job['job']}"
            if not (folder / "record.json").is_file():
                state.update(
                    status="incomplete", decision="missing_complete_record", missing_job=job["job"]
                )
                write_report(stage_path, state)
                return
            record = json.loads((folder / "record.json").read_text())
            reports.append(record)
            fields = []
            if record.get("all_problems_verified"):
                for i in range(3):
                    with np.load(folder / f"target-{i:02d}.npz", allow_pickle=False) as data:
                        fields.append(data["state"].copy())
            arrays.append(fields)
        result = confirmation_summary(reports, settings, arrays)
        write_report(output / (phase + "-summary.json"), result)
        if (
            args.stage == "confirm"
            and result["ten_percent_time_saving"]
            and result["observed_timing_ranges_separated"]
        ):
            selected = json.loads((output / "frozen-choice.json").read_text())
            complete_runs("confirmation", 5, selected)
            follow("report")
        else:
            state["decision"] = (
                "confirmation_complete" if args.stage == "report" else "development_gate_not_met"
            )
    if state["status"] == "dispatching":
        state["status"] = "dispatched" if state["jobs"] else "complete"
    write_report(stage_path, state)


if __name__ == "__main__":
    main()
