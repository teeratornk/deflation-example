"""Dependency-driven execution of the frozen reference-retention protocol."""

import argparse
import json
from pathlib import Path
import re
import subprocess

from deflation_example.coupled_trace import read_manifest
from deflation_example.coupled_retention_report import (
    summarize,
    plot,
    summarize_feedback,
    plot_feedback,
    plot_complete,
)
from deflation_example.reporting import write_report, file_sha256


def submit(root, source, mode, arguments, dependency=None, gpu=False):
    command = ["sbatch", "--parsable"]
    if gpu:
        command += ["--gres=gpu:1"]
    if dependency:
        command += ["--dependency=afterany:" + ":".join(dependency)]
    command += [str(root / "coupled-retention-v8.sbatch"), source, mode, *map(str, arguments)]
    result = subprocess.run(command, check=True, text=True, capture_output=True, cwd=root)
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("Unexpected scheduler identifier")
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument(
        "--stage",
        choices=("selection", "widths", "heldout", "assessment", "feedback", "pilot", "final"),
        required=True,
    )
    args = parser.parse_args()
    if (
        not re.fullmatch(r"[0-9a-f]{40}", args.source)
        or not args.capture.isdigit()
        or not args.bank.isdigit()
    ):
        raise ValueError("Frozen source and numeric job identifiers are required")
    root = args.run_root.resolve()
    campaign = root / f"coupled-retention-campaign-v8-{args.bank}"
    campaign.mkdir(exist_ok=True)
    stage_path = campaign / (args.stage + "-jobs.json")
    if stage_path.exists():
        raise FileExistsError("Stage already dispatched; inspect its retained job identifiers")
    trace = root / f"coupled-retention-capture-v8-{args.capture}" / "linear-systems"
    manifest = read_manifest(trace)
    bank_path = root / f"coupled-retention-bank-v8-{args.bank}" / "record.json"
    bank = json.loads(bank_path.read_text())
    capture = json.loads((trace.parent / "record.json").read_text())
    if (
        not capture["all_problems_verified"]
        or manifest["status"] != "complete"
        or bank["status"] != "complete"
        or bank["trace_sha256"] != file_sha256(trace / "manifest.json")
    ):
        raise ValueError("Verified capture and matching complete reference bank are required")
    state = {
        "source": args.source,
        "capture": args.capture,
        "bank": args.bank,
        "stage": args.stage,
        "jobs": [],
        "status": "dispatching",
    }
    write_report(stage_path, state)
    selection_q = [i for i, q in enumerate(manifest["quadratics"]) if q["partition"] == "selection"]
    heldout_q = [i for i, q in enumerate(manifest["quadratics"]) if q["partition"] == "held_out"]
    lanes = [None] * 4

    def launch(mode, parameters, role, **metadata):
        lane = len(state["jobs"]) % len(lanes)
        job = submit(
            root, args.source, mode, parameters, [lanes[lane]] if lanes[lane] else None, gpu=True
        )
        lanes[lane] = job
        state["jobs"].append({"job": job, "role": role, **metadata})
        write_report(stage_path, state)
        return job

    def follow(stage):
        state["next_job"] = submit(
            root, args.source, "dispatch", [args.capture, args.bank, stage], [j for j in lanes if j]
        )

    def replays(quadratics, policies):
        # Interleave methods by quadratic; each worker retains all three repetitions.
        for q in quadratics:
            for policy, rank, width in policies:
                launch(
                    "replay",
                    [args.capture, args.bank, q, policy, rank, width],
                    "replay",
                    quadratic=q,
                    policy=policy,
                    rank=rank,
                    width=width,
                )

    def records(stages):
        results = []
        for stage in stages:
            prior = json.loads((campaign / (stage + "-jobs.json")).read_text())
            for item in prior["jobs"]:
                if item["role"] != "replay":
                    continue
                path = root / f"coupled-retention-replay-v8-{item['job']}" / "record.json"
                if not path.exists():
                    raise RuntimeError(
                        f"Missing replay result for job {item['job']}; retained in stage registry"
                    )
                results.append(json.loads(path.read_text()))
        return results

    def summary(stages, partition):
        result = summarize(manifest, records(stages), partition)
        output = campaign / (args.stage + "-summary")
        output.mkdir(exist_ok=False)
        write_report(output / "summary.json", result)
        plot(result, output)
        return result

    def feedback_summary():
        prior = json.loads((campaign / "assessment-jobs.json").read_text())
        reports = []
        for item in prior["jobs"]:
            if item["role"] == "feedback":
                path = root / f"coupled-retention-feedback-v8-{item['job']}" / "record.json"
                if not path.exists():
                    raise RuntimeError(f"Missing feedback record for job {item['job']}")
                reports.append(json.loads(path.read_text()))
        output = summarize_feedback(reports)
        write_report(campaign / "feedback-summary.json", output)
        plot_feedback(output, campaign)

    if args.stage == "selection":
        policies = [("jacobi", 0, 5)] + [
            (p, r, 5) for p in ("thermal", "nominal_coupled") for r in (20, 50, 100, 200)
        ]
        replays(selection_q, policies)
        follow("widths")
    elif args.stage == "widths":
        result = summary(["selection"], "selection")
        best = result["best_reference"]
        if best is None:
            state["decision"] = "no_verified_reference_configuration"
        else:
            replays(selection_q, [(best["policy"], best["rank"], w) for w in (1, 10, 20)])
            follow("heldout")
    elif args.stage == "heldout":
        result = summary(["selection", "widths"], "selection")
        best = result["best_reference"]
        if best is None or not heldout_q:
            state["decision"] = "heldout_evidence_unavailable"
        else:
            frozen = {
                "policy": best["policy"],
                "rank": best["rank"],
                "width": best["width"],
                "selection_summary_sha256": file_sha256(campaign / "heldout-summary/summary.json"),
            }
            write_report(campaign / "frozen-choice.json", frozen)
            policies = list(
                dict.fromkeys(
                    [
                        (best["policy"], best["rank"], best["width"]),
                        ("jacobi", 0, 5),
                        ("thermal", 200, 5),
                    ]
                )
            )
            replays(heldout_q, policies)
            follow("assessment")
    elif args.stage == "assessment":
        result = summary(["heldout"], "held_out")
        frozen = json.loads((campaign / "frozen-choice.json").read_text())
        key = (frozen["policy"], frozen["rank"], frozen["width"])
        chosen = next(
            (r for r in result["rows"] if (r["policy"], r["rank"], r["width"]) == key), None
        )
        plain = next((r for r in result["rows"] if r["policy"] == "jacobi"), None)
        improvement = (
            chosen is not None
            and chosen["eligible"]
            and plain is not None
            and plain["eligible"]
            and chosen["median_replay_total_seconds"] < plain["median_replay_total_seconds"]
        )
        # Sensitivity work proceeds regardless of the performance outcome.
        for q in sorted({selection_q[0], selection_q[-1]}):
            policies = list(
                dict.fromkeys(
                    [
                        ("jacobi", 0, 5),
                        ("thermal", frozen["rank"], 5),
                        ("nominal_coupled", frozen["rank"], frozen["width"]),
                    ]
                )
            )
            for multiplier in (0, 0.5, 1):
                for policy, rank, width in policies:
                    launch(
                        "feedback",
                        [args.capture, args.bank, q, policy, rank, width, multiplier],
                        "feedback",
                        quadratic=q,
                        policy=policy,
                        rank=rank,
                        feedback=multiplier,
                    )
        if improvement:
            methods = [
                ("jacobi", 0),
                ("thermal", 200),
                (frozen["policy"], frozen["rank"]),
                ("recycling", frozen["rank"]),
            ]
            methods = list(dict.fromkeys(methods))
            for rep in range(3):
                ordered = methods[rep % len(methods) :] + methods[: rep % len(methods)]
                for policy, rank in ordered:
                    launch(
                        "run",
                        [policy, rank, frozen["width"], rep],
                        "pilot",
                        policy=policy,
                        rank=rank,
                        repetition=rep,
                    )
            follow("pilot")
        else:
            state["decision"] = "no_heldout_complete_cost_motivation; stop optimization expansion"
            follow("feedback")
    elif args.stage == "feedback":
        feedback_summary()
        state["decision"] = "diagnostics_complete; no complete-cost improvement claimed"
    elif args.stage in {"pilot", "final"}:
        if args.stage == "pilot":
            feedback_summary()
        previous = "assessment" if args.stage == "pilot" else "pilot"
        prior = json.loads((campaign / (previous + "-jobs.json")).read_text())
        role = "pilot" if args.stage == "pilot" else "final"
        groups = {}
        for item in prior["jobs"]:
            if item["role"] != role:
                continue
            path = root / f"coupled-retention-complete-v8-{item['job']}" / "record.json"
            if not path.exists():
                raise RuntimeError(f"Missing complete result for {item['job']}")
            record = json.loads(path.read_text())
            groups.setdefault((item["policy"], item["rank"]), []).append(
                {
                    "job": item["job"],
                    "verified": record.get("all_problems_verified", False),
                    "sequence_seconds": record["sequence_seconds"],
                    "status": record["status"],
                    "record_sha256": file_sha256(path),
                    "sampled_gpu_peak_bytes": record.get("memory", {}).get(
                        "peak_gpu_process_bytes", 0
                    ),
                    "sampled_host_peak_bytes": record.get("memory", {}).get(
                        "peak_host_rss_bytes", 0
                    ),
                    "maximum_kkt": max(
                        (
                            max(c.get("kkt", {}).values(), default=float("inf"))
                            for c in record.get("cases", [])
                        ),
                        default=float("inf"),
                    ),
                    "inner_iterations": sum(
                        c.get("inner_iterations", 0) for c in record.get("cases", [])
                    ),
                }
            )
        import numpy as np

        required = 3 if args.stage == "pilot" else 5
        rows = [
            {
                "policy": key[0],
                "rank": key[1],
                "outcomes": values,
                "verified": len(values) == required and all(v["verified"] for v in values),
                "median_seconds": float(np.median([v["sequence_seconds"] for v in values])),
            }
            for key, values in groups.items()
        ]
        write_report(campaign / (args.stage + "-complete-summary.json"), {"rows": rows})
        plot_complete(rows, campaign, args.stage)
        frozen = json.loads((campaign / "frozen-choice.json").read_text())
        chosen = next(
            (r for r in rows if (r["policy"], r["rank"]) == (frozen["policy"], frozen["rank"])),
            None,
        )
        promising = (
            chosen is not None
            and len(rows) >= 3
            and all(r["verified"] for r in rows)
            and chosen["median_seconds"] < min(r["median_seconds"] for r in rows if r is not chosen)
        )
        if args.stage == "pilot" and promising:
            methods = list(groups)
            for rep in range(5):
                ordered = methods[rep % len(methods) :] + methods[: rep % len(methods)]
                for policy, rank in ordered:
                    launch(
                        "run",
                        [policy, rank, frozen["width"], rep + 3],
                        "final",
                        policy=policy,
                        rank=rank,
                        repetition=rep + 3,
                    )
            follow("final")
        else:
            state["decision"] = (
                "final_comparisons_complete"
                if args.stage == "final"
                else "pilot_does_not_support_expansion"
            )
    state["status"] = "dispatched" if state["jobs"] else "complete"
    write_report(stage_path, state)


if __name__ == "__main__":
    main()
