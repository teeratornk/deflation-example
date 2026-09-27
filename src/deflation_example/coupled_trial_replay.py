"""Reevaluate one captured coupled trial without changing or optimizing its temperature."""

import argparse
import json
from pathlib import Path
import time

from threadpoolctl import threadpool_limits

from .coupled_control import FlowEvaluationError
from .coupled_optimize import equation_acceptance, load_problem
from .coupled_recovery import identity
from .coupled_sequence import RestoredEvaluation
from .coupled_targets import desired_temperature
from .coupled_trial_capture import read_trial
from .reporting import environment, file_sha256, write_report


def replay(problem, arrays, desired):
    base = arrays["base"]
    initial = RestoredEvaluation(base["state"], base["velocity"], base["pressure"])
    start = time.perf_counter()
    try:
        evaluation = problem.evaluate(arrays["candidate"]["state"], initial=initial)
        objective, _ = problem.objective_gradient(evaluation, desired)
        return {
            "status": "flow_converged",
            "objective_scaled": objective,
            "equations": problem.verify(evaluation),
            "flow_histories": [flow.history for flow in evaluation.flows],
            "seconds": time.perf_counter() - start,
        }
    except FlowEvaluationError as failure:
        return {
            "status": "flow_" + failure.result.status,
            "slab": failure.slab,
            "metrics": failure.metrics,
            "flow_history": failure.result.history,
            "seconds": time.perf_counter() - start,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--continuation", choices=("off", "on"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the preceding replay")
    record = json.loads((args.run / "record.json").read_text())
    cfg, env = record["configuration"], environment()
    if env["source_sha256"] != record["environment"]["source_sha256"]:
        raise ValueError("Replay requires the captured numerical source")
    if not 0 <= args.position < len(cfg["queries"]):
        raise ValueError("Target position is out of range")
    directory = args.run / "trial-capture" / f"target-{args.position:02d}"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["identity"] != identity(cfg, env["source_sha256"]):
        raise ValueError("Captured configuration or source differs")
    row, arrays = read_trial(directory, args.trial)
    cfg = {**cfg, "flow_continuation": args.continuation == "on"}
    with threadpool_limits(cfg["threads"]):
        problem, baseline = load_problem(cfg)
        if baseline["baseline_sha256"] != record["baseline_sha256"]:
            raise ValueError("Replay baseline differs")
        desired = desired_temperature(
            problem,
            cfg["queries"][args.position]["target"],
            cfg["target_count"],
            cfg["target_startup_s"],
        )
        result = replay(problem, arrays, desired)
    result.update(
        schema="coupled-trial-replay-v1",
        environment=env,
        original_trial=row,
        continuation=cfg["flow_continuation"],
        capture_manifest_sha256=file_sha256(directory / "manifest.json"),
        equations_passed=result["status"] == "flow_converged"
        and equation_acceptance(result["equations"], cfg),
        scope="Fixed captured temperature and initial flow. No optimization or completed-solve speedup; replay cost is separate.",
    )
    write_report(args.output, result)
    if not result["equations_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
