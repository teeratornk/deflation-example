"""Bounded initial-guess controls for a failed, fixed-source backward-Euler step.

Substeps supply a predictor only. A successful repair must satisfy the original
full-step equations with the original previous state, source and tolerances.
This diagnostic does not change the frozen replay or certify a full trajectory.
"""

import argparse
from copy import copy
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_fixed_point import solve_forward
from .coupled_forward_checkpoint import load_snapshot, save_snapshot
from .coupled_local_diagnostics import derivative_diagnostic
from .coupled_newton_replay import criteria_met, newton_step, step_equations
from .coupled_optimize import load_problem
from .coupled_prefix_diagnostics import read_forward
from .coupled_resolution import forward_model
from .coupled_saved import load_saved_solution, require_matching_baseline
from .reporting import environment, file_sha256, write_report
from .validation import integer


def verify(problem, source, previous, flow, slab, result):
    return step_equations(
        problem, forward_model(problem), result.state, result.flow, source, previous, flow, slab
    )[1]


def attempt_step(problem, source, previous, previous_flow, slab, policy, callback):
    """Retain every attempt; independently check original equations after correction."""
    if policy not in {"half_predictor", "anderson_previous"}:
        raise ValueError("Choose half_predictor or anderson_previous")
    start = time.perf_counter()
    attempts = []

    def retain(result, active, prior, flow, name):
        checks = verify(active, source, prior, flow, slab, result)
        details = {
            "procedure": name,
            "status": result.status,
            "seconds": result.seconds,
            "time_step_s": float(active.physical_steps[slab]),
            "independent_equations": checks,
            "independent_criteria_met": criteria_met(checks, 1e-12),
            "history": result.history,
        }
        attempts.append(details)
        callback(result, details)
        return details["independent_criteria_met"]

    if policy == "half_predictor":
        half = copy(problem)
        half.physical_steps = problem.physical_steps.copy()
        half.physical_steps[slab] *= 0.5
        half.steps = half.physical_steps / half.time_scale
        predictor_state, predictor_flow = previous.copy(), previous_flow
        for index in range(2):
            result = newton_step(
                half,
                source,
                predictor_state,
                predictor_flow,
                slab,
                tolerance=1e-12,
                max_iterations=100,
                line_search="fixed_scaled",
                backtrack_cap=21,
            )
            if not retain(result, half, predictor_state, predictor_flow, f"predictor-{index + 1}"):
                return result, {
                    "status": "predictor_failed",
                    "full_step_verified": False,
                    "attempts": attempts,
                    "seconds": time.perf_counter() - start,
                }
            predictor_state, predictor_flow = result.state, result.flow
        result = newton_step(
            problem,
            source,
            previous,
            previous_flow,
            slab,
            initial_state=predictor_state,
            initial_flow=predictor_flow,
            tolerance=1e-12,
            max_iterations=100,
            line_search="fixed_scaled",
            backtrack_cap=21,
        )
        passed = retain(result, problem, previous, previous_flow, "original-step-corrector")
    else:
        result = solve_forward(
            forward_model(problem),
            source,
            previous.copy(),
            previous_flow,
            previous_state=previous,
            previous_velocity=previous_flow.velocity,
            time_step=float(problem.physical_steps[slab]),
            policy="anderson",
            mixing="coupled",
            depth=5,
            relaxation=0.5,
            tolerance=1e-12,
            max_iterations=300,
            flow_cap=100,
        )
        passed = retain(result, problem, previous, previous_flow, "anderson-from-previous")
    return result, {
        "status": "converged" if passed else "full_step_not_converged",
        "full_step_verified": passed,
        "attempts": attempts,
        "seconds": time.perf_counter() - start,
    }


def checkpoint_fields(directory, record, fields, slab, problem):
    """Bind selected original fields to their immutable checkpoint records."""
    directory = directory / "checkpoints"
    index = json.loads((directory / "index.json").read_text())
    protocol = index["protocol"]
    for key in (
        "optimization_field_sha256",
        "baseline_sha256",
        "forward_slabs",
        "forward_solver",
        "forward_formulation",
        "configuration",
    ):
        if protocol[key] != record[key]:
            raise ValueError("Checkpoint and replay identities differ")
    if len(index["steps"]) != len(record["steps"]):
        raise ValueError("Checkpoint and replay step counts differ")
    for n in (slab - 1, slab):
        if n < 0:
            continue
        document, arrays = load_snapshot(directory, index["steps"][n], protocol)
        if (
            document["details"]["slab_zero_based"] != n
            or document["status"] != record["steps"][n]["status"]
        ):
            raise ValueError("Checkpoint step or status differs")
        for key in ("state", "velocity", "pressure"):
            expected = (
                problem.full_temperature(fields[key][n]) if key == "state" else fields[key][n]
            )
            if not np.array_equal(arrays[key], expected):
                raise ValueError("Replay fields differ from the committed checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--policy", choices=("diagnose", "half_predictor", "anderson_previous"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        saved, cfg, controls, digest = load_saved_solution(args.optimization, "reference", 0)
        record, fields, slab = read_forward(args.replay)
        if slab != len(record["steps"]) - 1 or record["status"] in {"converged", "running"}:
            raise ValueError("Select the first failed step of a terminated replay")
        if record["forward_solver"].get("time_scheme") != "backward_euler":
            raise ValueError("This diagnostic preserves backward-Euler equations only")
        if not record["forward_formulation"]["consistent_stabilization"]:
            raise ValueError("The assessment requires consistent stabilization")
        problem, baseline = load_problem(
            {
                **cfg,
                "slabs": record["forward_slabs"],
                "baseline_directory": str(args.baseline),
                "consistent_stabilization": True,
            }
        )
        require_matching_baseline(saved, baseline)
        public_cfg = {
            k: v
            for k, v in cfg.items()
            if k not in {"baseline_directory", "reference_baseline_directory", "output"}
        }
        if (
            record["configuration"] != public_cfg
            or digest != record["optimization_field_sha256"]
            or baseline["baseline_sha256"] != record["baseline_sha256"]
        ):
            raise ValueError("Baseline, control or configuration differs from the replay")
        if problem.slabs % cfg["slabs"]:
            raise ValueError("The saved source intervals must contain whole forward steps")
        checkpoint_fields(args.replay, record, fields, slab, problem)
        ratio = problem.slabs // cfg["slabs"]

        def state_at(n):
            if n < 0:
                return problem.full_temperature(problem.initial), problem.initial_flow
            return problem.full_temperature(fields["state"][n]), FlowResult(
                fields["velocity"][n].copy(), fields["pressure"][n].copy(), "saved", []
            )

        def source_at(n):
            source = np.zeros(len(problem.mesh.nodes))
            source[problem.free] = controls["control"].reshape(cfg["slabs"], -1)[n // ratio]
            return source

        previous, flow = state_at(slab - 1)
        model = forward_model(problem)
        preceding_checks = None
        if slab:
            older, older_flow = state_at(slab - 2)
            preceding_checks = step_equations(
                problem, model, previous, flow, source_at(slab - 1), older, older_flow, slab - 1
            )[1]
            if not criteria_met(preceding_checks, 1e-12):
                raise ValueError("The saved preceding state fails original-equation verification")
        protocol = {
            "schema": "coupled-late-step-diagnostic-v1",
            "environment": environment(),
            "original_source": record["environment"]["git_head"],
            "replay_record_sha256": file_sha256(args.replay / "record.json"),
            "replay_fields_sha256": file_sha256(args.replay / "states.npz"),
            "optimization_field_sha256": digest,
            "baseline_sha256": baseline["baseline_sha256"],
            "policy": args.policy,
            "slab_zero_based": slab,
            "start_s": float(problem.physical_steps[:slab].sum()),
            "end_s": float(problem.physical_steps[: slab + 1].sum()),
            "source_interval_zero_based": slab // ratio,
            "tolerance": 1e-12,
            "conservation_tolerance": 1e-6,
            "scope": "Fixed saved source and unchanged full-step equations; all predictor work retained; diagnostic cost is separate from complete-trajectory timing.",
        }
        write_report(args.output / "protocol.json", protocol)
        write_report(args.output / "preceding_verification.json", preceding_checks)
        if args.policy == "diagnose":
            for name, (current, current_flow) in (
                ("previous", (previous, flow)),
                ("stalled", state_at(slab)),
            ):
                report = derivative_diagnostic(
                    problem, source_at(slab), previous, flow, slab, current, current_flow
                )
                write_report(args.output / f"{name}.json", report)
        else:
            entries = []

            def callback(result, details):
                entry = save_snapshot(args.output, details["procedure"], result, details, protocol)
                entries.append(entry)
                write_report(args.output / "progress.json", {"attempts": entries, "last": details})

            result, report = attempt_step(
                problem, source_at(slab), previous, flow, slab, args.policy, callback
            )
            report["protocol"] = protocol
            report["snapshots"] = entries
            write_report(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
