"""Matched fixed-point screens and restartable fixed-control trajectories.

All field archives are immutable generations. A manifest commits a completed
step only after its arrays have been written. Source and input hashes bind each
restart. Selection and full-trajectory evidence remain separate.
"""

import argparse
import json
from pathlib import Path
import time
import uuid

import numpy as np
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_fixed_point import solve_forward, merit
from .coupled_flow_solve import solve_momentum
from .coupled_newton_replay import newton_step, step_equations, criteria_met
from .coupled_optimize import load_problem
from .coupled_resolution import forward_model
from .coupled_saved import load_saved_solution, require_matching_baseline, file_digest
from .coupled_spatial_resolution import transfer_source
from .coupled_targets import desired_temperature
from .coupled_time_integration import saved_history, replay_time_scheme, effective_step
from .memory import ProcessMemory
from .reporting import environment, write_arrays, write_report
from .validation import integer

POLICIES = {
    "newton": {"policy": "newton"},
    "relaxed025": {"policy": "relaxed", "relaxation": 0.25},
    "relaxed050": {"policy": "relaxed", "relaxation": 0.5},
    "relaxed100": {"policy": "relaxed", "relaxation": 1.0},
    "anderson3": {"policy": "anderson", "depth": 3, "relaxation": 0.5},
    "anderson5": {"policy": "anderson", "depth": 5, "relaxation": 0.5},
}


def prepare(args, subdivision):
    saved, config, fields, digest = load_saved_solution(
        args.optimization, "reference", args.target_position
    )
    if not config["transient"]:
        raise ValueError("This study requires a complete transient control")
    comparison = {**config, "slabs": config["slabs"] * subdivision}
    coarse, baseline = load_problem({**comparison, "baseline_directory": str(args.baseline)})
    require_matching_baseline(saved, baseline)
    controls = np.repeat(fields["control"].reshape(config["slabs"], -1), subdivision, axis=0)
    if args.fine_baseline is None:
        problem, other = coarse, baseline
    else:
        problem, other = load_problem({**comparison, "baseline_directory": str(args.fine_baseline)})
        for key in (
            "properties",
            "inlet_velocity_m_s",
            "grad_div_coefficient_m2_s",
            "input_sha256",
        ):
            if baseline.get(key) != other.get(key):
                raise ValueError("Spatial refinement requires identical physical inputs")
        if (
            baseline["configuration"]["convection_form"]
            != other["configuration"]["convection_form"]
        ):
            raise ValueError("Momentum formulations differ")
        controls, _ = transfer_source(coarse, problem, controls)
    metadata = {
        "schema": "coupled-fixed-point-study-v1",
        "numerical_policy": "residual-equation-oseen-map-v3"
        if args.family == "momentum"
        else "matched-momentum-target-and-initial-guard-v2",
        "environment": environment(),
        "configuration": config,
        "policy": args.policy,
        "settings": POLICIES[args.policy],
        "family": args.family,
        "threads": args.threads,
        "repetition": args.repetition,
        "optimization_field_sha256": digest,
        "baseline_sha256": baseline["baseline_sha256"],
        "fine_baseline_sha256": other["baseline_sha256"],
        "subdivision": subdivision,
        "spatial_state_dofs": problem.spatial_size,
        "temperature_offset_K": problem.temperature_offset,
        "temperature_scale_K": problem.temperature_scale,
        "slabs": problem.slabs,
        "tolerance": 1e-12,
        "internal_momentum_tolerance": 1e-12,
        "flow_cap": 100,
        "coupling_cap": 300,
        "newton_cap": 100,
        "mass_tolerance": 1e-6,
        "energy_tolerance": 1e-6,
        "timing_boundary": "Step timers include original-equation verification. Model and input preparation, serialization, and process startup are separate. All failed attempts are retained.",
        "control_policy": "Unchanged signed source; piecewise constant on original control intervals. P1 prolongation on the nested spatial refinement.",
    }
    return problem, controls, metadata


def run_step(
    problem, control, previous, previous_flow, state, flow, slab, policy, family, callback=None
):
    model = forward_model(problem)
    settings = POLICIES[policy]
    start = time.perf_counter()
    if family == "momentum":
        result = solve_momentum(
            problem.flow,
            model._force(state, True),
            problem.boundary_indices,
            problem.boundary_values,
            initial=flow,
            previous=previous_flow.velocity,
            time_step=float(problem.physical_steps[slab]),
            pressure_gauge=problem.pressure_gauge,
            tolerance=1e-12,
            max_iterations=100,
            method="picard" if settings["policy"] == "relaxed" else settings["policy"],
            relaxation=settings.get("relaxation", 0.5),
            depth=settings.get("depth", 3),
        )
        checks = problem.flow.verify(
            result,
            model._force(state, True),
            problem.boundary_indices,
            problem.boundary_values,
            previous=previous_flow.velocity,
            time_step=float(problem.physical_steps[slab]),
            pressure_gauge=problem.pressure_gauge,
        )
        final_state, final_flow = state, result
        verified = result.status == "converged" and merit(checks) <= 1e-12
    else:
        if policy == "newton":
            result = newton_step(
                problem,
                control,
                previous,
                previous_flow,
                slab,
                initial_state=state,
                initial_flow=flow,
                tolerance=1e-12,
                max_iterations=100,
                line_search="equation_max",
                backtrack_cap=21,
            )
        else:
            result = solve_forward(
                model,
                control,
                state,
                flow,
                previous_state=previous,
                previous_velocity=previous_flow.velocity,
                time_step=float(problem.physical_steps[slab]),
                tolerance=1e-12,
                max_iterations=300,
                flow_cap=100,
                callback=callback,
                **settings,
            )
        _, checks = step_equations(
            problem, model, result.state, result.flow, control, previous, previous_flow, slab
        )
        final_state, final_flow = result.state, result.flow
        verified = result.status == "converged" and criteria_met(checks, 1e-12)
    row = {
        "slab_zero_based": slab,
        "status": result.status,
        "verified": bool(verified),
        "seconds": time.perf_counter() - start,
        "checks": checks,
        "history": result.history,
    }
    arrays = {
        "state": final_state[problem.free],
        "velocity": final_flow.velocity,
        "pressure": final_flow.pressure,
    }
    return row, arrays


def local(args):
    replay = json.loads((args.replay / "record.json").read_text())
    subdivision = integer(replay.get("subdivision", 1), "Subdivision", 1)
    problem, controls, metadata = prepare(args, subdivision)
    for key in ("optimization_field_sha256", "baseline_sha256"):
        if replay[key] != metadata[key]:
            raise ValueError("Replay input hash mismatch")
    if (
        replay.get("fine_baseline_sha256", replay["baseline_sha256"])
        != metadata["fine_baseline_sha256"]
    ):
        raise ValueError("Replay mesh hash mismatch")
    with np.load(args.replay / "states.npz", allow_pickle=False) as data:
        states, velocities, pressures, times = [
            data[k] for k in ("state", "velocity", "pressure", "times_s")
        ]
    slab = len(states) - 1 if args.slab is None else integer(args.slab, "Slab", 0)
    if slab >= len(states) or not np.allclose(
        times, np.cumsum(problem.physical_steps)[: len(states)]
    ):
        raise ValueError("Replay fields do not identify the selected physical step")
    if any(row["status"] != "converged" for row in replay["steps"][:slab]):
        raise ValueError("Selected step must have a verified preceding trajectory")
    scheme = replay_time_scheme(replay)
    effective, previous, previous_flow, coefficients = saved_history(
        problem, states, velocities, pressures, slab, scheme, restart=slab % subdivision == 0
    )
    state, flow = (
        (previous.copy(), previous_flow)
        if args.initial == "previous"
        else (
            problem.full_temperature(states[slab]),
            FlowResult(velocities[slab], pressures[slab], "saved", []),
        )
    )
    control = np.zeros(len(problem.mesh.nodes))
    control[problem.free] = controls[slab]
    metadata.update(
        time_scheme=scheme,
        time_s=float(times[slab]),
        initial=args.initial,
        replay_record_sha256=file_digest(args.replay / "record.json"),
        replay_fields_sha256=file_digest(args.replay / "states.npz"),
        coefficients=coefficients.tolist(),
        scope="Matched single-step solve; separate from complete-trajectory evidence.",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "record.json", {**metadata, "status": "running"})
    sampler = ProcessMemory(interval=0.05)
    sampler.start()
    try:
        row, arrays = run_step(
            effective,
            control,
            previous,
            previous_flow,
            state,
            flow,
            slab,
            args.policy,
            args.family,
            lambda item: write_report(args.output / "progress.json", item),
        )
        write_arrays(args.output / "fields.npz", **arrays)
        metadata.update(
            row=row, field_sha256=file_digest(args.output / "fields.npz"), status="complete"
        )
    except Exception as error:
        metadata.update(
            status="execution_failure", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        metadata["memory"] = sampler.finish()
        write_report(args.output / "record.json", metadata)


def commit_step(output, manifest, row, arrays):
    name = f"step-{row['slab_zero_based']:04d}-{uuid.uuid4().hex}.npz"
    write_arrays(output / name, **arrays)
    item = {**row, "fields": name, "field_sha256": file_digest(output / name)}
    committed = {**manifest, "steps": [*manifest["steps"], item]}
    if (output / "record.json").exists():
        write_report(output / "previous.json", manifest)
    write_report(output / "record.json", committed)
    return committed


def verify_restart(old, new, output):
    for key in (
        "optimization_field_sha256",
        "baseline_sha256",
        "fine_baseline_sha256",
        "policy",
        "subdivision",
        "family",
        "threads",
        "repetition",
        "time_scheme",
    ):
        if old[key] != new[key]:
            raise ValueError("Restart identity differs: " + key)
    if old["environment"]["source_sha256"] != new["environment"]["source_sha256"]:
        raise ValueError("Restart numerical source differs")
    for i, row in enumerate(old["steps"]):
        path = output / row["fields"]
        if (
            row["slab_zero_based"] != i
            or not row["verified"]
            or path.parent != output
            or file_digest(path) != row["field_sha256"]
        ):
            raise ValueError("Restart requires consecutive verified immutable steps")


def trajectory(args):
    if args.family != "forward":
        raise ValueError("A trajectory requires the complete forward equations")
    began = time.perf_counter()
    problem, controls, metadata = prepare(args, args.subdivision)
    metadata.update(
        time_scheme="backward_euler",
        steps=[],
        status="running",
        attempts=[],
        scope="Complete fixed-control trajectory; independently checked physical time levels.",
    )
    if args.resume:
        old = json.loads((args.output / "record.json").read_text())
        if old["status"] != "running":
            raise ValueError("Resume only an interrupted trajectory, never a numerical failure")
        verify_restart(old, metadata, args.output)
        metadata = old
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    attempt = {
        "id": uuid.uuid4().hex,
        "started_unix": time.time(),
        "input_preparation_seconds": time.perf_counter() - began,
    }
    metadata["attempts"] = [*metadata["attempts"], attempt]
    write_report(args.output / "record.json", metadata)
    state, flow = problem.full_temperature(problem.initial), problem.initial_flow
    if metadata["steps"]:
        with np.load(args.output / metadata["steps"][-1]["fields"], allow_pickle=False) as data:
            state = problem.full_temperature(data["state"])
            flow = FlowResult(data["velocity"], data["pressure"], "restored", [])
    sampler = ProcessMemory(interval=0.05)
    sampler.start()
    try:
        for n in range(len(metadata["steps"]), problem.slabs):
            source = np.zeros(len(problem.mesh.nodes))
            source[problem.free] = controls[n]
            effective, previous, previous_flow, _ = effective_step(
                problem, n, state, flow, None, None, "backward_euler"
            )
            row, arrays = run_step(
                effective,
                source,
                previous,
                previous_flow,
                state,
                flow,
                n,
                args.policy,
                args.family,
                lambda item: write_report(args.output / "progress.json", {"slab": n, **item}),
            )
            row["time_s"] = float(np.sum(problem.physical_steps[: n + 1]))
            metadata = commit_step(args.output, metadata, row, arrays)
            if not row["verified"]:
                metadata["status"] = "numerical_failure"
                break
            state = problem.full_temperature(arrays["state"])
            flow = FlowResult(arrays["velocity"], arrays["pressure"], "converged", [])
        else:
            metadata["status"] = "converged"
            states, velocities, pressures = [], [], []
            for row in metadata["steps"]:
                with np.load(args.output / row["fields"], allow_pickle=False) as data:
                    states.append(data["state"])
                    velocities.append(data["velocity"])
                    pressures.append(data["pressure"])
            states = np.stack(states)
            cfg = metadata["configuration"]
            desired = desired_temperature(
                problem, cfg["query"], cfg["target_count"], cfg.get("target_startup_s", 0)
            )
            physical = problem.temperature_offset + problem.temperature_scale * states
            flat = int(np.argmax(physical))
            slab, index = np.unravel_index(flat, physical.shape)
            metadata["quantities"] = {
                "maximum_temperature_K": float(physical.max()),
                "upper_bound_violation_K": float(max(0, physical.max() - cfg["upper_K"])),
                "lower_bound_violation_K": float(max(0, cfg["lower_K"] - physical.min())),
                "maximum_time_s": float(np.sum(problem.physical_steps[: slab + 1])),
                "maximum_location_rz": problem.mesh.nodes[problem.free[index]].tolist(),
                "weighted_tracking_squared_K2": float(
                    problem.temperature_scale**2
                    * np.sum(problem.weights * (states.ravel() - desired) ** 2)
                    / problem.weights.sum()
                ),
            }
            write_arrays(
                args.output / "states.npz",
                state=states,
                velocity=np.stack(velocities),
                pressure=np.stack(pressures),
                times_s=np.cumsum(problem.physical_steps),
            )
    except Exception as error:
        metadata.update(
            status="execution_failure", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        metadata["attempts"][-1].update(
            finished_unix=time.time(),
            process_seconds=time.perf_counter() - began,
            memory=sampler.finish(),
        )
        metadata["sum_step_seconds"] = sum(row["seconds"] for row in metadata["steps"])
        metadata["all_attempt_process_seconds"] = (
            sum(row["process_seconds"] for row in metadata["attempts"])
            if all("process_seconds" in row for row in metadata["attempts"])
            else None
        )
        write_report(args.output / "record.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("step", "trajectory"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fine-baseline", type=Path)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--target-position", type=int)
    parser.add_argument("--policy", choices=tuple(POLICIES), required=True)
    parser.add_argument("--family", choices=("forward", "momentum"), default="forward")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--slab", type=int)
    parser.add_argument("--initial", choices=("saved", "previous"), default="saved")
    parser.add_argument("--subdivision", type=int, default=1)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "step" and (args.replay is None or args.resume):
        parser.error("A local step requires a saved replay and a new output directory")
    args.subdivision = integer(args.subdivision, "Subdivision", 1)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        (local if args.mode == "step" else trajectory)(args)


if __name__ == "__main__":
    main()
