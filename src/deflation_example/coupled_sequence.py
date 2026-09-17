"""One complete coupled optimization sequence in a fresh worker process.

Repeat this command in separate processes for independent complete timings.
All declared targets remain in the output. A failed target supplies no warm
start; the next target starts without that failed state or recycling history.

A staged run executes a subset of the declared problems, restoring the
preceding stage's verified warm start from its saved arrays, writing a
checkpoint after every accepted nonlinear iterate, and resuming an interrupted
attempt from its last checkpoint. Stage records are assembled into complete
sequence records by ``coupled_campaign``.
"""

import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import hydra
import numpy as np
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_bounds import temperature_bounds
from .coupled_control import FlowEvaluationError
from .coupled_optimize import (
    equations_verified,
    load_problem,
    observe_linear_solves,
    prolonged_initial_state,
    solver_options,
)
from .coupled_optimizer import NUMERICAL_POLICY, minimize_coupled
from .memory import ProcessMemory
from .coupled_reference import configured_reference
from .coupled_targets import desired_temperature
from .reporting import environment, file_sha256, write_arrays, write_fields, write_report
from .study_solvers import StudySolver
from .validation import integer

COMPLETE_SCHEMA = "coupled-complete-sequence-v1"
STAGE_SCHEMA = "coupled-sequence-stage-v1"
CHECKPOINT_SCHEMA = "coupled-stage-checkpoint-v1"
EXCLUDED_CONFIGURATION = {
    "output",
    "baseline_directory",
    "reference_baseline_directory",
    "methods",
    "query",
    "upper_K",
    "mode",
}


def prepare_device(device, interval):
    sampler = ProcessMemory("cpu", interval)
    details = {}

    def barrier():
        pass

    solver_class = StudySolver
    if device in {"cuda", "hybrid"}:
        import cupy as cp
        import pynvml
        from cupy_backends.cuda.libs import cusparse
        from .coupled_cuda_solver import CudaCoupledSolver

        cp.zeros(1)
        barrier = cp.cuda.get_current_stream().synchronize
        barrier()
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByPciBusId(cp.cuda.Device().pci_bus_id)
            sampler.device_uuid = pynvml.nvmlDeviceGetUUID(handle)
        finally:
            pynvml.nvmlShutdown()
        details = {
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "cusparse": cusparse.getVersion(cp.cuda.device.get_cusparse_handle()),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        }
        solver_class = CudaCoupledSolver
        if device == "hybrid":
            from .coupled_hybrid_solver import HybridCoupledSolver

            solver_class = HybridCoupledSolver
            details["operator_policy"] = (
                "CPU vectors and original residuals; CUDA blocks of at least 20 columns"
            )
    elif device != "cpu":
        raise ValueError("Choose cpu, cuda or hybrid")
    return sampler, barrier, solver_class, details


class RestoredEvaluation:
    """The accepted state and converged flows of a preceding verified problem.

    Only these two attributes feed a warm start, so the first evaluation of the
    next problem is the computation the single-process runner performs.
    """

    def __init__(self, state, velocity, pressure):
        self.state = np.asarray(state, dtype=float).copy()
        velocity = np.asarray(velocity, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        if (
            self.state.ndim != 1
            or velocity.ndim != 3
            or pressure.ndim != 2
            or len(velocity) != len(pressure)
            or not np.isfinite(self.state).all()
            or not np.isfinite(velocity).all()
            or not np.isfinite(pressure).all()
        ):
            raise ValueError(
                "A restored warm start needs finite state, slab velocities and pressures"
            )
        self.flows = tuple(
            FlowResult(velocity[n].copy(), pressure[n].copy(), "converged", [])
            for n in range(len(velocity))
        )


def stage_settings(cfg):
    """Validate the optional stage group; None means the whole declared sequence."""
    stage = cfg.get("stage")
    if stage is None or stage.get("positions") is None:
        return None
    positions = [integer(p, "Stage position", 0) for p in stage["positions"]]
    if (
        not positions
        or len(set(positions)) != len(positions)
        or positions != sorted(positions)
        or positions[-1] >= len(cfg["queries"])
    ):
        raise ValueError("Stage positions must be distinct, increasing and declared")
    restore, resume = stage.get("restore"), stage.get("resume")
    if restore is not None and positions[0] == 0:
        raise ValueError("The first declared problem has no preceding stage to restore")
    if resume is not None and len(positions) != 1:
        raise ValueError("Resume exactly one problem per stage")
    return {
        "positions": positions,
        "restore": None if restore is None else Path(restore),
        "resume": None if resume is None else Path(resume),
    }


def recorded_configuration(cfg):
    return {k: v for k, v in cfg.items() if k not in EXCLUDED_CONFIGURATION}


def configuration_digest(configuration):
    """Identity of every setting except the stage bookkeeping."""
    identity = {k: v for k, v in configuration.items() if k != "stage"}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_restore(path):
    """Read a completed stage's warm start; hashes are checked before any use."""
    tick = time.perf_counter()
    record_path = path / "record.json"
    record = json.loads(record_path.read_text())
    if record.get("schema") != STAGE_SCHEMA or record.get("status") != "complete":
        raise ValueError("A warm start restores only a complete stage record")
    final = record["stage"]["final"]
    restored, arrays = None, None
    if final["warm_start_valid"]:
        arrays_path = path / final["path"]
        digest = file_sha256(arrays_path)
        if digest != final["sha256"]:
            raise ValueError("Restored warm-start arrays do not match the recorded checksum")
        with np.load(arrays_path) as data:
            arrays = {key: data[key] for key in data.files}
        restored = RestoredEvaluation(arrays["state"], arrays["velocity"], arrays["pressure"])
    return {
        "record": record,
        "record_sha256": file_sha256(record_path),
        "arrays_sha256": None if arrays is None else final["sha256"],
        "warm_start_valid": bool(final["warm_start_valid"]),
        "evaluation": restored,
        "recycling": None
        if arrays is None or "recycling_basis" not in arrays
        else {
            "indices": arrays["recycling_indices"],
            "basis": arrays["recycling_basis"],
            "previous": arrays["recycling_previous"]
            if arrays.get("recycling_previous_present", np.array(False))
            else None,
        },
        "seconds": time.perf_counter() - tick,
        "path": str(path),
    }


def load_resume(path, position, digest, method):
    """Read an interrupted attempt's last checkpoint for the same declared problem."""
    tick = time.perf_counter()
    meta = json.loads((path / "checkpoint-latest.json").read_text())
    if meta.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Unknown checkpoint schema")
    if meta["position"] != position or meta["method"] != method:
        raise ValueError("A checkpoint resumes only its own declared problem and method")
    if meta["configuration_sha256"] != digest:
        raise ValueError("A checkpoint resumes only an identical configuration")
    arrays_path = path / "checkpoint-latest.npz"
    if file_sha256(arrays_path) != meta["arrays_sha256"]:
        raise ValueError("Checkpoint arrays do not match the recorded checksum")
    with np.load(arrays_path) as data:
        arrays = {key: data[key] for key in data.files}
    steps, gradients = arrays["secant_steps"], arrays["secant_gradients"]
    resume = {
        "state": arrays["state"],
        "initial_evaluation": RestoredEvaluation(
            arrays["state"], arrays["velocity"], arrays["pressure"]
        ),
        "history": meta["history"],
        "damping": meta["damping"],
        "secants": list(zip(steps, gradients, strict=True)),
        "iteration": meta["iteration"],
    }
    recycling = None
    if "recycling_basis" in arrays:
        recycling = {
            "indices": arrays["recycling_indices"],
            "basis": arrays["recycling_basis"],
            "previous": arrays["recycling_previous"]
            if bool(arrays.get("recycling_previous_present", np.array(False)))
            else None,
        }
    return {
        "resume": resume,
        "recycling": recycling,
        "iteration": int(meta["iteration"]),
        "prior_seconds": float(meta["stage_elapsed_seconds"]),
        "prior_process_preparation_seconds": float(meta.get("process_preparation_seconds", 0.0)),
        "checkpoint_sha256": meta["arrays_sha256"],
        "seconds": time.perf_counter() - tick,
        "path": str(path),
    }


def history_arrays(payload):
    """Recycling arrays for an archive; the presence flag keeps None explicit."""
    if payload is None:
        return {}
    previous = payload.get("previous")
    return {
        "recycling_indices": np.asarray(payload["indices"], dtype=np.int64),
        "recycling_basis": np.asarray(payload["basis"], dtype=float),
        "recycling_previous": np.empty(0, dtype=np.int64)
        if previous is None
        else np.asarray(previous, dtype=np.int64),
        "recycling_previous_present": np.array(previous is not None),
    }


def optimize_targets(
    problem,
    solver,
    cfg,
    callback=None,
    *,
    positions=None,
    previous=None,
    checkpoint=None,
    resume=None,
    initial_state=None,
):
    """Keep every outcome; equivalent accepted-state warm starts across methods.

    ``positions`` selects the declared problems of one stage; ``previous`` is
    the restored warm start of the preceding stage; ``checkpoint`` receives the
    optimizer state after each accepted iterate; ``resume`` restarts the first
    selected problem from such a state; ``initial_state`` is a declared initial
    iterate for the first selected problem when no warm start is restored.
    """
    cases, fields = [], []
    positions = list(range(len(cfg["queries"]))) if positions is None else list(positions)
    for index, position in enumerate(positions):
        query = cfg["queries"][position]
        start = time.perf_counter()
        desired = desired_temperature(
            problem, query["target"], cfg["target_count"], cfg.get("target_startup_s", 0.0)
        )
        bounds = temperature_bounds(cfg, upper_K=query["upper_K"])
        lower = (
            bounds["optimization_lower_K"] - problem.temperature_offset
        ) / problem.temperature_scale
        upper = (
            bounds["optimization_upper_K"] - problem.temperature_offset
        ) / problem.temperature_scale
        row = {
            "position": position,
            **query,
            "temperature_bounds": bounds,
            "warm_start_used": previous is not None,
        }
        restart = resume if resume is not None and index == 0 else None
        if restart is not None:
            row["resumed_from_iteration"] = int(restart["iteration"])
        declared = initial_state if index == 0 and previous is None else None
        row["initial_control_used"] = declared is not None
        try:
            result = minimize_coupled(
                problem,
                desired,
                lower,
                upper,
                solver,
                initial=declared if previous is None else previous.state,
                initial_evaluation=previous,
                tolerance=cfg["nonlinear_tolerance"],
                max_iterations=cfg["nonlinear_cap"],
                qp_tolerance=cfg["qp_tolerance"],
                qp_cap=cfg["qp_cap"],
                backtracking=cfg["backtracking"],
                secant_memory=cfg["secant_memory"],
                callback=None if callback is None else lambda row, ev: callback(position, row, ev),
                checkpoint=None
                if checkpoint is None
                else lambda payload: checkpoint(position, payload),
                resume=restart,
            )
            checks = problem.verify(result.evaluation)
            adjoint = problem.verify_adjoint(result.evaluation, desired)
            adjoint_pass = (
                np.isfinite(adjoint["maximum_momentum_adjoint_relative_residual"])
                and adjoint["maximum_momentum_adjoint_relative_residual"] <= 1e-8
            )
            verified = result.status == "converged" and equations_verified(checks) and adjoint_pass
            row.update(
                status=result.status
                if equations_verified(checks) and adjoint_pass
                else "adjoint_verification_failed"
                if equations_verified(checks)
                else "equation_verification_failed",
                verified=verified,
                kkt=result.kkt,
                equations=checks,
                adjoint=adjoint,
                objective=result.objective * problem.objective_scale,
                nonlinear_iterations=len(result.history),
                history=result.history,
                inner_iterations=sum(
                    step.get("linear_iterations", 0)
                    for outer in result.history
                    for attempt in outer["attempts"]
                    for step in attempt["qp_history"]
                ),
            )
            fields.append(
                {
                    "state": result.evaluation.state.copy(),
                    "control": result.evaluation.control.copy(),
                    "desired": desired.copy(),
                    "velocity": np.stack([v.velocity for v in result.evaluation.flows]),
                    "pressure": np.stack([v.pressure for v in result.evaluation.flows]),
                }
            )
            previous = result.evaluation if cfg["warm_start"] and verified else None
            del result
        except FlowEvaluationError as failure:
            row.update(
                status="flow_" + failure.result.status,
                verified=False,
                failed_slab=failure.slab,
                metrics=failure.metrics,
                flow_history=failure.result.history,
            )
            fields.append(
                {
                    "velocity": failure.result.velocity.copy(),
                    "pressure": failure.result.pressure.copy(),
                    "state": np.empty(0)
                    if failure.temperature is None
                    else failure.temperature.copy(),
                }
            )
            previous = None
        except (MemoryError, RuntimeError, np.linalg.LinAlgError) as failure:
            row.update(
                status="memory_limited" if isinstance(failure, MemoryError) else "numerical_error",
                verified=False,
                error_type=type(failure).__name__,
            )
            fields.append({"state": np.empty(0)})
            row["seconds"] = time.perf_counter() - start
            cases.append(row)
            # A device/runtime failure can invalidate subsequent operations.
            # Preserve the remaining declared cases with explicit non-run status.
            for next_position in positions[index + 1 :]:
                cases.append(
                    {
                        "position": next_position,
                        **cfg["queries"][next_position],
                        "status": "not_run_after_numerical_error",
                        "verified": False,
                        "warm_start_used": False,
                        "seconds": 0.0,
                    }
                )
                fields.append({"state": np.empty(0)})
            break
        if not row["verified"]:
            solver.reset_history()
        row["seconds"] = time.perf_counter() - start
        cases.append(row)
    return cases, fields


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(config):
    cfg = OmegaConf.to_container(config, resolve=True)
    if cfg["method"] not in {"jacobi", "reference", "recycling"}:
        raise ValueError("The coupled study compares jacobi, reference and recycling")
    if not cfg["queries"] or not isinstance(cfg["warm_start"], bool):
        raise ValueError("Declare a nonempty target sequence and Boolean warm-start policy")
    integer(cfg["repetition"], "Repetition", 0)
    keys = [(integer(q["target"], "Target", 0), float(q["upper_K"])) for q in cfg["queries"]]
    if len(set(keys)) != len(keys):
        raise ValueError("The sequence must contain distinct target/bound pairs")
    stage = stage_settings(cfg)
    initial_control = bool(cfg.get("initial_control_directory"))
    if initial_control and (
        stage is None or len(stage["positions"]) != 1 or stage["restore"] is not None
    ):
        raise ValueError(
            "A saved initial control applies to one staged problem without a restored warm start"
        )
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    configuration = recorded_configuration(cfg)
    digest = configuration_digest(configuration)
    # Restore and resume inputs are read and hash-checked before any timer; a
    # mismatch is an error, never a silently changed path.
    restore = resume = None
    if stage is not None and stage["restore"] is not None:
        restore = load_restore(stage["restore"])
    if stage is not None and stage["resume"] is not None:
        resume = load_resume(stage["resume"], stage["positions"][0], digest, cfg["method"])
    started_utc = utc_now()
    with threadpool_limits(integer(cfg["threads"], "Threads", 1)):
        process_start = time.perf_counter()
        metadata = {
            "schema": COMPLETE_SCHEMA if stage is None else STAGE_SCHEMA,
            "numerical_policy": NUMERICAL_POLICY,
            "environment": environment(),
            "configuration": configuration,
            "timing_boundary": "Reference construction, model assembly, all nonlinear and active-set solves, transfers, independent verification and cleanup. Array serialization follows the timer. Common isothermal calibration and process preparation are separate.",
            "scope": "Coupled nonlinear stationary solutions; final physical-resolution and performance populations require their declared gates.",
        }
        write_report(output / "record.json", {**metadata, "status": "running"})
        sampler, barrier, solver_class, device = prepare_device(
            cfg["device"], cfg["memory_interval"]
        )
        if cfg["device"] == "hybrid":
            device = {**device, "coarse_correction_device": cfg.get("hybrid_coarse_device", "cpu")}
        sampler.start()
        preparation = time.perf_counter() - process_start
        barrier()
        start = tick = time.perf_counter()
        solver = reference = problem = None
        fatal_error = None
        cases, fields, parts, storage = [], [], {}, {}
        checkpoint_log, exported_history = [], None
        prior_seconds = 0.0 if resume is None else resume["prior_seconds"]
        try:
            problem, baseline = load_problem(cfg)
            if cfg.get("evaluation_progress", False):
                problem.evaluation_callback = lambda row: write_report(
                    output / "evaluation-progress.json", row
                )
            metadata.update(
                baseline_sha256=baseline["baseline_sha256"],
                baseline_configuration=baseline["configuration"],
                calibration_seconds=baseline["seconds"],
                input_sha256=baseline["input_sha256"],
                state_dofs_per_problem=problem.size,
            )
            if restore is not None:
                previous_record = restore["record"]
                if (
                    configuration_digest(previous_record["configuration"]) != digest
                    or previous_record["baseline_sha256"] != baseline["baseline_sha256"]
                    or previous_record["environment"]["source_sha256"]
                    != metadata["environment"]["source_sha256"]
                    or previous_record["stage"]["positions"][-1] != stage["positions"][0] - 1
                ):
                    raise ValueError(
                        "A restored stage must share configuration, baseline, source and precede this stage"
                    )
            initial_state = None
            if initial_control and resume is None:
                query = cfg["queries"][stage["positions"][0]]
                initial_state, metadata["initial_control"] = prolonged_initial_state(
                    {**cfg, "query": query["target"], "upper_K": query["upper_K"]},
                    problem,
                    baseline,
                )
            elif initial_control:
                metadata["initial_control"] = {
                    "status": "superseded_by_resume",
                    "directory": str(cfg["initial_control_directory"]),
                }
            parts["model_assembly_and_baseline_verification"] = time.perf_counter() - tick
            tick = time.perf_counter()
            if cfg["method"] == "reference":
                reference = configured_reference(problem, cfg, baseline)
                storage = reference.storage()
                metadata["reference_description"] = reference.description
                metadata["calibration_seconds"] += reference.description.get(
                    "additional_coarse_calibration_seconds", 0.0
                )
            parts["reference_construction"] = time.perf_counter() - tick
            tick = time.perf_counter()
            solver = solver_class(
                cfg["method"],
                reference=reference,
                rank=cfg["rank"],
                window=cfg["recycle_window"],
                rtol=cfg["inner_tolerance"],
                maxiter=cfg["inner_cap"],
                cg_factor=0.1,
                residual_policy="refine",
                **solver_options(cfg, solver_class),
            )
            if restore is not None and restore["recycling"] is not None:
                solver.import_history(restore["recycling"])
            if resume is not None and resume["recycling"] is not None:
                solver.import_history(resume["recycling"])
            if cfg.get("linear_progress", False):
                observe_linear_solves(solver, output / "linear-progress.json")
            parts["solver_resources"] = time.perf_counter() - tick
            tick = time.perf_counter()
            callback = (
                (
                    lambda position, row, ev: write_report(
                        output / "optimization-progress.json", {"position": position, **row}
                    )
                )
                if cfg.get("evaluation_progress", False) or cfg.get("linear_progress", False)
                else None
            )
            checkpoint = None
            if stage is not None:

                def checkpoint(position, payload):
                    began = time.perf_counter()
                    arrays = {
                        "state": np.asarray(payload["state"], dtype=float),
                        "velocity": np.stack([f.velocity for f in payload["flows"]]),
                        "pressure": np.stack([f.pressure for f in payload["flows"]]),
                        "secant_steps": np.array([s for s, _ in payload["secants"]]).reshape(
                            -1, problem.size
                        ),
                        "secant_gradients": np.array([g for _, g in payload["secants"]]).reshape(
                            -1, problem.size
                        ),
                        **history_arrays(solver.export_history()),
                    }
                    write_arrays(output / "checkpoint-latest.npz", **arrays)
                    write_report(
                        output / "checkpoint-latest.json",
                        {
                            "schema": CHECKPOINT_SCHEMA,
                            "position": position,
                            "method": cfg["method"],
                            "configuration_sha256": digest,
                            "iteration": payload["iteration"],
                            "damping": payload["damping"],
                            "objective": payload["objective"],
                            "history": payload["history"],
                            "optimizer_seconds": payload["optimizer_seconds"],
                            "stage_elapsed_seconds": prior_seconds + time.perf_counter() - start,
                            "process_preparation_seconds": preparation,
                            "arrays_sha256": file_sha256(output / "checkpoint-latest.npz"),
                            "written_utc": utc_now(),
                        },
                    )
                    checkpoint_log.append(time.perf_counter() - began)

            cases, fields = optimize_targets(
                problem,
                solver,
                cfg,
                callback=callback,
                positions=None if stage is None else stage["positions"],
                previous=None if restore is None else restore["evaluation"],
                checkpoint=checkpoint,
                resume=None if resume is None else resume["resume"],
                initial_state=initial_state,
            )
            exported_history = solver.export_history()
            barrier()
            parts["target_optimizations_and_verification"] = time.perf_counter() - tick
        except Exception as failure:
            fatal_error = failure
        finally:
            tick = time.perf_counter()
            try:
                if solver is not None:
                    solver.close()
            except Exception as failure:
                fatal_error = fatal_error or failure
            solver = reference = problem = None
            gc.collect()
            try:
                barrier()
            except Exception as failure:
                fatal_error = fatal_error or failure
            parts["cleanup"] = time.perf_counter() - tick
            total = time.perf_counter() - start
            try:
                memory = sampler.finish()
            except Exception as failure:
                fatal_error = fatal_error or failure
                memory = {"status": "measurement_failed", "error_type": type(failure).__name__}
        # The measured wall interval is primary. Small unclassified Python
        # intervals are explicit, so the partition has no overlaps or omissions.
        parts["bookkeeping"] = total - sum(parts.values())
        if stage is not None:
            # Algorithmic time of an interrupted attempt up to its last checkpoint
            # is part of this stage; the interrupted process's later work is not.
            parts["prior_attempt_work"] = prior_seconds
            total += prior_seconds
        finished_utc = utc_now()
        for row, arrays in zip(cases, fields, strict=True):
            write_fields(output / f"target-{row['position']:02d}.npz", **arrays)
        expected = len(cfg["queries"]) if stage is None else len(stage["positions"])
        report = {
            **metadata,
            "status": "complete" if fatal_error is None else "sequence_error",
            "error_type": None if fatal_error is None else type(fatal_error).__name__,
            # The type alone does not identify a failure that took a quarter of an
            # hour to reach. The message and the traceback's last frames say where.
            "error_message": None if fatal_error is None else str(fatal_error),
            "error_traceback": (
                None
                if fatal_error is None
                else "".join(
                    traceback.format_exception(
                        type(fatal_error), fatal_error, fatal_error.__traceback__
                    )
                )[-4000:]
            ),
            "device": device,
            "cases": cases,
            "verified_problems": sum(row["verified"] for row in cases),
            "all_problems_verified": fatal_error is None
            and len(cases) == expected
            and all(row["verified"] for row in cases),
            "sequence_seconds": total,
            "components_seconds": parts,
            "process_preparation_seconds": preparation,
            "preparation_inclusive_seconds": metadata.get("calibration_seconds", 0.0)
            + preparation
            + total,
            "memory": memory,
            "reference_storage": storage,
        }
        if stage is not None:
            valid = (
                fatal_error is None
                and bool(cases)
                and cfg["warm_start"]
                and cases[-1]["verified"]
                and "velocity" in fields[-1]
            )
            final = {"path": "stage-final.npz", "sha256": None, "warm_start_valid": valid}
            if valid:
                write_arrays(
                    output / final["path"],
                    state=fields[-1]["state"],
                    velocity=fields[-1]["velocity"],
                    pressure=fields[-1]["pressure"],
                    **history_arrays(exported_history),
                )
                final["sha256"] = file_sha256(output / final["path"])
            report["stage"] = {
                "positions": stage["positions"],
                "chain": {"method": cfg["method"], "repetition": cfg["repetition"]},
                "restore": None
                if restore is None
                else {
                    k: v
                    for k, v in restore.items()
                    if k not in {"record", "evaluation", "recycling"}
                },
                "resume": None
                if resume is None
                else {k: v for k, v in resume.items() if k not in {"resume", "recycling"}},
                "checkpoints": {
                    "count": len(checkpoint_log),
                    "seconds": float(sum(checkpoint_log)),
                    "timing_scope": "Included in target_optimizations_and_verification.",
                },
                "final": final,
                "job": {
                    key: os.environ.get(key)
                    for key in (
                        "SLURM_JOB_ID",
                        "SLURM_ARRAY_JOB_ID",
                        "SLURM_ARRAY_TASK_ID",
                        "SLURM_JOB_NODELIST",
                        "SLURM_CPUS_PER_TASK",
                        "SLURM_JOB_PARTITION",
                        "SLURM_JOB_QOS",
                    )
                },
                "node": os.uname().nodename,
                "started_utc": started_utc,
                "finished_utc": finished_utc,
            }
        write_report(output / "record.json", report)
        return report


@hydra.main(version_base="1.3", config_path="conf", config_name="coupled_sequence")
def main(config):
    run(config)


if __name__ == "__main__":
    main()
