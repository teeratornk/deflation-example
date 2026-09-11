"""One complete coupled optimization sequence in a fresh worker process.

Repeat this command in separate processes for independent complete timings.
All declared targets remain in the output. A failed target supplies no warm
start; the next target starts without that failed state or recycling history.
"""

import gc
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy import sparse
from threadpoolctl import threadpool_limits

from .coupled_control import FlowEvaluationError
from .coupled_optimize import equations_verified, load_problem
from .coupled_optimizer import minimize_coupled
from .memory import ProcessMemory
from .mesh_control import build_mesh_control
from .mesh_reference import build_mesh_reference
from .mesh_showcases import desired_temperature
from .reporting import environment, write_fields, write_report
from .study_solvers import StudySolver
from .validation import integer


def build_reference(problem, rank):
    frozen = build_mesh_control(
        problem.assembly,
        alpha=problem.alpha,
        time_steps=problem.steps if len(problem.steps) else None,
    )
    return build_mesh_reference(
        frozen,
        problem.assembly,
        sparse.eye(problem.spatial_size, format="csr"),
        rank,
        spatial_policy="scaled_schur",
        temporal_metric="jacobi",
    )


def prepare_device(device, interval):
    sampler = ProcessMemory("cpu", interval)
    details = {}

    def barrier():
        pass

    solver_class = StudySolver
    if device == "cuda":
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
    elif device != "cpu":
        raise ValueError("Choose cpu or cuda")
    return sampler, barrier, solver_class, details


def optimize_targets(problem, solver, cfg):
    """Keep every outcome; equivalent accepted-state warm starts across methods."""
    cases, fields, previous = [], [], None
    lower = (cfg["lower_K"] - problem.temperature_offset) / problem.temperature_scale
    for position, query in enumerate(cfg["queries"]):
        start = time.perf_counter()
        desired = desired_temperature(problem, query["target"], cfg["target_count"])
        upper = (query["upper_K"] - problem.temperature_offset) / problem.temperature_scale
        row = {"position": position, **query, "warm_start_used": previous is not None}
        try:
            result = minimize_coupled(
                problem,
                desired,
                lower,
                upper,
                solver,
                initial=None if previous is None else previous.state,
                initial_evaluation=previous,
                tolerance=cfg["nonlinear_tolerance"],
                max_iterations=cfg["nonlinear_cap"],
                qp_tolerance=cfg["qp_tolerance"],
                qp_cap=cfg["qp_cap"],
                backtracking=cfg["backtracking"],
                secant_memory=cfg["secant_memory"],
            )
            checks = problem.verify(result.evaluation)
            verified = result.status == "converged" and equations_verified(checks)
            row.update(
                status=result.status
                if equations_verified(checks)
                else "equation_verification_failed",
                verified=verified,
                kkt=result.kkt,
                equations=checks,
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
            for next_position, remaining in enumerate(cfg["queries"][position + 1 :], position + 1):
                cases.append(
                    {
                        "position": next_position,
                        **remaining,
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
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(integer(cfg["threads"], "Threads", 1)):
        process_start = time.perf_counter()
        metadata = {
            "schema": "coupled-complete-sequence-v1",
            "environment": environment(),
            "configuration": {
                k: v
                for k, v in cfg.items()
                if k not in {"output", "baseline_directory", "methods", "query", "upper_K", "mode"}
            },
            "timing_boundary": "Reference construction, model assembly, all nonlinear and active-set solves, transfers, independent verification and cleanup. Array serialization follows the timer. Common isothermal calibration and process preparation are separate.",
            "scope": "Coupled nonlinear stationary solutions; final physical-resolution and performance populations require their declared gates.",
        }
        write_report(output / "record.json", {**metadata, "status": "running"})
        sampler, barrier, solver_class, device = prepare_device(
            cfg["device"], cfg["memory_interval"]
        )
        sampler.start()
        preparation = time.perf_counter() - process_start
        barrier()
        start = tick = time.perf_counter()
        solver = reference = problem = None
        fatal_error = None
        cases, fields, parts, storage = [], [], {}, {}
        try:
            problem, baseline = load_problem(cfg)
            metadata.update(
                baseline_sha256=baseline["baseline_sha256"],
                baseline_configuration=baseline["configuration"],
                calibration_seconds=baseline["seconds"],
                input_sha256=baseline["input_sha256"],
                state_dofs_per_problem=problem.size,
            )
            parts["model_assembly_and_baseline_verification"] = time.perf_counter() - tick
            tick = time.perf_counter()
            if cfg["method"] == "reference":
                reference = build_reference(problem, cfg["rank"])
                storage = reference.storage()
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
            )
            parts["solver_resources"] = time.perf_counter() - tick
            tick = time.perf_counter()
            cases, fields = optimize_targets(problem, solver, cfg)
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
        for index, arrays in enumerate(fields):
            write_fields(output / f"target-{index:02d}.npz", **arrays)
        report = {
            **metadata,
            "status": "complete" if fatal_error is None else "sequence_error",
            "error_type": None if fatal_error is None else type(fatal_error).__name__,
            "device": device,
            "cases": cases,
            "verified_problems": sum(row["verified"] for row in cases),
            "all_problems_verified": fatal_error is None
            and len(cases) == len(cfg["queries"])
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
        write_report(output / "record.json", report)
        return report


@hydra.main(version_base="1.3", config_path="conf", config_name="coupled_sequence")
def main(config):
    run(config)


if __name__ == "__main__":
    main()
