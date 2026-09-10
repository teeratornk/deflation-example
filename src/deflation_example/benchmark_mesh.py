"""Fresh-process complete optimization studies on body-fitted thermal meshes."""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from threadpoolctl import threadpool_limits

from .benchmark_cht import pack_mask
from .mesh_control import build_mesh_control
from .mesh_reference import build_mesh_reference
from .mesh_reference import DeviceMeshReference
from .mesh_showcases import build_showcase, desired_temperature
from .reporting import environment, write_report, write_fields
from .solvers import pdas, independent_residual, LinearResult, relative_norm
from .study_solvers import StudySolver
from .validation import integer, positive_real


@dataclass
class MeshStudyConfig:
    geometry: str = "engine_3d"
    level: int = 0
    transient: bool = False
    slabs: int = 4
    horizon: float = 0.1
    alpha: float = 1e-6
    bound: float = 0.2
    initial_temperature: float = 0.0
    targets: int = 3
    rank: int = 20
    recycle_rank: int | None = None
    construction: str = "mode_dependent"
    spatial_reference: str = "diffusion"
    temporal_metric: str = "euclidean"
    reference_device: str = "cpu"
    methods: list[str] = field(default_factory=lambda: ["jacobi", "reference", "recycling"])
    device: str = "cpu"
    repeats: int = 1
    warm_start: str = "outer_inner"
    phase: str = "pilot"
    threads: int = 4
    inner_cap: int = 10000
    residual_refresh: int = 10000
    outer_cap: int = 100
    rtol: float = 1e-10
    kkt_tolerance: float = 1e-8
    cg_factor: float = 0.1
    amgx_factor: float = 0.1
    residual_policy: str = "terminal"
    matrix_free_inner: bool = False
    monitor_memory: bool = False
    save_fields: bool = True
    timeout_seconds: float = 7200.0
    output: str = "runs/mesh-study"


def controls(config):
    c = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.structured(MeshStudyConfig), config),
        resolve=True,
        throw_on_missing=True,
    )
    c["recycle_rank"] = c["rank"] if c["recycle_rank"] is None else c["recycle_rank"]
    integer(c["recycle_rank"], "Recycling rank", 1)
    for key in ("level",):
        integer(c[key], key)
    for key in (
        "slabs",
        "targets",
        "rank",
        "repeats",
        "threads",
        "inner_cap",
        "outer_cap",
        "residual_refresh",
    ):
        integer(c[key], key, 1)
    for key in (
        "horizon",
        "alpha",
        "rtol",
        "kkt_tolerance",
        "cg_factor",
        "amgx_factor",
        "timeout_seconds",
    ):
        positive_real(c[key], key)
    for key in ("bound", "initial_temperature"):
        if not np.isfinite(c[key]):
            raise ValueError(f"{key} must be finite")
    if c["initial_temperature"] > c["bound"]:
        raise ValueError("Initial temperature must satisfy the fixed temperature bound")
    if c["bound"] < 0:
        raise ValueError("The upper bound must include the zero-excess Dirichlet temperature")
    if max(c["cg_factor"], c["amgx_factor"]) > 1:
        raise ValueError("Internal tolerance factors must not exceed one")
    if c["residual_policy"] not in {"terminal", "refine"}:
        raise ValueError("Residual policy must be terminal or refine")
    if c["geometry"] not in {"transformer_2d", "engine_3d"}:
        raise ValueError("Unknown geometry")
    if c["phase"] not in {"pilot", "final"} or c["device"] not in {"cpu", "cuda"}:
        raise ValueError("Unknown study phase or device")
    if c["construction"] not in {"mode_dependent", "tensor"}:
        raise ValueError("Unknown reference construction")
    if c["spatial_reference"] not in {"diffusion", "scaled_schur"}:
        raise ValueError("Unknown spatial reference construction")
    if c["temporal_metric"] not in {"euclidean", "jacobi"}:
        raise ValueError("Unknown temporal selection metric")
    if c["reference_device"] not in {"cpu", "cuda"} or (
        c["reference_device"] == "cuda" and c["device"] != "cuda"
    ):
        raise ValueError("CUDA reference restriction requires CUDA execution")
    if c["warm_start"] not in {"outer_inner", "cold"}:
        raise ValueError("Choose matched outer_inner or cold starts")
    if not c["methods"] or len(set(c["methods"])) != len(c["methods"]):
        raise ValueError("Methods must form a nonempty list without duplicates")
    if any(m not in {"jacobi", "reference", "recycling", "amgx", "direct"} for m in c["methods"]):
        raise ValueError("Unknown solver")
    if "amgx" in c["methods"] and c["device"] != "cuda":
        raise ValueError("AmgX requires CUDA")
    if "direct" in c["methods"] and (c["device"] != "cpu" or c["matrix_free_inner"]):
        raise ValueError("The direct validation solver requires assembled CPU restrictions")
    if c["matrix_free_inner"] and c["device"] != "cpu":
        raise ValueError("Matrix-free inner solves currently support CPU execution")
    if c["phase"] == "final" and (c["repeats"] < 5 or c["targets"] < 16):
        raise ValueError(
            "Final populations require at least 16 targets and five complete repetitions"
        )
    return c


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build_model(c):
    showcase = build_showcase(c["geometry"], c["level"])
    steps = np.full(c["slabs"], c["horizon"] / c["slabs"]) if c["transient"] else None
    initial = np.full(len(showcase.assembly.mesh.free), c["initial_temperature"])
    model = build_mesh_control(
        showcase.assembly,
        alpha=c["alpha"],
        time_steps=steps,
        initial=initial,
        assembled_restriction=not c["matrix_free_inner"],
    )
    return showcase, model


def sequence(c, method, torch=None, api=None):
    synchronize = (lambda: None) if torch is None else torch.cuda.synchronize
    sampler = None
    if c["monitor_memory"]:
        from .memory import ProcessMemory

        sampler = ProcessMemory(c["device"])
        try:
            sampler.start()
        except Exception as error:
            return (
                {
                    "success": False,
                    "failure": {
                        "status": "preparation_failed",
                        "stage": "memory_initialization",
                        "error_type": type(error).__name__,
                        "measurement_error_type": getattr(sampler, "initial_state", {}).get(
                            "error"
                        ),
                    },
                    "cases": [],
                },
                [],
                None,
            )
    synchronize()
    start = tick = time.perf_counter()
    adapter = reference = None
    cases, fields, components, storage = [], [], {}, {}
    problem = showcase = None
    failure = None
    initial_setup_seconds = None
    try:
        showcase, problem = build_model(c)
        components["assembly"] = time.perf_counter() - tick
        storage["state_operator_bytes"] = sum(
            a.nbytes for a in (problem.A.data, problem.A.indices, problem.A.indptr)
        )
        tick = time.perf_counter()
        if method == "reference":
            reference = build_mesh_reference(
                problem,
                showcase.coarse_assembly,
                showcase.prolongation,
                c["rank"],
                c["construction"],
                c["spatial_reference"],
                c["temporal_metric"],
            )
            if c["reference_device"] == "cuda":
                reference = DeviceMeshReference(reference, torch)
            storage.update(reference.storage())
            storage["reference_description"] = reference.description
        components["reference_construction"] = time.perf_counter() - tick
        tick = time.perf_counter()
        if method != "direct":
            retained_rank = c["recycle_rank"] if method == "recycling" else c["rank"]
            adapter = StudySolver(
                method,
                device=c["device"],
                rank=retained_rank,
                window=retained_rank,
                reference=reference,
                rtol=c["rtol"],
                cg_factor=c["cg_factor"],
                amgx_factor=c["amgx_factor"],
                maxiter=c["inner_cap"],
                refresh=c["residual_refresh"],
                torch=torch,
                api=api,
                resident_recycling=c["reference_device"] == "cuda",
                residual_policy=c.get("residual_policy", "terminal"),
            )
        components["solver_resources"] = time.perf_counter() - tick
        initial_setup_seconds = time.perf_counter() - start
        previous = None
        for query in range(c["targets"]):
            tick = time.perf_counter()
            desired = desired_temperature(problem, query, c["targets"])
            load = problem.load(desired)
            warm = c["warm_start"] == "outer_inner"
            active = (
                previous["active"] if previous is not None and warm else np.ones(problem.size, bool)
            )
            state = (
                previous["y"].copy() if previous is not None and warm else np.zeros(problem.size)
            )
            inner = []

            def solve(B, b, I):
                initial = state[I].copy() if warm else None
                if method == "direct":
                    from scipy.sparse.linalg import spsolve

                    solve_start = time.perf_counter()
                    y = spsolve(B, b)
                    residual = independent_residual(B, y, b)
                    solved = LinearResult(
                        y, 1, residual, "converged" if residual <= c["rtol"] else "residual_failed"
                    )
                    timing = {"callback_seconds": time.perf_counter() - solve_start}
                else:
                    solved, timing = adapter.solve(B, b, I, initial)
                if warm and solved.status == "converged":
                    state[:] = c["bound"]
                    state[I] = solved.x
                mask = np.zeros(problem.size, dtype=bool)
                mask[I] = True
                inner.append(
                    {
                        **timing,
                        "status": solved.status,
                        "iterations": solved.iterations,
                        "original_residual": solved.residual,
                        "deployed_rank": solved.rank,
                        "coarse_condition": solved.coarse_condition,
                        "fallback_reason": solved.fallback_reason,
                        "inactive_mask_bits": pack_mask(mask),
                        "rhs_sha256": _hash(b),
                        "initial_sha256": _hash(np.zeros(len(I)) if initial is None else initial),
                    }
                )
                return solved

            solved = pdas(
                problem.H,
                load,
                c["bound"],
                initial_active=active,
                tolerance=c["kkt_tolerance"],
                maxiter=c["outer_cap"],
                linear_solver=solve,
            )
            control, adjoint = problem.recover(solved["y"])
            recovery = relative_norm(
                problem.weights * (solved["y"] - desired)
                + problem.apply_transpose(adjoint)
                + solved["multiplier"],
                load,
            )
            status = solved["status"]
            if status == "converged" and (
                not np.isfinite(recovery) or recovery > c["kkt_tolerance"]
            ):
                status = "recovery_failed"
            if status == "converged" and any(row["original_residual"] > c["rtol"] for row in inner):
                status = "residual_failed"
            case = {
                "query": query,
                "status": status,
                "target_sha256": _hash(desired),
                "solution_sha256": _hash(solved["y"]),
                "seconds": time.perf_counter() - tick,
                "outer_iterations": solved["iterations"],
                "kkt": solved["kkt"],
                "inner_iterations": sum(row["iterations"] for row in inner),
                "outer_history": solved["history"],
                "outer_timing": solved["timing"],
                "inner": inner,
                "active_count": int(solved["active"].sum()),
                "active_mask_bits": pack_mask(solved["active"]),
                "objective": problem.objective(solved["y"], desired),
                "adjoint_recovery_relative": recovery,
            }
            if previous is not None:
                case["query_newly_active"] = int(np.sum(solved["active"] & ~previous["active"]))
                case["query_newly_inactive"] = int(np.sum(~solved["active"] & previous["active"]))
            if c["transient"]:
                masks = solved["active"].reshape(c["slabs"], -1)
                case["time_newly_active"] = np.sum(masks[1:] & ~masks[:-1], axis=1).tolist()
                case["time_newly_inactive"] = np.sum(~masks[1:] & masks[:-1], axis=1).tolist()
            case["cumulative_seconds"] = time.perf_counter() - start
            cases.append(case)
            if c["save_fields"]:
                fields.append(
                    {
                        "state": solved["y"].copy(),
                        "desired": desired.copy(),
                        "active": solved["active"].copy(),
                        "control": control.copy(),
                        "multiplier": solved["multiplier"].copy(),
                    }
                )
            previous = solved if status == "converged" else None
            if previous is None and adapter is not None:
                adapter.reset_history()
    except Exception as error:
        failure = {
            "status": "memory_error" if isinstance(error, MemoryError) else "exception",
            "error_type": type(error).__name__,
        }
    finally:
        tick = time.perf_counter()
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                failure = {"status": "cleanup_error", "error_type": type(error).__name__}
        adapter = reference = None
        synchronize()
        components["cleanup"] = time.perf_counter() - tick
    seconds = time.perf_counter() - start
    components["query_sequences"] = sum(case["seconds"] for case in cases)
    components["bookkeeping_and_unfinished_work"] = seconds - sum(components.values())
    mesh = None if showcase is None else showcase.assembly.mesh
    return (
        {
            "method": method,
            "seconds": seconds,
            "components_seconds": components,
            "initial_setup_seconds": initial_setup_seconds,
            "success": failure is None
            and len(cases) == c["targets"]
            and all(row["status"] == "converged" for row in cases),
            "failure": failure,
            "cases": cases,
            "storage": storage,
            "mesh": None if showcase is None else showcase.preparation,
            "parameters": None if showcase is None else showcase.parameters,
            "memory": None if sampler is None else sampler.finish(),
            "problem_size": None if problem is None else problem.size,
            "timing_scope": "assembly, reference construction, all PDAS steps and inner solves, transfers, verification, solver cleanup; field serialization and common startup excluded",
        },
        fields,
        mesh,
    )


def worker(specification, output, method, repetition):
    c = json.loads(Path(specification).read_text())
    if method not in c["methods"] or repetition not in range(c["repeats"]):
        raise ValueError("Worker must use a declared method and repetition")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    torch = api = None
    initialized = False
    fields, mesh = [], None
    startup = 0.0
    with threadpool_limits(c["threads"]):
        try:
            if c["device"] == "cuda":
                from .gpu import require_cuda

                torch = require_cuda()
                torch.set_num_threads(c["threads"])
                if "amgx" in c["methods"]:
                    import pyamgx as api
                    from .benchmark_cht import warmup

                    api.initialize()
                    initialized = True
                    warmup(torch, api)
                else:
                    from .gpu import gpu_deflated_cg

                    gpu_deflated_cg(
                        sparse.eye(16, format="csr"),
                        np.ones(16),
                        np.eye(16)[:, :2],
                        basis_backend="gpu_qr",
                    )
            # Identical small dense/sparse CPU warmup in every fresh process.
            np.linalg.cholesky(np.eye(16))
            sparse.eye(16, format="csr") @ np.ones(16)
            startup = time.perf_counter() - begin
            record, fields, mesh = sequence(c, method, torch, api)
        except Exception as error:
            record = {
                "success": False,
                "failure": {"status": "worker_exception", "error_type": type(error).__name__},
            }
        finally:
            finish_start = time.perf_counter()
            if initialized:
                try:
                    api.finalize()
                    torch.cuda.synchronize()
                except Exception as error:
                    record["success"] = False
                    record["finalization_error"] = type(error).__name__
            finalization = time.perf_counter() - finish_start
        record.update(
            {
                "protocol": "mesh-cht-complete-v1",
                "controls": c,
                "method": method,
                "repetition": repetition,
                "environment": environment(),
                "startup_after_import_seconds": startup,
                "finalization_seconds": finalization,
            }
        )
        if torch is not None:
            record["environment"].update(
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()
            )
        if initialized:
            from .amgx import amgx_configuration

            record["amgx_configuration"] = amgx_configuration(
                c["rtol"] * c["amgx_factor"], c["inner_cap"], True
            )
        write_report(output / "record.json", record)
        if mesh is not None and fields:
            write_fields(
                output / "geometry.npz",
                nodes=mesh.nodes,
                cells=mesh.cells,
                materials=mesh.materials,
                free=mesh.free,
            )
            for q, values in enumerate(fields):
                write_fields(output / f"fields-{q}.npz", **values)
    return record["success"]


def run(config):
    c = controls(config)
    output = Path(c.pop("output"))
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "protocol.json", c)
    results = []
    for repetition in range(c["repeats"]):
        order = (
            c["methods"][repetition % len(c["methods"]) :]
            + c["methods"][: repetition % len(c["methods"])]
        )
        for method in order:
            directory = output / f"{method}-{repetition}"
            start = time.perf_counter()
            command = [
                sys.executable,
                "-m",
                "deflation_example.benchmark_mesh",
                "--worker",
                str((output / "protocol.json").resolve()),
                "--output",
                str(directory.resolve()),
                "--method",
                method,
                "--repetition",
                str(repetition),
            ]
            try:
                result = subprocess.run(command, timeout=c["timeout_seconds"], check=False)
                status = "completed" if result.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
            results.append(
                {
                    "method": method,
                    "repetition": repetition,
                    "status": status,
                    "process_wall_seconds": time.perf_counter() - start,
                    "record": f"{method}-{repetition}/record.json",
                }
            )
            write_report(output / "attempts.json", results)
            print(f"{method} repetition {repetition}: {status}", flush=True)
    return all(row["status"] == "completed" for row in results)


@hydra.main(version_base=None, config_path="conf", config_name="mesh_study")
def main(config: DictConfig):
    if not run(config):
        raise SystemExit(1)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--method", required=True)
        parser.add_argument("--repetition", type=int, required=True)
        args = parser.parse_args()
        raise SystemExit(0 if worker(args.worker, args.output, args.method, args.repetition) else 1)
    main()
