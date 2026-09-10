"""Fresh-process, four-way complete CHT optimization comparisons."""

import argparse
import base64
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import zlib

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from threadpoolctl import threadpool_limits

from .amgx import amgx_cg, amgx_configuration
from .benchmark_extended import digest, target_parameters, target_values
from .memory import ProcessMemory
from .problems import build_problem
from .reporting import environment, summarize_pdas, write_report
from .solvers import calibrate_bound, pdas
from .spectral import analytical_reference
from .spacetime_reference import build_space_time_reference
from .study_solvers import ArrayReference, METHODS, StudySolver
from .timing import PHASES
from .transient import ThermalTrajectory, build_transient_cht
from .validation import finite_real, integer, positive_real


@dataclass
class StudyConfig:
    phase: str = "pilot"
    device: str = "cuda"
    problem: str = "steady"
    n: int = 8
    slabs: int = 8
    horizon: float = 0.1
    solid_capacity: float = 2.0
    fluid_capacity: float = 1.0
    initial_temperature: float = 0.0
    reference_construction: str = "mode_dependent"
    reference_capacity: float = 1.0
    targets: int = 4
    rank: int = 12
    recycle_rank: int | None = None
    window: int = 12
    repeats: int = 1
    methods: list[str] = field(default_factory=lambda: list(METHODS))
    warm_starts: list[str] = field(default_factory=lambda: ["outer_inner"])
    rtol: float = 1e-10
    amgx_factor: float = 0.1
    outer_tolerance: float = 1e-8
    inner_cap: int = 10000
    residual_refresh: int = 1000
    outer_cap: int = 100
    initial_active: str = "empty"
    threads: int = 4
    bound: float | None = None
    calibration_grid: int = 8
    calibration_steps: int = 8
    calibration_activity: float = 0.2
    memory_interval: float = 0.01
    sequence_timeout: float = 7200
    output: str = "runs/cht-study"


def query_parameters(index, count):
    parameters = target_parameters(index, count)
    parameters["theta"] = parameters.pop("t")
    return parameters


def specification(config):
    config = OmegaConf.merge(OmegaConf.structured(StudyConfig), config)
    values = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    values.pop("output")
    if values["phase"] not in {"pilot", "final"}:
        raise ValueError("Declare pilot or final phase")
    if values["device"] not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    if values["problem"] not in {"steady", "transient"}:
        raise ValueError("Problem must be steady or transient CHT")
    if values["reference_construction"] not in {"mode_dependent", "tensor"}:
        raise ValueError("Unknown space-time construction")
    if values["initial_active"] not in {"empty", "all"}:
        raise ValueError("Initial active set must be empty or all")
    integer(values["slabs"], "Time slabs", 1)
    for key in ("horizon", "solid_capacity", "fluid_capacity", "reference_capacity"):
        positive_real(values[key], key)
    finite_real(values["initial_temperature"], "Initial temperature")
    for key in ("n", "calibration_grid", "targets"):
        integer(values[key], key, 2)
    for key in (
        "window",
        "repeats",
        "inner_cap",
        "residual_refresh",
        "outer_cap",
        "threads",
        "calibration_steps",
    ):
        integer(values[key], key, 1)
    integer(values["rank"], "Reference rank")
    if values["recycle_rank"] is None:
        values["recycle_rank"] = values["rank"]
    integer(values["recycle_rank"], "Recycling rank")
    dimension = values["n"] ** 3 * (values["slabs"] if values["problem"] == "transient" else 1)
    if max(values["rank"], values["recycle_rank"]) >= dimension:
        raise ValueError("Ranks must be smaller than the full optimization dimension")
    for key in ("rtol", "amgx_factor", "outer_tolerance", "memory_interval", "sequence_timeout"):
        positive_real(values[key], key)
    if values["amgx_factor"] > 1 or not 0 < values["calibration_activity"] < 1:
        raise ValueError("Invalid stopping margin or calibration fraction")
    if values["bound"] is not None:
        positive_real(values["bound"], "Physical bound")
    for key, allowed in (("methods", METHODS), ("warm_starts", ("cold", "outer_inner"))):
        requested = values[key]
        if not requested or len(set(requested)) != len(requested) or set(requested) - set(allowed):
            raise ValueError("Methods and start policies must be distinct declared choices")
    if values["device"] == "cpu" and "amgx" in values["methods"]:
        raise ValueError("The AmgX comparison requires CUDA")
    return {
        "protocol": "full-reference-cht-study-v1",
        "controls": values,
        "physics": {"alpha": 0.001, "conductivity_ratio": 100.0, "transport_amplitude": 50.0},
        "targets": [query_parameters(m, values["targets"]) for m in range(values["targets"])],
        "reference_policy": "one full-domain analytical Laplacian space per sequence; restrict at every inner solve",
        "transient_convention": "one complete trajectory per query; backward Euler with capacity=solid_capacity below x3=1/2 and fluid_capacity above; right-endpoint time weights in both objective terms; common cell-volume times mean-step factor removed from reduced equations; initial temperature prescribed; bound imposed at all positive time levels",
        "space_time_reference": "mode_dependent uses the temporal eigensystem for each spatial Laplacian eigenvalue; tensor uses one pure-time eigensystem; order candidate products by reference eigenvalue or Rayleigh value and retain the declared total rank; store spatial and temporal factors before restriction",
        "temporal_target": "existing unequal Gaussian weights and query-dependent centers, plus rotation 0.8*t/T, vertical motion 0.05*sin(2*pi*t/T), and amplitude 0.15+0.85*sin(pi*t/T)^2",
        "recycling_policy": "empty initial history; retain deployed coarse vectors and the last window new directions from each accepted CG solve; normalize columns in Jacobi coordinates; QR/small-factor SVD rank threshold 1e-12; retain the lowest scaled Ritz vectors; transfer by zero extension; coarse space fixed during a solve and residual restarts; clear history after a failed target",
        "coarse_condition_limit": 1e10,
        "rank_threshold": 1e-12,
        "residual_refresh": values["residual_refresh"],
        "stopping": "same independent original relative residual and outer KKT limits; AmgX native absolute target is amgx_factor*rtol*||b||; CG triggers fresh verification at rtol",
        "amgx_configuration": amgx_configuration(
            values["rtol"] * values["amgx_factor"], values["inner_cap"], True
        ),
        "resources": "AmgX base Config/Resources persist per sequence; each inner solve rebuilds matrix, vectors, solver and hierarchy",
        "warm_starts": "outer_inner passes the previous accepted target's active set and state; every accepted inner solve updates the full state; without accepted outer history use the declared initial_active policy; cold uses that same initial active set and zero inner guesses for each query; each repetition starts without history",
        "timing": "fresh process for every method/repetition/start policy; one complete sequence includes assembly, reference or resource setup, all targets and PDAS inner solves, transfers, recycling updates, verification and cleanup; all-method small warmup and process initialization are separate",
        "memory": "10 ms default sampled host RSS and NVML allocation of the same solver process; independent monitor process avoids delays from native calls holding the solver interpreter lock; all solver libraries and allocator caches included; monitor memory excluded; fresh worker process per sequence; sampling interval and maximum gap reported",
        "failures": "retain every declared sequence and target, inner cap, outer cap, cycle, exception, timeout and memory error; clear warm-start and recycling history after a failed target; report accepted timings separately",
        "mask_encoding": "zlib: followed by base64 of zlib-compressed, little-endian packed Boolean bits",
    }


def pack_mask(mask):
    raw = np.packbits(mask, bitorder="little").tobytes()
    return "zlib:" + base64.b64encode(zlib.compress(raw)).decode("ascii")


def unpack_mask(encoded, size):
    integer(size, "Mask dimension")
    if encoded.startswith("zlib:"):
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(
            base64.b64decode(encoded[5:], validate=True), (size + 7) // 8 + 1
        )
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Mask encoding has invalid compressed data")
    else:
        decoded = base64.b64decode(encoded, validate=True)
    raw = np.frombuffer(decoded, dtype=np.uint8)
    if len(raw) != (size + 7) // 8:
        raise ValueError("Mask encoding has the wrong dimension")
    return np.unpackbits(raw, bitorder="little")[:size].astype(bool)


def build_model(c, n=None):
    n = c["n"] if n is None else n
    if c["problem"] == "steady":
        return build_problem("cht", n)
    return build_transient_cht(
        n,
        c["slabs"],
        c["horizon"],
        solid_capacity=c["solid_capacity"],
        fluid_capacity=c["fluid_capacity"],
        initial=np.full(n**3, c["initial_temperature"]),
    )


def desired_and_load(model, parameters):
    if isinstance(model, ThermalTrajectory):
        desired = model.target(parameters)
        return desired, model.load(desired)
    desired = target_values(model.coordinates, parameters)
    return desired, desired


def build_reference(model, controls):
    if isinstance(model, ThermalTrajectory):
        return build_space_time_reference(
            model,
            controls["rank"],
            controls["reference_construction"],
            controls["reference_capacity"],
        )
    basis, modes = analytical_reference(controls["n"], 3, controls["rank"])
    return ArrayReference(basis, {"construction": "analytical", "mode_indices": modes})


def complete_sequence(protocol, method, warm, torch=None, api=None):
    c = protocol["controls"]
    barrier = (lambda: None) if torch is None else torch.cuda.synchronize
    sampler = ProcessMemory(c["device"], c["memory_interval"]).start()
    barrier()
    start = tick = time.perf_counter()
    cases, parts, previous = [], {}, None
    problem = adapter = reference = result = state = desired = initial_active = None
    error = None
    storage = {}
    try:
        problem = build_model(c)
        parts["assembly"] = time.perf_counter() - tick
        matrices = [problem.A, problem.H]
        if isinstance(problem, ThermalTrajectory):
            matrices += [
                problem.spatial.A,
                problem.spatial.H,
                problem.spatial.reference,
                problem.time_matrix,
            ]
        else:
            matrices += [problem.reference]
        storage["sparse_operator_bytes"] = sum(
            a.data.nbytes + a.indices.nbytes + a.indptr.nbytes
            for a in {id(a): a for a in matrices}.values()
        )
        matrices = None
        tick = time.perf_counter()
        if method == "reference":
            reference = build_reference(problem, c)
            storage.update(reference.storage())
            storage["reference_description"] = reference.description
        parts["reference_construction"] = time.perf_counter() - tick
        tick = time.perf_counter()
        adapter = StudySolver(
            method,
            device=c["device"],
            rank=c["recycle_rank"] if method == "recycling" else c["rank"],
            window=c["window"],
            reference=reference,
            rtol=c["rtol"],
            amgx_factor=c["amgx_factor"],
            maxiter=c["inner_cap"],
            refresh=c["residual_refresh"],
            torch=torch,
            api=api,
        )
        parts["solver_resources"] = time.perf_counter() - tick
        for parameters in protocol["targets"]:
            tick = time.perf_counter()
            desired, load = desired_and_load(problem, parameters)
            initial_active = (
                previous["active"]
                if previous is not None and warm != "cold"
                else np.ones(len(desired), dtype=bool)
                if c["initial_active"] == "all"
                else None
            )
            state = (
                previous["y"].copy()
                if previous is not None and warm != "cold"
                else np.zeros(len(desired))
            )
            construction = time.perf_counter() - tick
            inner = []

            def solve(B, b, I):
                initial = state[I].copy() if warm != "cold" else None
                result, timing = adapter.solve(B, b, I, initial)
                if warm != "cold" and result.status == "converged":
                    state[:] = c["bound"]
                    state[I] = result.x
                inactive_mask = np.zeros(len(desired), dtype=bool)
                inactive_mask[I] = True
                inner.append(
                    {
                        **timing,
                        "iterations": result.iterations,
                        "status": result.status,
                        "original_residual": result.residual,
                        "deployed_rank": result.rank,
                        "coarse_condition": result.coarse_condition,
                        "fallback_reason": result.fallback_reason,
                        "n_inactive": len(I),
                        "inactive_mask_bits": pack_mask(inactive_mask),
                        "rhs_sha256": digest(b),
                        "initial_sha256": digest(np.zeros(len(I)) if initial is None else initial),
                    }
                )
                return result

            tick = time.perf_counter()
            try:
                result = pdas(
                    problem.H,
                    load,
                    c["bound"],
                    initial_active=initial_active,
                    tolerance=c["outer_tolerance"],
                    maxiter=c["outer_cap"],
                    linear_solver=solve,
                )
                case = {
                    **summarize_pdas(result, time.perf_counter() - tick),
                    "outer_timing": result["timing"],
                    "initial_active_count": 0
                    if initial_active is None
                    else int(initial_active.sum()),
                    "final_active_count": int(result["active"].sum()),
                    "final_active_bits": pack_mask(result["active"]),
                    "solution_sha256": digest(result["y"]),
                }
                if isinstance(problem, ThermalTrajectory):
                    case["physical_objective"] = problem.objective(result["y"], desired)
                    case["recovery"] = problem.recovery_metrics(
                        result["y"], desired, result["multiplier"]
                    )
                    masks = result["active"].reshape(c["slabs"], c["n"] ** 3)
                    case["physical_time_newly_active"] = np.sum(
                        masks[1:] & ~masks[:-1], axis=1
                    ).tolist()
                    case["physical_time_newly_inactive"] = np.sum(
                        ~masks[1:] & masks[:-1], axis=1
                    ).tolist()
                    if result["status"] == "converged" and any(
                        not np.isfinite(value) or value > c["outer_tolerance"]
                        for value in case["recovery"].values()
                    ):
                        case["pdas_status"] = result["status"]
                        case["status"] = "recovery_failed"
                else:
                    control = problem.A @ result["y"]
                    case["physical_objective"] = float(
                        0.5
                        * (
                            np.sum((result["y"] - desired) ** 2)
                            + problem.alpha * np.sum(control**2)
                        )
                        / (problem.n + 1) ** 3
                    )
                    control = None
                previous = result if case["status"] == "converged" else None
                if previous is None:
                    adapter.reset_history()
            except (RuntimeError, ValueError, MemoryError, np.linalg.LinAlgError) as exception:
                case = {
                    "status": "memory_error"
                    if isinstance(exception, MemoryError)
                    or type(exception).__name__ == "OutOfMemoryError"
                    else "exception",
                    "error_type": type(exception).__name__,
                    "seconds": time.perf_counter() - tick,
                    "outer_iterations": 0,
                    "inner_iterations": sum(row["iterations"] for row in inner),
                }
                previous = None
                adapter.reset_history()
            cases.append(
                {
                    **case,
                    "query": parameters["index"],
                    "target_sha256": digest(desired),
                    "target_construction_seconds": construction,
                    "inner": inner,
                    "cumulative_seconds": time.perf_counter() - start,
                }
            )
    except (RuntimeError, ValueError, MemoryError, np.linalg.LinAlgError) as exception:
        error = type(exception).__name__
    finally:
        tick = time.perf_counter()
        try:
            if adapter is not None:
                adapter.close()
        except Exception as exception:
            error = error or type(exception).__name__
        finally:
            adapter = reference = problem = previous = result = state = desired = initial_active = (
                None
            )
            load = matrices = masks = None
        try:
            barrier()
        except Exception as exception:
            error = error or type(exception).__name__
        parts["cleanup"] = time.perf_counter() - tick
    total = time.perf_counter() - start
    memory = sampler.finish()
    for key in PHASES:
        parts["kernel_" + key] = sum(
            row["components_seconds"][key] for case in cases for row in case["inner"]
        )
    parts["target_construction"] = sum(case["target_construction_seconds"] for case in cases)
    parts["reference_and_recycling_transfer"] = sum(
        row["restriction_and_transfer_seconds"] for case in cases for row in case["inner"]
    )
    for key in ("preparation", "restriction", "inner_verification", "kkt_and_update"):
        parts["outer_" + key] = sum(
            case.get("outer_timing", {}).get("components_seconds", {}).get(key, 0.0)
            for case in cases
        )
    parts["remaining_host_work"] = total - sum(parts.values())
    if parts["remaining_host_work"] < -1e-9:
        raise ValueError("Complete-sequence timing components overlap")
    return {
        "method": method,
        "warm_start": warm,
        "total_seconds": total,
        "components_seconds": parts,
        "cases": cases,
        "storage": storage,
        "memory": memory,
        "error_type": error,
        "inner_iterations": sum(case["inner_iterations"] for case in cases),
        "outer_iterations": sum(case["outer_iterations"] for case in cases),
        "success": error is None
        and len(cases) == c["targets"]
        and all(case["status"] == "converged" for case in cases),
    }


def warmup(torch, api):
    """The same small warmup initializes both libraries in every GPU worker."""
    from .gpu import gpu_deflated_cg

    A = sparse.diags([-np.ones(31), 3 * np.ones(32), -np.ones(31)], [-1, 0, 1], format="csr")
    b = np.linspace(0.1, 1.0, 32)
    basis = np.linalg.eigh(A.toarray())[1][:, :4]
    first, _ = gpu_deflated_cg(A, b, basis, basis_backend="gpu_qr")
    second, _ = amgx_cg(
        A,
        b,
        api=api,
        synchronize=torch.cuda.synchronize,
        rtol=1e-11,
        acceptance_rtol=1e-10,
        rhs_relative=True,
    )
    if first.status != "converged" or second.status != "converged":
        raise RuntimeError("Common warmup failed its original residual check")


def worker(protocol_path, output, method, warm):
    protocol = json.loads(Path(protocol_path).read_text())
    c = protocol["controls"]
    if method not in c["methods"] or warm not in c["warm_starts"]:
        raise ValueError("Worker must use a declared method and start policy")
    if Path(output).exists():
        raise FileExistsError("Worker output already exists")
    torch = api = None
    initialized = False
    initialization = warmup_seconds = finalization = 0.0
    with threadpool_limits(limits=c["threads"]):
        observed_environment = environment()
        stage = "initialization"
        tick = time.perf_counter()
        try:
            if c["device"] == "cuda":
                from .gpu import require_cuda
                import pyamgx

                torch, api = require_cuda(), pyamgx
                torch.set_num_threads(c["threads"])
                torch.ones(1, device="cuda").sum().item()
                api.initialize()
                initialized = True
            initialization = time.perf_counter() - tick
            stage, tick = "warmup", time.perf_counter()
            if torch is not None:
                warmup(torch, api)
            warmup_seconds = time.perf_counter() - tick
            stage, tick = "sequence", time.perf_counter()
            result = complete_sequence(protocol, method, warm, torch, api)
        except Exception as exception:
            result = {
                "success": False,
                "status": "worker_exception",
                "failure_stage": stage,
                "error_type": type(exception).__name__,
                "failed_stage_seconds": time.perf_counter() - tick,
            }
        finally:
            tick = time.perf_counter()
            if initialized:
                try:
                    api.finalize()
                    torch.cuda.synchronize()
                except Exception as exception:
                    result["success"] = False
                    result["cleanup_error_type"] = type(exception).__name__
            finalization = time.perf_counter() - tick
    result.update(
        initialization_seconds=initialization,
        warmup_seconds=warmup_seconds,
        finalization_seconds=finalization,
        environment=observed_environment,
    )
    # Preserve the numerical outcome even if optional native metadata fails.
    write_report(output, result)
    if initialized:
        result["environment"].update(
            torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()
        )
        result["environment"]["amgx_api"] = api.get_api_version()
        result["environment"]["pyamgx_binary_sha256"] = hashlib.sha256(
            Path(api.__file__).read_bytes()
        ).hexdigest()
        distribution = importlib.metadata.distribution("pyamgx")
        result["environment"]["pyamgx"] = distribution.version
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id")
        if (
            isinstance(commit, str)
            and len(commit) == 40
            and all(ch in "0123456789abcdef" for ch in commit)
        ):
            result["environment"]["pyamgx_source_commit"] = commit
    for package in ("psutil", "nvidia-ml-py"):
        try:
            result["environment"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["environment"][package] = None
    write_report(output, result)
    return result


def run(config):
    protocol = specification(config)
    c = protocol["controls"]
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "declared_protocol.json", protocol)
    tick = time.perf_counter()
    fraction = None
    if c["bound"] is None:
        with threadpool_limits(limits=c["threads"]):
            calibration = build_model(c, c["calibration_grid"])
            _, calibration_load = desired_and_load(calibration, target_parameters(1, 3))
            c["bound"], fraction = calibrate_bound(
                calibration.H,
                calibration_load,
                c["calibration_activity"],
                steps=c["calibration_steps"],
            )
            del calibration, calibration_load
    calibration_seconds = time.perf_counter() - tick
    write_report(output / "protocol.json", protocol)
    protocol_hash = hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest()
    report = {
        "specification": protocol,
        "protocol_sha256": protocol_hash,
        "environment": environment(),
        "calibration_seconds": calibration_seconds,
        "calibrated_activity": fraction,
        "sequences": [],
        "success": False,
    }
    write_report(output / "results.json", report)
    for repetition in range(c["repeats"]):
        order = [(warm, method) for warm in c["warm_starts"] for method in c["methods"]]
        shift = repetition % len(order)
        order = order[shift:] + order[:shift]
        for warm, method in order:
            name = f"sequence-{repetition}-{warm}-{method}.json"
            tick = time.perf_counter()
            command = [
                sys.executable,
                "-m",
                "deflation_example.benchmark_cht",
                "worker",
                "--protocol",
                str((output / "protocol.json").resolve()),
                "--output",
                str((output / name).resolve()),
                "--method",
                method,
                "--warm",
                warm,
            ]
            try:
                child = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=c["sequence_timeout"],
                    check=False,
                )
                process_seconds = time.perf_counter() - tick
                if (output / name).is_file():
                    sequence = json.loads((output / name).read_text())
                    if child.returncode != 0:
                        sequence.update(
                            success=False, status="worker_failed", returncode=child.returncode
                        )
                else:
                    sequence = {
                        "success": False,
                        "status": "worker_failed",
                        "returncode": child.returncode,
                    }
            except subprocess.TimeoutExpired:
                process_seconds = time.perf_counter() - tick
                sequence = {"success": False, "status": "walltime_cap"}
            sequence.update(
                method=method,
                warm_start=warm,
                repetition=repetition,
                process_seconds=process_seconds,
                protocol_sha256=protocol_hash,
            )
            if "total_seconds" in sequence:
                measured = sum(
                    sequence[key]
                    for key in (
                        "total_seconds",
                        "initialization_seconds",
                        "warmup_seconds",
                        "finalization_seconds",
                    )
                )
                sequence["launch_import_report_and_exit_seconds"] = process_seconds - measured
                sequence["preparation_inclusive_seconds"] = process_seconds + calibration_seconds
            write_report(output / name, sequence)
            report["sequences"].append(
                {
                    "path": name,
                    "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                    "method": method,
                    "warm_start": warm,
                    "repetition": repetition,
                    "success": sequence["success"],
                    "total_seconds": sequence.get("total_seconds"),
                    "process_seconds": process_seconds,
                }
            )
            write_report(output / "results.json", report)
            print(
                f"{method}, {warm}, repeat {repetition}: accepted={sequence['success']}, sequence={sequence.get('total_seconds')}, process={process_seconds:.3f}s",
                flush=True,
            )
    report["complete"] = len(report["sequences"]) == c["repeats"] * len(c["methods"]) * len(
        c["warm_starts"]
    )
    report["success"] = report["complete"] and all(row["success"] for row in report["sequences"])
    write_report(output / "results.json", report)
    return report


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_cht")
def main(config: DictConfig):
    if not run(config)["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        parser = argparse.ArgumentParser()
        parser.add_argument("--protocol", required=True, dest="protocol_path")
        parser.add_argument("--output", required=True)
        parser.add_argument("--method", required=True, choices=METHODS)
        parser.add_argument("--warm", required=True, choices=["cold", "outer_inner"])
        worker(**vars(parser.parse_args(sys.argv[2:])))
    else:
        main()
