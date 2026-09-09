"""Public reproduction of matched GPU timings and resource-reuse controls.

Run with: python -m deflation_example.benchmark_gpu output=runs/gpu
No private data, machine paths, cluster settings or other repository is needed.
"""

from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path
import sys
import time
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from threadpoolctl import threadpool_limits
from .amgx import AmgxSession, amgx_cg, amgx_configuration
from .gpu import gpu_deflated_cg, require_cuda
from .problems import laplacian
from .reporting import environment, write_report
from .solvers import calibrate_bound, deflated_cg, pdas
from .spectral import analytical_reference
from .timing import PhaseTimer
from .validation import integer


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def problem_instances(n):
    """The six original moving-Gaussian Laplacian systems, including CPU PDAS."""
    A = laplacian(n)
    H = (sparse.eye(n * n) + 0.001 * (A.T @ A)).tocsr()
    x, y = np.meshgrid(np.arange(1, n + 1) / (n + 1), np.arange(1, n + 1) / (n + 1), indexing="ij")
    x, y = x.ravel(), y.ravel()
    for index, theta in enumerate((0.0, 0.35, 0.7)):
        f = np.exp(
            -((x - 0.5 - 0.15 * np.cos(theta)) ** 2 + (y - 0.5 - 0.15 * np.sin(theta)) ** 2) / 0.025
        )
        if index == 0:
            bound, _ = calibrate_bound(H, f, 0.2, steps=8)
        start = time.perf_counter()
        outer = pdas(H, f, bound, tolerance=1e-8)
        outer_seconds = time.perf_counter() - start
        if outer["status"] != "converged":
            raise RuntimeError("CPU PDAS did not meet the benchmark acceptance criterion")
        I, J = np.flatnonzero(~outer["active"]), np.flatnonzero(outer["active"])
        start = time.perf_counter()
        B = H[I][:, I].tocsr()
        b = f[I] - H[I][:, J] @ np.full(len(J), bound)
        restriction = time.perf_counter() - start
        yield (
            B,
            b,
            I,
            {
                "n": n,
                "index": index,
                "theta": theta,
                "bound": bound,
                "n_inactive": len(I),
                "mask_sha256": digest(I),
                "rhs_sha256": digest(b),
                "matrix_data_sha256": digest(B.data),
                "outer_kkt": outer["kkt"],
                "outer_status": outer["status"],
                "outer_seconds": outer_seconds,
                "restriction_seconds": restriction,
            },
        )


def measured_call(function):
    result, metrics = function()
    return {
        **metrics,
        "iterations": result.iterations,
        "status": result.status,
        "residual": result.residual,
        "rank": result.rank,
        "coarse_condition": result.coarse_condition,
        "fallback_reason": result.fallback_reason,
    }


def representative(runs):
    if not runs or len(runs) % 2 != 1:
        raise ValueError("An odd, nonempty repetition set is required")
    selected = dict(sorted(runs, key=lambda r: r["total_seconds"])[len(runs) // 2])
    selected["all_repetitions_converged"] = all(
        r["status"] == "converged" and np.isfinite(r["residual"]) and r["residual"] <= 1e-10
        for r in runs
    )
    if not selected["all_repetitions_converged"]:
        selected["status"] = "repetition_failed"
    return selected


def run_benchmark(output, grids=(64, 128), rank=100, repeats=5, threads=4, include_cpu=True):
    grids = [integer(n, "Grid", 2) for n in grids]
    integer(rank, "Rank", 1)
    integer(repeats, "Repetitions", 1)
    integer(threads, "Threads", 1)
    if not grids or len(set(grids)) != len(grids) or rank >= min(grids) ** 2 or repeats % 2 != 1:
        raise ValueError("Use distinct grids, rank below each dimension, and odd repetitions")
    if not isinstance(include_cpu, bool):
        raise ValueError("include_cpu must be Boolean")
    torch = require_cuda()
    try:
        import pyamgx
    except ImportError as error:
        raise RuntimeError(
            "Install AmgX and pyamgx as described in docs/gpu-benchmark.md"
        ) from error
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    report = {
        "protocol": "itemized-gpu-resource-control-v1",
        "success": False,
        "controls": {
            "grids": grids,
            "rank": rank,
            "repeats": repeats,
            "threads": threads,
            "rtol": 1e-10,
            "outer_tolerance": 1e-8,
            "maxiter": 10000,
            "jacobi_maxiter": 20000,
            "rank_threshold": 1e-12,
            "coarse_condition_limit": 1e10,
            "initial_guess": "zero",
            "residual_refresh": 1000,
            "hierarchy_reuse": False,
            "include_cpu": include_cpu,
        },
        "amgx_configuration": amgx_configuration(),
        "cases": [],
        "grids": [],
        "timing_scope": "CPU CSR input through CPU residual verification and per-call cleanup; "
        "explicit barriers itemized; reference/restriction/runtime/session costs separate",
        "aggregation": "median-total measured repetition per instance, not a repeated complete sequence",
    }
    initialized = False
    try:
        with threadpool_limits(limits=threads):
            start = time.perf_counter()
            torch.ones(1, device="cuda").sum().item()
            torch.cuda.synchronize()
            report["torch_runtime_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            pyamgx.initialize()
            initialized = True
            torch.cuda.synchronize()
            report["amgx_runtime_seconds"] = time.perf_counter() - start
            report["environment"] = {
                **environment(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "amgx_api": pyamgx.get_api_version(),
                "pyamgx_binary_sha256": hashlib.sha256(
                    Path(pyamgx.__file__).read_bytes()
                ).hexdigest(),
            }
            try:
                report["environment"]["pyamgx"] = importlib.metadata.version("pyamgx")
            except importlib.metadata.PackageNotFoundError:
                report["environment"]["pyamgx"] = "unregistered"
            for n in grids:
                start = time.perf_counter()
                Phi, modes = analytical_reference(n, 2, rank)
                grid = {
                    "n": n,
                    "reference_seconds": time.perf_counter() - start,
                    "reference_sha256": digest(Phi),
                    "mode_indices": modes,
                    "warmups": {},
                }
                report["grids"].append(grid)
                # Session ownership spans this grid, but no matrix/hierarchy does.
                session = AmgxSession(pyamgx)
                timer = PhaseTimer()
                session.open()
                timer.mark("resource_creation")
                timer.synchronize(torch.cuda.synchronize)
                grid["persistent_session_startup"] = timer.finish()
                try:
                    for B, b, I, case in problem_instances(n):
                        start = time.perf_counter()
                        Z = Phi[I, :]
                        case["basis_restriction_seconds"] = time.perf_counter() - start
                        methods = {
                            "deflated_gpu_qr": lambda: gpu_deflated_cg(
                                B, b, Z, maxiter=10000, basis_backend="gpu_qr"
                            ),
                            "deflated_cpu_svd_gpu": lambda: gpu_deflated_cg(
                                B, b, Z, maxiter=10000, basis_backend="cpu_svd"
                            ),
                            "amgx_fresh_resources": lambda: amgx_cg(
                                B, b, api=pyamgx, synchronize=torch.cuda.synchronize
                            ),
                            "amgx_persistent_resources": lambda: amgx_cg(
                                B,
                                b,
                                api=pyamgx,
                                synchronize=torch.cuda.synchronize,
                                session=session,
                            ),
                        }
                        if case["index"] == 0:
                            for name, function in methods.items():
                                grid["warmups"][name] = measured_call(function)
                        case["repetitions"] = {name: [] for name in methods}
                        names = list(methods)
                        for repeat in range(repeats):
                            shift = (repeat + case["index"]) % len(names)
                            for name in names[shift:] + names[:shift]:
                                run = measured_call(methods[name])
                                run["repetition"] = repeat
                                case["repetitions"][name].append(run)
                        case["solvers"] = {
                            name: representative(runs) for name, runs in case["repetitions"].items()
                        }
                        case["solvers"]["jacobi_gpu"] = measured_call(
                            lambda: gpu_deflated_cg(B, b, maxiter=20000)
                        )
                        if include_cpu:
                            start = time.perf_counter()
                            result = deflated_cg(B, b, Z, B.diagonal(), maxiter=10000)
                            case["solvers"]["deflated_cpu"] = {
                                "total_seconds": time.perf_counter() - start,
                                "iterations": result.iterations,
                                "status": result.status,
                                "residual": result.residual,
                                "rank": result.rank,
                                "fallback_reason": result.fallback_reason,
                            }
                        report["cases"].append(case)
                        write_report(output / "results.json", report)
                        print(
                            f"Grid {n}, instance {case['index']}: "
                            + ", ".join(
                                f"{key}={value['status']}" for key, value in case["solvers"].items()
                            ),
                            flush=True,
                        )
                finally:
                    timer = PhaseTimer()
                    session.close()
                    timer.mark("cleanup")
                    timer.synchronize(torch.cuda.synchronize)
                    grid["persistent_session_cleanup"] = timer.finish()
            report["success"] = all(
                r["status"] == "converged" for c in report["cases"] for r in c["solvers"].values()
            ) and all(
                r["status"] == "converged" for g in report["grids"] for r in g["warmups"].values()
            )
    finally:
        if initialized:
            start = time.perf_counter()
            pyamgx.finalize()
            report["amgx_finalize_seconds"] = time.perf_counter() - start
        write_report(output / "results.json", report)
    return report


@dataclass
class BenchmarkConfig:
    grids: list[int]
    rank: int = 100
    repeats: int = 5
    threads: int = 4
    include_cpu: bool = True
    output: str = "???"


@hydra.main(version_base="1.3", config_path="conf", config_name="benchmark_gpu")
def run_config(config: DictConfig):
    checked = OmegaConf.merge(OmegaConf.structured(BenchmarkConfig), config)
    args = OmegaConf.to_container(checked, resolve=True, throw_on_missing=True)
    report = run_benchmark(**args)
    if not report["success"]:
        raise SystemExit(1)


def main():
    if any(
        arg == "-m" or (arg.startswith("--") and len(arg) > 2 and "--multirun".startswith(arg))
        for arg in sys.argv[1:]
    ):
        raise SystemExit("Configuration sweeps are not enabled; invoke benchmarks separately.")
    run_config()


if __name__ == "__main__":
    main()
