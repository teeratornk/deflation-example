"""A complete small CPU optimization sequence, with optional matched GPU kernels."""

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits
from .problems import build_problem, reference_modes
from .solvers import calibrate_bound, deflated_cg, pdas


def environment():
    """Record the installed source, even outside Git or after wheel installation."""
    source = Path(__file__).resolve().parent
    head = None
    candidate = source.parent.parent
    if (candidate / ".git").exists():
        try:
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=candidate, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    blas = [{k: v for k, v in item.items() if k != "filepath"} for item in threadpool_info()]
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "package": importlib.metadata.version("deflation-example"),
        "git_head": head,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))
        },
        "blas": blas,
    }


def json_safe(value):
    """Keep failure records valid JSON without disguising nonfinite quantities."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def write_report(path, report):
    path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")


def summarize_pdas(result, seconds):
    return {
        "status": result["status"],
        "kkt": result["kkt"],
        "objective_reduced": result["objective"],
        "outer_iterations": result["iterations"],
        "inner_iterations": sum(row["linear_iterations"] for row in result["history"]),
        "seconds": seconds,
        "history": result["history"],
    }


def accepted_pdas(result):
    return result["status"] == "converged" and all(
        np.isfinite(v) and v <= 1e-9 for v in result["kkt"].values()
    )


def run_demo(
    output,
    problem="diffusion",
    grid=None,
    rank=20,
    alpha=1e-3,
    device="cpu",
    plot=False,
    threads=1,
    maxiter=20000,
):
    """Run in a new directory; numerical failures are saved and return success=False."""
    if device not in {"cpu", "cuda"} or not isinstance(threads, int) or threads < 1:
        raise ValueError("Use device cpu/cuda and a positive integer thread count")
    if not isinstance(maxiter, int) or maxiter < 0:
        raise ValueError("Iteration cap must be a nonnegative integer")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("Rank must be a positive integer")
    if plot:
        try:
            import matplotlib  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Plotting requested: install deflation-example[plot]") from error
    if device == "cuda":
        from .gpu import require_cuda

        torch = require_cuda()
        torch.set_num_threads(threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "success": False,
        "cases": [],
        "configuration": {
            "problem": problem,
            "grid": grid,
            "rank": rank,
            "alpha": alpha,
            "device": device,
            "threads": threads,
            "maxiter": maxiter,
            "angles": [0.0, float(np.pi / 4), float(np.pi / 2)],
            "kkt_tolerance": 1e-9,
            "kernel_rtol": 1e-10,
            "inner_absolute_budget": 1e-10,
            "activity_target": 0.2,
            "calibration_angle": float(np.pi / 4),
        },
    }
    try:
        with threadpool_limits(limits=threads):
            report["environment"] = environment()
            t = time.perf_counter()
            model = build_problem(problem, grid, alpha)
            report["assembly_seconds"] = time.perf_counter() - t
            report["configuration"]["grid"] = model.n
            t = time.perf_counter()
            reference = reference_modes(model, rank)
            report["reference_seconds"] = time.perf_counter() - t
            report["reference_type"] = (
                "frozen full thermal eigenspace"
                if problem == "thermal"
                else "analytical homogeneous Laplacian modes"
            )
            t = time.perf_counter()
            bound, activity = calibrate_bound(model.H, model.target(np.pi / 4), 0.2)
            report.update(
                bound=bound,
                calibrated_activity=activity,
                calibration_seconds=time.perf_counter() - t,
            )
            if device == "cuda":
                t = time.perf_counter()
                torch.ones(1, device="cuda").sum().item()
                torch.cuda.synchronize()
                report["gpu_warmup_seconds"] = time.perf_counter() - t
                report["environment"].update(
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0),
                    gpu_capacity_bytes=torch.cuda.get_device_properties(0).total_memory,
                )
            for index, theta in enumerate(report["configuration"]["angles"]):
                f = model.target(theta)
                t = time.perf_counter()
                direct = pdas(model.H, f, bound)
                direct_seconds = time.perf_counter() - t
                t = time.perf_counter()
                accelerated = pdas(
                    model.H,
                    f,
                    bound,
                    linear_solver=lambda B, b, I: deflated_cg(
                        B,
                        b,
                        reference[I],
                        B.diagonal(),
                        rtol=min(1e-10, 1e-10 / max(np.linalg.norm(b), 1)),
                        maxiter=maxiter,
                    ),
                )
                row = {
                    "theta": theta,
                    "direct": summarize_pdas(direct, direct_seconds),
                    "deflated": summarize_pdas(accelerated, time.perf_counter() - t),
                    "kernels": {},
                }
                objective_error = abs(accelerated["objective"] - direct["objective"])
                row.update(
                    objective_difference=objective_error,
                    state_relative_error=float(
                        np.linalg.norm(accelerated["y"] - direct["y"])
                        / max(np.linalg.norm(direct["y"]), 1e-300)
                    ),
                    active_fraction=float(direct["active"].mean()),
                )
                row["optimization_passed"] = (
                    accepted_pdas(direct)
                    and accepted_pdas(accelerated)
                    and objective_error <= 1e-9 * max(1.0, abs(direct["objective"]))
                )
                # The mask belongs to a verified direct solve, never to a failed optimizer.
                if accepted_pdas(direct):
                    I, J = np.flatnonzero(~direct["active"]), np.flatnonzero(direct["active"])
                    B = model.H[I][:, I].tocsr()
                    b = f[I] - model.H[I][:, J] @ np.full(len(J), bound)
                    if len(I):
                        for label, basis in [("jacobi", None), ("deflated", reference[I])]:
                            t = time.perf_counter()
                            result = deflated_cg(B, b, basis, B.diagonal(), maxiter=maxiter)
                            timing = {"total_seconds": time.perf_counter() - t}
                            values = asdict(result)
                            values.pop("x")
                            row["kernels"][label + "_cpu"] = {**values, **timing}
                            if device == "cuda":
                                from .gpu import gpu_deflated_cg

                                result, timing = gpu_deflated_cg(
                                    B, b, basis, B.diagonal(), maxiter=maxiter
                                )
                                values = asdict(result)
                                values.pop("x")
                                row["kernels"][label + "_cuda"] = {**values, **timing}
                    else:
                        row["kernel_note"] = "All nodes active: no inactive system to solve"
                row["kernels_passed"] = all(
                    v["status"] == "converged"
                    and np.isfinite(v["residual"])
                    and v["residual"] <= 1e-10
                    for v in row["kernels"].values()
                )
                row["success"] = row["optimization_passed"] and row["kernels_passed"]
                np.savez_compressed(
                    output / f"fields-{index}.npz",
                    desired=f,
                    state=direct["y"],
                    deflated_state=accelerated["y"],
                    multiplier=direct["multiplier"],
                    active=direct["active"],
                    control=model.A @ direct["y"],
                    n=model.n,
                    dimension=model.dim,
                    bound=bound,
                    direct_status=direct["status"],
                    deflated_status=accelerated["status"],
                )
                report["cases"].append(row)
                write_report(output / "results.json", report)
                print(
                    f"{problem} angle={theta:.3f}: PDAS {direct['status']}/{accelerated['status']}, "
                    f"KKT={max(accelerated['kkt'].values()):.2e}, passed={row['success']}",
                    flush=True,
                )
            report["success"] = all(row["success"] for row in report["cases"])
            report["pdas_sequence_seconds"] = {
                "direct": sum(r["direct"]["seconds"] for r in report["cases"]),
                "deflated_including_reference": report["reference_seconds"]
                + sum(r["deflated"]["seconds"] for r in report["cases"]),
            }
            write_report(output / "results.json", report)
            write_csv(output, report)
            if plot:
                from .plotting import plot_results

                plot_results(output, report)
        return report
    except Exception as error:
        report["success"] = False
        # Exception text may contain installation paths; keep the saved record portable.
        report["error"] = {
            "type": type(error).__name__,
            "message": "Execution failed; see the command output for details",
        }
        write_report(output / "results.json", report)
        raise


def write_csv(output, report):
    import csv

    with (output / "kernels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "theta",
                "method",
                "status",
                "iterations",
                "residual",
                "rank",
                "total_seconds",
                "fallback_reason",
            ],
        )
        writer.writeheader()
        for row in report["cases"]:
            for method, values in row["kernels"].items():
                writer.writerow(
                    {
                        "theta": row["theta"],
                        "method": method,
                        **{k: values[k] for k in writer.fieldnames if k not in {"theta", "method"}},
                    }
                )
