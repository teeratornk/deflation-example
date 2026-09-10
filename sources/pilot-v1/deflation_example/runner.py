"""A complete small CPU optimization sequence, with optional matched GPU kernels."""

from dataclasses import asdict
from collections.abc import Mapping
from pathlib import Path
import re
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .problems import build_problem, reference_modes
from .solvers import calibrate_bound, deflated_cg, independent_residual, pdas
from .reporting import environment, summarize_pdas, write_csv, write_fields, write_report
from .backends import LinearKernel, default_kernels
from .validation import integer, real_array


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
    kernels: Mapping[str, LinearKernel] | None = None,
):
    """Run in a new directory; numerical failures are saved and return success=False."""
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        raise ValueError("Use device cpu/cuda and a positive integer thread count")
    threads = integer(threads, "Thread count", 1)
    maxiter = integer(maxiter, "Iteration cap")
    rank = integer(rank, "Reference rank", 1)
    if not isinstance(plot, bool):
        raise ValueError("Plot must be Boolean")
    kernels = dict(default_kernels(device) if kernels is None else kernels)
    if not kernels or any(
        not isinstance(k, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", k) or not callable(v)
        for k, v in kernels.items()
    ):
        raise ValueError("Kernels need nonempty lowercase names and callable implementations")
    if plot:
        try:
            import matplotlib  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Plotting requested: install deflation-example[plot]") from error
    if device == "cuda":
        from .gpu import require_cuda

        torch = require_cuda()
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
            "plot": plot,
            "kernel_backends": list(kernels),
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
                            for backend, solve in kernels.items():
                                result, timing = solve(
                                    B, b, basis, B.diagonal(), rtol=1e-10, maxiter=maxiter
                                )
                                solution = real_array(result.x, "Kernel solution")
                                if solution.shape != b.shape:
                                    raise ValueError("Kernel returned an incompatible solution")
                                values = asdict(result)
                                values.pop("x")
                                integer(values["iterations"], "Kernel iteration count")
                                integer(values["rank"], "Kernel rank")
                                values["residual"] = independent_residual(B, solution, b)
                                if values["status"] == "converged" and values["residual"] > 1e-10:
                                    values["status"] = "residual_failed"
                                if "total_seconds" not in timing or any(
                                    not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0
                                    for v in timing.values()
                                ):
                                    raise ValueError(
                                        "Kernel timings must be finite and nonnegative"
                                    )
                                if values.keys() & timing.keys():
                                    raise ValueError(
                                        "Kernel timing fields must not override solver results"
                                    )
                                row["kernels"][label + "_" + backend] = {**values, **timing}
                    else:
                        row["kernel_note"] = "All nodes active: no inactive system to solve"
                row["kernels_passed"] = all(
                    v["status"] == "converged"
                    and np.isfinite(v["residual"])
                    and v["residual"] <= 1e-10
                    for v in row["kernels"].values()
                )
                row["success"] = row["optimization_passed"] and row["kernels_passed"]
                write_fields(
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
        try:
            write_report(output / "results.json", report)
        except OSError:
            pass  # Preserve the original failure if the output device also failed.
        raise
