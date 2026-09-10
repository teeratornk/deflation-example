"""Manufactured matrix-free GPU kernels with rank-zero and spectral controls."""

import argparse
from pathlib import Path
import resource
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .benchmark_extended import digest
from .gpu import orthonormalize_gpu, require_cuda
from .problems import laplacian
from .reporting import environment, write_report
from .solvers import independent_residual, restricted_normal_operator
from .spectral import lowest_modes


def stencil_operator(n, indices, torch, alpha=0.001):
    N = n**3

    def L(v):
        a = v.reshape(n, n, n, -1)
        out = 6 * a.clone()
        out[1:] -= a[:-1]
        out[:-1] -= a[1:]
        out[:, 1:] -= a[:, :-1]
        out[:, :-1] -= a[:, 1:]
        out[:, :, 1:] -= a[:, :, :-1]
        out[:, :, :-1] -= a[:, :, 1:]
        return out.reshape(N, -1) * (n + 1) ** 2

    def apply(v):
        V = v[:, None] if v.ndim == 1 else v
        full = torch.zeros((N, V.shape[1]), device=indices.device, dtype=torch.float64)
        full[indices] = V
        out = V + alpha * L(L(full))[indices]
        return out[:, 0] if v.ndim == 1 else out

    neighbors = torch.full((n, n, n), 6.0, dtype=torch.float64, device=indices.device)
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = 0, -1
        neighbors[tuple(lo)] -= 1
        neighbors[tuple(hi)] -= 1
    diagonal = (1 + alpha * (36 + neighbors.reshape(-1)) * (n + 1) ** 4)[indices]
    return apply, diagonal


def solve(n, rank, maxiter=10000, rtol=1e-10):
    torch = require_cuda()
    torch.manual_seed(20260908)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = tick = time.perf_counter()
    N = n**3
    I = torch.arange((n // 5) * n * n, N, device="cuda")
    B, d = stencil_operator(n, I, torch)
    coordinates = torch.arange(1, n + 1, device="cuda", dtype=torch.float64) / (n + 1)
    modes = lowest_modes(n, 3, rank)
    Z = torch.empty((len(I), rank), device="cuda", dtype=torch.float64)
    for column, mode in enumerate(modes):
        a, b, c = [torch.sin(np.pi * k * coordinates) for k in mode]
        Z[:, column] = torch.einsum("i,j,k->ijk", a, b, c).reshape(-1)[I]
    Z = orthonormalize_gpu(Z, torch)
    orthogonalized_rank = Z.shape[1]
    torch.cuda.synchronize()
    construction = time.perf_counter() - tick
    tick = time.perf_counter()
    condition, fallback, factor = 1.0, None, None
    if Z.shape[1]:
        AZ = B(Z)
        E = Z.T @ AZ
        E = (E + E.T) / 2
        eig = torch.linalg.eigvalsh(E)
        condition = float(eig[-1] / eig[0]) if float(eig[0]) > 0 else float("inf")
        if not np.isfinite(condition) or condition > 1e10:
            Z = Z[:, :0]
            fallback = "coarse_condition_limit"
        else:
            factor = torch.linalg.cholesky(E)
        del AZ, E
    deployed_rank = Z.shape[1]

    def Q(v):
        return (
            Z @ torch.cholesky_solve((Z.T @ v)[:, None], factor)[:, 0]
            if deployed_rank
            else torch.zeros_like(v)
        )

    def precondition(r):
        z = r / d
        return z - Q(B(z)) if deployed_rank else z

    truth = torch.randn(len(I), device="cuda", dtype=torch.float64)
    rhs = B(truth)
    x = Q(rhs)
    r = rhs - B(x)
    target = rtol * float(torch.linalg.vector_norm(rhs))
    z = precondition(r)
    p, rho = z.clone(), torch.dot(r, z)
    torch.cuda.synchronize()
    setup = time.perf_counter() - tick
    tick = time.perf_counter()
    status, iterations = "maxiter", 0
    if float(torch.linalg.vector_norm(r)) <= target:
        status = "converged"
    else:
        for k in range(maxiter):
            Bp = B(p)
            curvature = torch.dot(p, Bp)
            if (
                float(curvature) <= 0
                or float(rho) <= 0
                or not bool(torch.isfinite(curvature + rho))
            ):
                status = "breakdown"
                break
            step = rho / curvature
            x += step * p
            r -= step * Bp
            iterations = k + 1
            restart = iterations % 1000 == 0 or float(torch.linalg.vector_norm(r)) <= target
            if restart:
                r = rhs - B(x)
                if float(torch.linalg.vector_norm(r)) <= target:
                    status = "converged"
                    break
                x += Q(r)
                r = rhs - B(x)
                if float(torch.linalg.vector_norm(r)) <= target:
                    status = "converged"
                    break
            z = precondition(r)
            next_rho = torch.dot(r, z)
            p = z.clone() if restart else z + (next_rho / rho) * p
            rho = next_rho
    torch.cuda.synchronize()
    iteration_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    gpu_residual = float(torch.linalg.vector_norm(rhs - B(x)) / torch.linalg.vector_norm(rhs))
    error = float(torch.linalg.vector_norm(x - truth) / torch.linalg.vector_norm(truth))
    x_cpu, b_cpu = x.cpu().numpy(), rhs.cpu().numpy()
    truth_hash = digest(truth.cpu().numpy())
    indices = I.cpu().numpy()
    # Separate CPU sparse operator and transpose independently check every size.
    cpu = restricted_normal_operator(laplacian(n, 3), 0.001, indices)
    cpu_residual = independent_residual(cpu, x_cpu, b_cpu)
    if status == "converged" and max(gpu_residual, cpu_residual) > rtol:
        status = "residual_failed"
    verification_seconds = time.perf_counter() - tick
    peak = torch.cuda.max_memory_allocated()
    tick = time.perf_counter()
    x = rhs = truth = Z = factor = p = r = z = rho = d = B = I = coordinates = cpu = None
    torch.cuda.synchronize()
    cleanup = time.perf_counter() - tick
    total = time.perf_counter() - start
    return {
        "n": n,
        "full_dof": N,
        "inactive_dof": len(indices),
        "rank": rank,
        "orthogonalized_rank": orthogonalized_rank,
        "deployed_rank": deployed_rank,
        "mode_indices": modes,
        "coarse_condition": condition,
        "fallback_reason": fallback,
        "iterations": iterations,
        "status": status,
        "gpu_original_residual": gpu_residual,
        "cpu_original_residual": cpu_residual,
        "solution_relative_error": error,
        "truth_sha256": truth_hash,
        "rhs_sha256": digest(b_cpu),
        "inactive_sha256": digest(indices),
        "construction_seconds": construction,
        "setup_seconds": setup,
        "iteration_seconds": iteration_seconds,
        "verification_seconds": verification_seconds,
        "cleanup_seconds": cleanup,
        "total_seconds": total,
        "peak_torch_bytes": peak,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 64, 102, 128])
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 20, 200])
    parser.add_argument("--maxiter", type=int, default=10000)
    args = parser.parse_args()
    if args.maxiter < 1 or any(n < 4 for n in args.grids) or any(r < 0 for r in args.ranks):
        parser.error("Invalid grid, rank or iteration limit")
    if any(r >= n**3 - (n // 5) * n * n for r in args.ranks for n in args.grids):
        parser.error("Rank exceeds the inactive dimension")
    args.output.mkdir(parents=True, exist_ok=False)
    specification = {
        "protocol": "matrix-free-rank-controls-v1",
        "grids": args.grids,
        "ranks": args.ranks,
        "maxiter": args.maxiter,
        "rtol": 1e-10,
        "alpha": 0.001,
        "seed": 20260908,
        "refresh": 1000,
        "rank_threshold": 1e-12,
        "coarse_limit": 1e10,
        "repetitions": 1,
        "initial_guess": "zero before coarse correction",
        "scope": "manufactured solutions and prescribed plane mask; no optimization claim",
        "timing_scope": "construction through independent CPU verification and cleanup; initialization separately",
        "mask": "first floor(n/5) x1-slabs active",
        "rank_order": "ascending for every grid",
    }
    write_report(args.output / "protocol.json", specification)
    report = {
        "specification": specification,
        "environment": environment(),
        "rows": [],
        "complete": False,
    }
    torch = require_cuda()
    torch.set_num_threads(4)
    tick = time.perf_counter()
    torch.ones(1, device="cuda").sum().item()
    torch.cuda.synchronize()
    report["runtime_initialization_seconds"] = time.perf_counter() - tick
    report["environment"].update(
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()
    )
    with threadpool_limits(limits=4):
        for n in args.grids:
            for rank in args.ranks:
                try:
                    row = solve(n, rank, args.maxiter)
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                    row = {
                        "n": n,
                        "rank": rank,
                        "status": "exception",
                        "error_type": type(error).__name__,
                    }
                report["rows"].append(row)
                write_report(args.output / "results.json", report)
                print(f"n={n}, rank={rank}: {row['status']}", flush=True)
        report["complete"] = True
        report["success"] = all(r["status"] == "converged" for r in report["rows"])
        write_report(args.output / "results.json", report)
    # A cap or failure is an experimental outcome and remains in the report.


if __name__ == "__main__":
    main()
