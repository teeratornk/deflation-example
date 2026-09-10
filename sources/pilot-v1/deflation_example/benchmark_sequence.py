"""Repeated complete heterogeneous CHT PDAS sequences on matched GPU hardware."""

import argparse
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .amgx import AmgxSession, amgx_cg, amgx_configuration
from .gpu import gpu_deflated_cg, require_cuda
from .problems import build_problem
from .reporting import environment, summarize_pdas, write_report
from .solvers import calibrate_bound, pdas
from .spectral import analytical_reference
from .validation import integer, positive_real


def complete_sequence(n, rank, bound, method, torch, api, rtol, outer_tolerance):
    """Time assembly through final verification and cleanup, without a direct inner solve."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    problem = build_problem("cht", n)
    assembly_seconds = time.perf_counter() - start
    session, Phi = None, None
    reference_seconds, resource_seconds = 0.0, 0.0
    cases = []
    try:
        tick = time.perf_counter()
        if method == "deflated_gpu_qr":
            Phi, _ = analytical_reference(n, 3, rank)
            reference_seconds = time.perf_counter() - tick
        elif method == "amgx_persistent_resources":
            session = AmgxSession(api, rtol=rtol).open()
            resource_seconds = time.perf_counter() - tick
        else:
            raise ValueError("Unknown sequence method")
        # Independent full sequences start with no active DOFs for each target.
        for theta in (0.0, np.pi / 4, np.pi / 2):
            tick = time.perf_counter()
            f = problem.target(theta)
            inner = []

            def solve(B, b, I):
                begin = time.perf_counter()
                if Phi is not None:
                    result, metrics = gpu_deflated_cg(
                        B, b, Phi[I], B.diagonal(), rtol=rtol, maxiter=10000, basis_backend="gpu_qr"
                    )
                else:
                    result, metrics = amgx_cg(
                        B,
                        b,
                        api=api,
                        synchronize=torch.cuda.synchronize,
                        session=session,
                        rtol=rtol,
                    )
                inner.append(
                    {
                        **metrics,
                        "callback_seconds": time.perf_counter() - begin,
                        "n_inactive": len(I),
                        "iterations": result.iterations,
                        "rank": result.rank,
                        "original_residual": result.residual,
                        "status": result.status,
                        "coarse_condition": result.coarse_condition,
                        "fallback_reason": result.fallback_reason,
                    }
                )
                return result

            outer = pdas(problem.H, f, bound, tolerance=outer_tolerance, linear_solver=solve)
            case = {
                "theta": theta,
                **summarize_pdas(outer, time.perf_counter() - tick),
                "inner": inner,
            }
            cases.append(case)
            if outer["status"] != "converged":
                # A failure is retained and aborts only this method's sequence.
                break
    finally:
        tick = time.perf_counter()
        if session is not None:
            session.close()
        Phi = None
        torch.cuda.synchronize()
        cleanup_seconds = time.perf_counter() - tick
    total = time.perf_counter() - start
    return {
        "method": method,
        "total_seconds": total,
        "assembly_seconds": assembly_seconds,
        "reference_seconds": reference_seconds,
        "resource_creation_seconds": resource_seconds,
        "cleanup_seconds": cleanup_seconds,
        "cases": cases,
        "outer_iterations": sum(c["outer_iterations"] for c in cases),
        "inner_iterations": sum(c["inner_iterations"] for c in cases),
        "success": len(cases) == 3 and all(c["status"] == "converged" for c in cases),
    }


def run(output, n=24, rank=100, repeats=5, rtol=1e-10, outer_tolerance=1e-8, threads=4):
    for value, name in [
        (n, "Grid"),
        (rank, "Rank"),
        (repeats, "Repetitions"),
        (threads, "Threads"),
    ]:
        integer(value, name, 1)
    positive_real(rtol, "Inner tolerance")
    positive_real(outer_tolerance, "Outer tolerance")
    if n < 2 or rank >= n**3:
        raise ValueError("Rank must be below the full dimension")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch = require_cuda()
    import pyamgx

    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        start = time.perf_counter()
        calibration_problem = build_problem("cht", n)
        bound, fraction = calibrate_bound(
            calibration_problem.H, calibration_problem.target(np.pi / 4), 0.2, steps=8
        )
        calibration_seconds = time.perf_counter() - start
        del calibration_problem
        start = time.perf_counter()
        torch.ones(1, device="cuda").sum().item()
        pyamgx.initialize()
        torch.cuda.synchronize()
        runtime_seconds = time.perf_counter() - start
        report = {
            "protocol": "complete-cht-pdas-v1",
            "n": n,
            "rank": rank,
            "repeats": repeats,
            "rtol": rtol,
            "outer_tolerance": outer_tolerance,
            "bound": bound,
            "calibrated_activity": fraction,
            "calibration_seconds": calibration_seconds,
            "shared_runtime_seconds": runtime_seconds,
            "initial_active": "empty for each target",
            "timing_scope": "each independent repetition includes problem assembly, reference or resource "
            "creation, all PDAS steps and inner solves, transfers, checks and cleanup; "
            "common bound calibration and process runtime initialization reported separately",
            "hierarchy_reused": False,
            "configuration": amgx_configuration(rtol),
            "environment": {
                **environment(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
            "sequences": [],
        }
        try:
            for repeat in range(repeats):
                methods = ["deflated_gpu_qr", "amgx_persistent_resources"]
                if repeat % 2:
                    methods.reverse()
                for method in methods:
                    sequence = complete_sequence(
                        n, rank, bound, method, torch, pyamgx, rtol, outer_tolerance
                    )
                    report["sequences"].append({"repetition": repeat, **sequence})
                    write_report(output / "results.json", report)
                    print(
                        f"{method}, repeat {repeat}: success={sequence['success']}, "
                        f"seconds={sequence['total_seconds']:.4f}, "
                        f"outer={sequence['outer_iterations']}, inner={sequence['inner_iterations']}",
                        flush=True,
                    )
            report["success"] = all(s["success"] for s in report["sequences"])
        finally:
            pyamgx.finalize()
            write_report(output / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--rank", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--outer-tolerance", type=float, default=1e-8)
    parser.add_argument("--threads", type=int, default=4)
    report = run(**vars(parser.parse_args()))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
