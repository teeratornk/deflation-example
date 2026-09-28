"""Matched CPU/CUDA actions on a checksum-bound coupled inactive system.

Two warmups precede five repetitions. Device intervals and synchronized wall
intervals describe the same action; they must not be added together. This is an
operator profile, not a completed-solve or nonlinear optimization comparison.
"""

import argparse
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_optimize import equation_acceptance, load_problem
from .coupled_retention_replay import rebuild
from .coupled_trace import read_arrays, read_manifest
from .reporting import environment, file_sha256, write_report


def measured_action(action, *, cp=None):
    for _ in range(2):
        action()
    if cp is not None:
        cp.cuda.get_current_stream().synchronize()
    rows = []
    for _ in range(5):
        events = None if cp is None else (cp.cuda.Event(), cp.cuda.Event())
        start = time.perf_counter()
        if events is not None:
            events[0].record()
        result = action()
        if events is not None:
            events[1].record()
            events[1].synchronize()
        rows.append(
            {
                "wall_seconds": time.perf_counter() - start,
                "device_seconds": None
                if events is None
                else cp.cuda.get_elapsed_time(*events) / 1000,
            }
        )
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quadratic", type=int, default=2)
    args = parser.parse_args()
    import cupy as cp
    from .coupled_cuda import CudaControlJacobian, CudaGaussNewton
    from .coupled_cuda_preconditioner import CudaFrozenSweepPreconditioner
    from .coupled_frozen_preconditioner import frozen_preconditioner_factory

    manifest = read_manifest(args.trace)
    row = manifest["quadratics"][args.quadratic]
    system = next(s for s in manifest["systems"] if s["quadratic"] == args.quadratic)
    arrays = read_arrays(args.trace, row["file"], row["sha256"])
    data = read_arrays(args.trace, system["file"], system["sha256"])
    indices = data["indices"]
    cfg = {**manifest["configuration"], "baseline_directory": str(args.baseline)}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-device-profile-v1",
        "status": "running",
        "environment": environment(),
        "trace_sha256": file_sha256(args.trace / "manifest.json"),
        "quadratic_sha256": row["sha256"],
        "system_sha256": system["sha256"],
        "components": [],
        "warmups": 2,
        "repetitions": 5,
        "scope": "Independent operator intervals; not additive sequence components or solve speedups.",
    }
    write_report(args.output / "record.json", report)
    resources = []
    start = time.perf_counter()
    try:
        with threadpool_limits(cfg["threads"]):
            problem, baseline = load_problem(cfg)
            if baseline["baseline_sha256"] != manifest["baseline_sha256"]:
                raise ValueError("The physical baseline differs")
            evaluation, H, diagonal, gradient = rebuild(problem, arrays, row)
            checks = problem.verify(evaluation)
            if not equation_acceptance(checks, cfg):
                raise ValueError("Reconstructed coupled equations fail verification")
            rng = np.random.default_rng(20260928)
            x = rng.standard_normal(problem.size)
            r = rng.standard_normal(len(indices))
            device_x, device_r = cp.asarray(x), cp.asarray(r)
            tick = time.perf_counter()
            J = CudaControlJacobian(evaluation.jacobian)
            resources.append(J)
            normal = CudaGaussNewton(
                J,
                problem.weights,
                problem.alpha,
                row["damping"],
                corrections=getattr(H, "corrections", ()),
            )
            host_P = frozen_preconditioner_factory(
                problem, evaluation, row["damping"], cfg["frozen_sweeps"]
            )(indices)
            device_P = CudaFrozenSweepPreconditioner(host_P)
            resources.append(device_P)
            grouped_P = CudaFrozenSweepPreconditioner(host_P, layout="block_diagonal")
            resources.append(grouped_P)
            report.update(
                setup_seconds=time.perf_counter() - tick,
                dimension=problem.size,
                inactive_dofs=len(indices),
                gpu=cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())[
                    "name"
                ].decode(),
            )
            for name, cpu_action, gpu_actions in (
                ("tangent", lambda: evaluation.jacobian @ x, [("cuda", lambda: J.apply(device_x))]),
                (
                    "transpose",
                    lambda: evaluation.jacobian.T @ x,
                    [("cuda", lambda: J.apply(device_x, transpose=True))],
                ),
                (
                    "restricted_normal",
                    lambda: H.restrict(indices) @ r,
                    [("cuda", lambda: normal.apply(device_r, indices))],
                ),
                (
                    "frozen_inverse",
                    lambda: host_P(r),
                    [
                        ("cuda_serial", lambda: device_P(device_r)),
                        ("cuda_block_diagonal", lambda: grouped_P(device_r)),
                    ],
                ),
            ):
                expected, timings = measured_action(cpu_action)
                report["components"].append({"action": name, "backend": "cpu", "timings": timings})
                write_report(args.output / "record.json", report)
                for backend, action in gpu_actions:
                    actual, timings = measured_action(action, cp=cp)
                    difference = float(
                        np.linalg.norm(cp.asnumpy(actual) - expected)
                        / max(np.linalg.norm(expected), np.finfo(float).tiny)
                    )
                    report["components"].append(
                        {
                            "action": name,
                            "backend": backend,
                            "timings": timings,
                            "relative_difference": difference,
                            "verified": difference <= 1e-8,
                        }
                    )
                    write_report(args.output / "record.json", report)
            _, transfers = measured_action(lambda: cp.asnumpy(cp.asarray(r)), cp=cp)
            report["round_trip_transfer"] = transfers
            report["status"] = (
                "verified"
                if all(c.get("verified", True) for c in report["components"])
                else "action_mismatch"
            )
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for resource in reversed(resources):
            resource.close()
        report["seconds"] = time.perf_counter() - start
        write_report(args.output / "record.json", report)


if __name__ == "__main__":
    main()
