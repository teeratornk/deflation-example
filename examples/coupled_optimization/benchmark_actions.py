"""Compare CPU and CUDA normal products at one saved coupled iterate.

These are operator measurements, separate from complete optimization timings.
Both backends apply the same fixed derivative factors and original equations.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from deflation_example.axisymmetric_flow import FlowResult
from deflation_example.coupled_cuda import CudaControlJacobian, CudaGaussNewton
from deflation_example.coupled_derivatives import GaussNewtonOperator
from deflation_example.coupled_optimize import load_problem
from deflation_example.reporting import environment, write_report
from deflation_example.validation import integer


def main():
    import cupy as cp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--method", default="jacobi")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, nargs="+", default=[1, 20, 100])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--ordering", choices=("COLAMD", "MMD_AT_PLUS_A", "MMD_ATA"), default="COLAMD"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    columns_requested = [integer(c, "Column count", 1) for c in args.columns]
    if len(set(columns_requested)) != len(columns_requested):
        raise ValueError("Column counts must be distinct")
    warmups = integer(args.warmups, "Warmup count", 1)
    repeats = integer(args.repeats, "Repetition count", 1)
    with threadpool_limits(integer(args.threads, "Thread count", 1)):
        record = json.loads((args.run / "record.json").read_text())
        problem, _ = load_problem(
            {**record["configuration"], "baseline_directory": str(args.baseline)}
        )
        field = args.run / (args.method + "-fields.npz")
        with np.load(field, allow_pickle=False) as data:
            state = data["state"].copy()
            flows = tuple(
                FlowResult(v.copy(), p.copy(), "initial", [])
                for v, p in zip(data["velocity"], data["pressure"], strict=True)
            )
        evaluation = problem.evaluate(state, initial=SimpleNamespace(flows=flows))
        tick = time.perf_counter()
        factors = []
        for n, flow in enumerate(evaluation.flows):
            dt = float(problem.physical_steps[n]) if len(problem.physical_steps) else None
            matrix = problem.flow.operator(flow.velocity, time_step=dt)
            matrix += problem.flow.convection_derivative(flow.velocity)
            matrix = matrix[problem.flow_free][:, problem.flow_free].tocsc()
            factors.append(splu(matrix, permc_spec=args.ordering))
        factorization = time.perf_counter() - tick
        evaluation.jacobian.factors = tuple(factors)
        H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
        cp.cuda.get_current_stream().synchronize()
        tick = time.perf_counter()
        derivative = CudaControlJacobian(evaluation.jacobian)
        gpu = CudaGaussNewton(derivative, problem.weights, problem.alpha)
        cp.cuda.get_current_stream().synchronize()
        upload = time.perf_counter() - tick
        rng = np.random.default_rng(782)
        rows = []
        metadata = {
            "environment": environment(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "columns": columns_requested,
            "warmups": warmups,
            "repeats": repeats,
            "cupy": cp.__version__,
            "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "field_sha256": hashlib.sha256(field.read_bytes()).hexdigest(),
            "factor_ordering": args.ordering,
            "factor_assembly_and_construction_seconds": factorization,
            "factor_nonzeros": [f.L.nnz + f.U.nnz for f in factors],
            "derivative_upload_seconds": upload,
            "derivative_device_bytes": derivative.storage_bytes(),
            "scope": "Fixed normal products with the recorded matched warmups and repetitions. CPU momentum factorization and complete optimization are outside these intervals.",
        }
        write_report(args.output, {**metadata, "status": "running", "rows": rows})
        for columns in columns_requested:
            x = rng.normal(size=(problem.size, columns))
            device_x = cp.asarray(x)
            # Consistent warmups precede each block-size measurement.
            for _ in range(warmups):
                H @ x
                gpu.apply(device_x)
            cp.cuda.get_current_stream().synchronize()
            cpu_times, gpu_times = [], []
            for _ in range(repeats):
                tick = time.perf_counter()
                expected = H @ x
                cpu_times.append(time.perf_counter() - tick)
                tick = time.perf_counter()
                actual = gpu.apply(device_x)
                cp.cuda.get_current_stream().synchronize()
                gpu_times.append(time.perf_counter() - tick)
            error = float(np.linalg.norm(cp.asnumpy(actual) - expected) / np.linalg.norm(expected))
            rows.append(
                {
                    "columns": columns,
                    "cpu_seconds": cpu_times,
                    "cuda_seconds": gpu_times,
                    "relative_action_error": error,
                }
            )
            write_report(args.output, {**metadata, "status": "running", "rows": rows})
        derivative.close()
        write_report(args.output, {**metadata, "status": "complete", "rows": rows})


if __name__ == "__main__":
    main()
