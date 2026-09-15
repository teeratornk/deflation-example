"""Measure one reference construction in a fresh CPU process before optimization.

The stored result is a setup and sampled-memory measurement. It does not measure
complete optimization, restricted-basis deployment or the coupled flow factors.
"""

import argparse
import gc
from pathlib import Path
import time

from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from .coupled_optimize import load_problem
from .coupled_reference import configured_reference
from .memory import ProcessMemory
from .reporting import environment, write_fields, write_report
from .validation import integer, positive_real


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--coarse-baseline", type=Path)
    parser.add_argument("--slabs", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--horizon", type=float, default=600.0)
    parser.add_argument("--alpha", type=float, default=1e-14)
    parser.add_argument("--temporal-solver", choices=("dense", "tridiagonal"), required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(
        OmegaConf.load(Path(__file__).parent / "conf/coupled_optimize.yaml")
    )
    cfg.update(
        baseline_directory=str(args.baseline),
        reference_baseline_directory=None
        if args.coarse_baseline is None
        else str(args.coarse_baseline),
        transient=True,
        slabs=integer(args.slabs, "Time slabs", 1),
        rank=integer(args.rank, "Rank", 1),
        horizon_s=positive_real(args.horizon, "Physical horizon"),
        alpha=positive_real(args.alpha, "Regularization"),
        threads=integer(args.threads, "Threads", 1),
        reference_temporal_solver=args.temporal_solver,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-reference-construction-screen-v1",
        "environment": environment(),
        "configuration": {
            key: value
            for key, value in cfg.items()
            if key not in {"baseline_directory", "reference_baseline_directory", "output"}
        },
        "scope": "One fresh-process reference construction; includes first-use costs. Restricted bases, flow linearizations and complete optimization are not measured.",
        "status": "running",
    }
    write_report(args.output / "record.json", report)
    failure, reference = None, None
    with threadpool_limits(cfg["threads"]):
        sampler = ProcessMemory("cpu", 0.01)
        try:
            sampler.start()
            tick = time.perf_counter()
            problem, baseline = load_problem(cfg)
            report["problem_assembly_seconds"] = time.perf_counter() - tick
            report["baseline_sha256"] = baseline["baseline_sha256"]
            report["spatial_state_degrees_of_freedom"] = problem.spatial_size
            report["trajectory_state_degrees_of_freedom"] = problem.size
            tick = time.perf_counter()
            reference = configured_reference(problem, cfg, baseline)
            report["reference_construction_seconds"] = time.perf_counter() - tick
            report["reference_storage"] = reference.storage()
            report["reference"] = reference.description
            report["deployed_reference_rank"] = reference.rank
            # Release the problem before writing the compact fields; serialization
            # remains outside the measured construction interval.
            problem = None
            gc.collect()
        except Exception as error:
            failure = type(error).__name__
        try:
            report["sampled_process_memory"] = sampler.finish()
        except Exception as error:
            failure = failure or type(error).__name__
            report["sampled_process_memory"] = {
                "complete": False,
                "error_type": type(error).__name__,
            }
    if reference is not None:
        write_fields(
            args.output / "reference.npz",
            spatial=reference.spatial,
            temporal=reference.temporal,
            spatial_columns=reference.spatial_columns,
        )
    write_report(
        args.output / "record.json",
        {
            **report,
            "status": "complete"
            if failure is None
            else "memory_limited"
            if failure == "MemoryError"
            else "construction_failed",
            "error_type": failure,
        },
    )
    if failure is not None:
        raise RuntimeError(f"Reference construction screen failed: {failure}")


if __name__ == "__main__":
    main()
