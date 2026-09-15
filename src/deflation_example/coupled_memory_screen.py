"""Screen named storage costs before a large coupled optimization.

This diagnostic factors momentum Jacobians at saved velocity fields and candidate
time steps. It does not solve a new optimization or certify its memory demand.
Factor-array extrapolations, explicit dense-array sizes and sampled process
memory are separate quantities; they must not be added as disjoint components.
"""

import argparse
import gc
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from .coupled_optimize import load_problem
from .coupled_factor_storage import RecomputedLU
from .coupled_saved import load_saved_solution, require_matching_baseline
from .memory import ProcessMemory
from .reporting import environment, write_report
from .validation import integer, positive_real


def dense_storage(spatial_size, slabs, rank):
    """Exact byte counts for named float64 arrays, with all state variables inactive.

    The compact bound permits rank distinct spatial columns. The dense temporal
    pool describes the existing implementation before its final selection.
    Workspace, sparse operators, allocator overhead and flow factors are excluded.
    """
    spatial_size = integer(spatial_size, "Spatial dimension", 2)
    slabs = integer(slabs, "Time slabs", 1)
    rank = integer(rank, "Rank", 0)
    size = spatial_size * slabs
    return {
        "state_degrees_of_freedom": size,
        "one_state_vector_bytes": size * 8,
        "one_full_restricted_basis_bytes": size * rank * 8,
        "one_cached_operator_product_bytes": size * rank * 8,
        "compact_reference_upper_bytes": (spatial_size + slabs) * rank * 8 + rank * 8,
        "dense_temporal_vector_pool_bytes": slabs**2 * min(rank, spatial_size - 1) * 8,
        "tridiagonal_selected_temporal_vectors_bytes": slabs * rank * 8,
        "tridiagonal_one_mode_vectors_upper_bytes": slabs * min(rank, slabs) * 8,
    }


def factor_inventory(matrix, checkpoint_comparison=False):
    """Expose sparse-LU array sizes and independently check one deterministic solve.

    SciPy's exported L/U arrays exclude SuperLU's opaque internal workspace.
    Accessing these arrays can itself increase the diagnostic's sampled memory.
    """
    tick = time.perf_counter()
    factor = splu(matrix.tocsc())
    factor_seconds = time.perf_counter() - tick
    L, U = factor.L, factor.U
    arrays = {
        "L_bytes": sum(a.nbytes for a in (L.data, L.indices, L.indptr)),
        "U_bytes": sum(a.nbytes for a in (U.data, U.indices, U.indptr)),
        "row_permutation_bytes": factor.perm_r.nbytes,
        "column_permutation_bytes": factor.perm_c.nbytes,
    }
    rhs = np.random.default_rng(417).normal(size=matrix.shape[0])
    solution = factor.solve(rhs)
    residual = np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs)
    result = {
        "factor_seconds": factor_seconds,
        "dimension": matrix.shape[0],
        "matrix_nonzeros": matrix.nnz,
        "factor_nonzeros": L.nnz + U.nnz,
        "exported_arrays": arrays,
        "exported_factor_array_bytes": sum(arrays.values()),
        "fresh_relative_residual": float(residual),
        "factor_check_passed": bool(np.isfinite(residual) and residual <= 1e-8),
    }
    if checkpoint_comparison:
        tick = time.perf_counter()
        checkpoint = RecomputedLU(matrix)
        checkpoint_construction_seconds = time.perf_counter() - tick
        rows = []
        for columns in (1, 20):
            values = np.random.default_rng(912 + columns).normal(size=(matrix.shape[0], columns))
            for transpose in ("N", "T"):
                for repetition in range(2):
                    order = (
                        ("retained", "recompute") if repetition == 0 else ("recompute", "retained")
                    )
                    for policy in order:
                        implementation = factor if policy == "retained" else checkpoint
                        tick = time.perf_counter()
                        answer = implementation.solve(values, trans=transpose)
                        seconds = time.perf_counter() - tick
                        A = matrix if transpose == "N" else matrix.T
                        relative = np.linalg.norm(A @ answer - values) / np.linalg.norm(values)
                        rows.append(
                            {
                                "policy": policy,
                                "columns": columns,
                                "transpose": transpose,
                                "repetition": repetition,
                                "seconds": seconds,
                                "fresh_relative_residual": float(relative),
                                "verified": bool(np.isfinite(relative) and relative <= 1e-8),
                            }
                        )
        result["checkpoint_comparison"] = {
            "matrix_checkpoint_bytes": checkpoint.checkpoint_bytes,
            "checkpoint_construction_seconds": checkpoint_construction_seconds,
            "recomputed_factorizations": checkpoint.factorizations,
            "recomputed_factor_seconds": checkpoint.factor_seconds,
            "recomputed_solve_seconds": checkpoint.solve_seconds,
            "all_actions_verified": all(row["verified"] for row in rows),
            "actions": rows,
            "scope": "Matched vector/block and transpose solves with reversed policy order. Recompute intervals include factorization. Retained intervals use the preceding factorization, whose cost is reported separately. Both objects coexist in this diagnostic; sampled process memory does not compare complete optimizer policies.",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("jacobi", "reference", "recycling"), default="reference"
    )
    parser.add_argument("--target-position", type=int)
    parser.add_argument("--sample-steps", type=int, nargs="+", required=True)
    parser.add_argument("--slabs", type=int, nargs="+", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 20, 100, 200])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-comparison", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    samples = [integer(n, "Sample step", 0) for n in args.sample_steps]
    slabs = [integer(n, "Time slabs", 1) for n in args.slabs]
    ranks = [integer(n, "Rank", 0) for n in args.ranks]
    if any(len(set(values)) != len(values) for values in (samples, slabs, ranks)):
        raise ValueError("Declare distinct sample steps, time grids and ranks")
    args.output.mkdir(parents=True, exist_ok=False)
    rows, failure = [], None
    metadata = {
        "schema": "coupled-memory-screen-v1",
        "environment": environment(),
        "declared_steps_zero_based": samples,
        "declared_time_slabs": slabs,
        "declared_ranks": ranks,
        "threads": integer(args.threads, "Threads", 1),
        "checkpoint_comparison": args.checkpoint_comparison,
        "scope": "Saved velocity fields at candidate time steps; no new flow trajectory or optimization. Factor-array extrapolations are estimates, not memory guarantees. Named arrays overlap sampled process memory and must not be added to it.",
    }
    write_report(args.output / "record.json", {**metadata, "status": "running", "rows": rows})
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        sampler = ProcessMemory("cpu", 0.01)
        try:
            sampler.start()
            record, cfg, _, digest = load_saved_solution(
                args.optimization, args.method, args.target_position
            )
            if not cfg["transient"]:
                raise ValueError("This screen requires a saved transient optimization")
            horizon = positive_real(cfg["horizon_s"], "Physical horizon")
            # A single-slab model supplies the unchanged spatial matrices. Candidate
            # time steps enter the momentum Jacobian explicitly below.
            problem, baseline = load_problem(
                {**cfg, "slabs": 1, "baseline_directory": str(args.baseline)}
            )
            require_matching_baseline(record, baseline)
            metadata.update(
                optimization_field_sha256=digest,
                baseline_sha256=baseline["baseline_sha256"],
                physical_horizon_s=horizon,
                spatial_state_degrees_of_freedom=problem.spatial_size,
            )
            filename = (
                args.method + "-fields.npz"
                if args.target_position is None
                else f"target-{args.target_position:02d}.npz"
            )
            with np.load(args.optimization / filename, allow_pickle=False) as data:
                velocities = data["velocity"]
                if (
                    velocities.shape != (cfg["slabs"], problem.flow.nv, 2)
                    or not np.isfinite(velocities).all()
                ):
                    raise ValueError("Saved velocities must match the complete original trajectory")
                if max(samples) >= len(velocities):
                    raise ValueError("Select available saved time steps")
                velocities = velocities[samples].copy()
            metadata["named_dense_arrays"] = [
                {"slabs": n, "rank": r, **dense_storage(problem.spatial_size, n, r)}
                for n in slabs
                for r in ranks
            ]
            for n in slabs:
                for step, velocity in zip(samples, velocities, strict=True):
                    row = {"slabs": n, "time_step_s": horizon / n, "saved_step_zero_based": step}
                    try:
                        J = problem.flow.operator(
                            velocity, time_step=horizon / n
                        ) + problem.flow.convection_derivative(velocity)
                        matrix = J[problem.flow_free][:, problem.flow_free].tocsc()
                        values = (
                            factor_inventory(matrix, checkpoint_comparison=True)
                            if args.checkpoint_comparison
                            else factor_inventory(matrix)
                        )
                        row.update(status="complete", **values)
                        row["one_trajectory_factor_array_extrapolation_bytes"] = (
                            n * values["exported_factor_array_bytes"]
                        )
                        row["current_and_trial_factor_array_extrapolation_bytes"] = (
                            2 * n * values["exported_factor_array_bytes"]
                        )
                        if args.checkpoint_comparison:
                            row["current_and_trial_matrix_checkpoint_extrapolation_bytes"] = (
                                2 * n * values["checkpoint_comparison"]["matrix_checkpoint_bytes"]
                            )
                        del J, matrix
                    except (MemoryError, RuntimeError) as error:
                        row.update(
                            status="memory_limited"
                            if isinstance(error, MemoryError)
                            else "factorization_failed",
                            error_type=type(error).__name__,
                        )
                    rows.append(row)
                    gc.collect()
                    write_report(
                        args.output / "record.json", {**metadata, "status": "running", "rows": rows}
                    )
        except Exception as error:
            failure = type(error).__name__
        try:
            memory = sampler.finish()
        except Exception as error:
            failure = failure or type(error).__name__
            memory = {"complete": False, "error_type": type(error).__name__}
    write_report(
        args.output / "record.json",
        {
            **metadata,
            "status": "complete" if failure is None else "screen_error",
            "error_type": failure,
            "rows": rows,
            "sampled_process_memory": memory,
            "all_factor_checks_passed": failure is None
            and len(rows) == len(samples) * len(slabs)
            and all(
                row.get("factor_check_passed", False)
                and (
                    not args.checkpoint_comparison
                    or row["checkpoint_comparison"]["all_actions_verified"]
                )
                for row in rows
            ),
            "optimization_memory_feasibility_established": False,
        },
    )
    if failure is not None:
        raise RuntimeError(f"Memory screen failed: {failure}")


if __name__ == "__main__":
    main()
