"""PDE separation example and matched previous-system Krylov-direction control."""

import argparse
from collections import deque
from pathlib import Path
import time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

from .benchmark_extended import digest, target_parameters, target_values
from .problems import build_problem, reference_modes
from .reporting import environment, write_report
from .solvers import calibrate_bound, deflated_cg, orthonormalize, pdas


def pde_separation(n=6):
    """One Laplacian reference mode, prescribed corner masks, unchanged operator."""
    problem = build_problem("diffusion", n)
    Phi = reference_modes(problem, 1)
    H = problem.H.toarray()
    rows = []
    for width in (1, 2, 3):
        mask = np.zeros((n, n), dtype=bool)
        mask[:width, :width] = True
        I, J = np.flatnonzero(~mask.ravel()), np.flatnonzero(mask.ravel())
        W = Phi[I]
        Z = orthonormalize(W)
        B = H[np.ix_(I, I)]
        values, vectors = linalg.eigh(B)
        T = Z.T @ B @ Z
        F = B @ Z - Z @ T
        separation = float(values[1] - T[0, 0])
        smin = float(linalg.svdvals(W)[-1])
        coupling = float(linalg.norm(H[np.ix_(I, J)] @ Phi[J], "fro"))
        angle = float(linalg.subspace_angles(vectors[:, :1], Z)[0])
        predicted = min(1.0, coupling / smin / separation) if separation > 0 else None
        rows.append(
            {
                "corner_width": width,
                "inactive_dof": len(I),
                "retained_rank": 1,
                "active_indices": J.tolist(),
                "minimum_restricted_singular_value": smin,
                "subspace_residual_frobenius": float(linalg.norm(F, "fro")),
                "restriction_coupling_frobenius": coupling,
                "operator_drift": 0.0,
                "separation": separation,
                "largest_angle_degrees": float(np.rad2deg(angle)),
                "actual_sine": float(np.sin(angle)),
                "sufficient_angle_bound": predicted,
                "residual_identity_error": float(
                    linalg.norm(
                        F + (np.eye(len(I)) - Z @ Z.T) @ H[np.ix_(I, J)] @ Phi[J] / (Z.T @ W).item()
                    )
                ),
            }
        )
    return {
        "n": n,
        "alpha": problem.alpha,
        "coordinates": "unscaled Euclidean",
        "scope": "controlled PDE inactive-set kernel with prescribed masks; no optimality claim",
        "rows": rows,
    }


def retained_directions(B, directions, rank):
    """Select low Ritz directions from the previous solve's bounded Krylov history."""
    if not directions:
        return np.empty((B.shape[0], 0))
    V = orthonormalize(np.column_stack(directions))
    count = min(rank, V.shape[1])
    if count == 0:
        return V
    T = V.T @ (B @ V)
    _, rotation = linalg.eigh((T + T.T) / 2, subset_by_index=[0, count - 1])
    return V @ rotation


def recycling_comparison(name, n, rank=20, targets=7, repeats=3):
    problem = build_problem(name, n)
    parameters = [target_parameters(i, targets) for i in range(targets)]
    for p in parameters:
        p["centers"] = [center[: problem.dim] for center in p["centers"]]
    tick = time.perf_counter()
    bound, activity = calibrate_bound(
        problem.H, target_values(problem.coordinates, parameters[targets // 2]), 0.2, steps=8
    )
    cases = []
    for p in parameters:
        f = target_values(problem.coordinates, p)
        outer = pdas(problem.H, f, bound, tolerance=1e-8)
        if outer["status"] != "converged":
            raise RuntimeError("Mask-generating PDAS failed")
        I, J = np.flatnonzero(~outer["active"]), np.flatnonzero(outer["active"])
        B = problem.H[I][:, I].tocsr()
        b = f[I] - problem.H[I][:, J] @ np.full(len(J), bound)
        cases.append(
            (
                I,
                B,
                b,
                {
                    "index": p["index"],
                    "target_sha256": digest(f),
                    "rhs_sha256": digest(b),
                    "inactive_sha256": digest(I),
                    "kkt": outer["kkt"],
                    "inactive_dof": len(I),
                },
            )
        )
    calibration_and_masks = time.perf_counter() - tick
    sequences = []
    for repeat in range(repeats):
        tick = time.perf_counter()
        Phi = reference_modes(problem, rank)
        reference_seconds = time.perf_counter() - tick
        recycled = np.empty((problem.H.shape[0], 0))
        costs = {"reference": reference_seconds, "recycling": 0.0}
        rows = []
        for index, (I, B, b, metadata) in enumerate(cases):
            tick = time.perf_counter()
            history_candidate = orthonormalize(recycled[I])
            transfer_seconds = time.perf_counter() - tick
            # First target seeds both methods with the same rank-zero solve.
            common_rank = min(rank, history_candidate.shape[1]) if index else 0
            tick = time.perf_counter()
            fixed_candidate = Phi[I, :common_rank]
            fixed_restriction_seconds = time.perf_counter() - tick
            candidates = {
                "reference": fixed_candidate,
                "recycling": history_candidate[:, :common_rank],
            }
            order = (
                ["reference", "recycling"]
                if (repeat + index) % 2 == 0
                else ["recycling", "reference"]
            )
            pair = []
            for method in order:
                directions = deque(maxlen=100)
                tick = time.perf_counter()
                result = deflated_cg(
                    B,
                    b,
                    candidates[method],
                    B.diagonal(),
                    rtol=1e-10,
                    maxiter=20000,
                    direction_callback=directions.append if method == "recycling" else None,
                )
                solve_seconds = time.perf_counter() - tick
                selection_seconds = 0.0
                if method == "recycling":
                    tick = time.perf_counter()
                    retained = (
                        retained_directions(B, directions, rank)
                        if result.status == "converged"
                        else np.empty((len(I), 0))
                    )
                    recycled = np.zeros((problem.H.shape[0], retained.shape[1]))
                    recycled[I] = retained
                    selection_seconds = time.perf_counter() - tick
                total = (
                    solve_seconds
                    + selection_seconds
                    + (transfer_seconds if method == "recycling" else fixed_restriction_seconds)
                )
                costs[method] += total
                pair.append(
                    {
                        **metadata,
                        "method": method,
                        "requested_rank": rank,
                        "matched_rank": common_rank,
                        "deployed_rank": result.rank,
                        "iterations": result.iterations,
                        "original_residual": result.residual,
                        "status": result.status,
                        "coarse_condition": result.coarse_condition,
                        "fallback_reason": result.fallback_reason,
                        "solve_seconds": solve_seconds,
                        "next_space_selection_seconds": selection_seconds,
                        "transfer_seconds": transfer_seconds
                        if method == "recycling"
                        else fixed_restriction_seconds,
                        "total_seconds": total,
                        "cumulative_seconds": costs[method],
                    }
                )
            matched = pair[0]["deployed_rank"] == pair[1]["deployed_rank"] == common_rank
            rows.extend({**row, "matched_deployed_rank": matched} for row in pair)
        sequences.append(
            {
                "repetition": repeat,
                "reference_construction_seconds": reference_seconds,
                "cost_sums": costs,
                "rows": rows,
            }
        )
    return {
        "family": name,
        "n": n,
        "rank": rank,
        "targets": targets,
        "repeats": repeats,
        "bound": bound,
        "calibrated_activity": activity,
        "calibration_and_mask_seconds": calibration_and_masks,
        "sequences": sequences,
        "success": all(r["status"] == "converged" for s in sequences for r in s["rows"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = {
        "protocol": "pde-separation-and-history-control-v1",
        "pde": {"grids": [4, 6, 12], "rank": 1, "corner_widths": [1, 2, 3], "alpha": 0.001},
        "recycling": {
            "families": [["diffusion", 16], ["thermal", 16], ["cht", 8]],
            "targets": 7,
            "repeats": 3,
            "rank": 20,
            "history_capacity": 100,
            "transfer": "zero extension of retained previous-system Ritz directions; restrict to current inactive set",
            "rank_matching": "both start at rank zero; later use min(20, transferred history rank)",
            "rtol": 1e-10,
            "maxiter": 20000,
            "rank_threshold": 1e-12,
            "coarse_limit": 1e10,
            "initial_guess": "zero for both methods",
            "pdas_tolerance": 1e-8,
            "timing_scope": "kernel, transfer and next-space selection costs; reference construction once; calibration and mask generation separately",
        },
    }
    write_report(args.output / "protocol.json", spec)
    report = {"specification": spec, "environment": environment(), "recycling": []}
    with threadpool_limits(limits=4):
        report["pde_separation"] = [pde_separation(n) for n in spec["pde"]["grids"]]
        write_report(args.output / "results.json", report)
        for family, n in spec["recycling"]["families"]:
            report["recycling"].append(recycling_comparison(family, n))
            write_report(args.output / "results.json", report)
        report["success"] = all(r["success"] for r in report["recycling"])
        write_report(args.output / "results.json", report)
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
