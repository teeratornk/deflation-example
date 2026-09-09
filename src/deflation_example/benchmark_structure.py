"""Small reproducible controls for restriction bounds and Ritz coordinates."""
import argparse
from pathlib import Path
import time
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
from .problems import build_problem
from .reporting import environment, write_report
from .solvers import calibrate_bound, pdas, orthonormalize, deflated_cg
from .spectral import analytical_reference
from .validation import integer, positive_real


def restriction_example():
    """Eight-dimensional SPD example, one deleted DOF, two reference modes.

    All diagnostics use unscaled Euclidean coordinates. Orthogonal plane
    rotations couple the first two eigendirections to the removed DOF.
    A positive semidefinite rank-one perturbation then produces a crossing.
    """
    Q = np.eye(8)
    for a, b, angle in [(0, 7, .2), (1, 7, .15), (0, 2, .1)]:
        rotation = np.eye(8)
        rotation[a, a] = rotation[b, b] = np.cos(angle)
        rotation[a, b], rotation[b, a] = -np.sin(angle), np.sin(angle)
        Q = Q @ rotation
    M0 = (Q * np.array([1., 2., 5., 8., 11., 14., 17., 20.])) @ Q.T
    Phi, I, J = Q[:, :2], np.arange(7), np.array([7])
    W = Phi[I]
    Z = orthonormalize(W)
    v = Z[:, 1]
    rows = []
    for amplitude in (0., .01, .1, .5, 1., 2., 4., 8.):
        delta = amplitude * np.outer(v, v)
        B = M0[np.ix_(I, I)] + delta
        eigenvalues, U = linalg.eigh(B)
        T = Z.T @ B @ Z
        F = B @ Z - Z @ T
        eta = eigenvalues[2] - linalg.eigvalsh(T)[-1]
        beta = linalg.norm(Phi[J], 2)
        coupling = linalg.norm(M0[np.ix_(I, J)] @ Phi[J], "fro")
        smin = linalg.svdvals(W)[-1]
        numerator = (coupling + linalg.norm(delta, 2)*np.sqrt(2)) / np.sqrt(1-beta**2)
        angle = np.max(linalg.subspace_angles(U[:, :2], Z))
        direct_bound = min(1., linalg.norm(F, "fro") / eta) if eta > 0 else None
        sufficient = min(1., numerator/eta) if eta > 0 else None
        residual_identity = (np.eye(7)-Z@Z.T) @ (
            -M0[np.ix_(I, J)] @ Phi[J] + delta @ W) @ linalg.inv(Z.T @ W)
        rows.append({"amplitude": amplitude, "retained_rank": Z.shape[1],
            "minimum_restricted_singular_value": smin, "removed_mass": beta,
            "coupling_frobenius": coupling, "subspace_residual_frobenius": linalg.norm(F, "fro"),
            "residual_identity_error": linalg.norm(F-residual_identity, "fro"),
            "separation": eta, "largest_angle_degrees": np.rad2deg(angle),
            "actual_sine": np.sin(angle), "residual_angle_bound": direct_bound,
            "sufficient_angle_bound": sufficient})
    return {"coordinates": "unscaled", "full_dimension": 8, "inactive_indices_zero_based": I,
            "reference_eigenvalues": [1., 2., 5., 8., 11., 14., 17., 20.],
            "rotations": [[0, 7, .2], [1, 7, .15], [0, 2, .1]], "rows": rows}


def ritz_space(B, candidates, rank, scaled=False):
    """Select in Euclidean or Jacobi coordinates, then return state coordinates."""
    integer(rank, "Retained rank", 1)
    root = np.sqrt(B.diagonal())
    V = orthonormalize(root[:, None] * candidates if scaled else candidates)
    if V.shape[1] < rank:
        raise ValueError("Candidate numerical rank is below the requested retained rank")
    BV = (B @ (V / root[:, None])) / root[:, None] if scaled else B @ V
    T = V.T @ BV
    _, vectors = linalg.eigh((T + T.T)/2, subset_by_index=[0, rank-1])
    selected = V @ vectors
    return selected / root[:, None] if scaled else selected


def ritz_comparison(n=12, pool=300, rank=100, repeats=5):
    integer(n, "Grid", 2)
    integer(pool, "Candidate rank", 1)
    integer(repeats, "Repetitions", 1)
    if not 0 < rank <= pool < n**3:
        raise ValueError("Require 0 < rank <= pool < n cubed")
    problem = build_problem("cht", n)
    bound, activity = calibrate_bound(problem.H, problem.target(np.pi/4), .2)
    start = time.perf_counter()
    Phi, indices = analytical_reference(n, 3, pool)
    reference_seconds = time.perf_counter()-start
    rows = []
    for theta in (0., np.pi/4, np.pi/2):
        f = problem.target(theta)
        outer = pdas(problem.H, f, bound)
        if outer["status"] != "converged":
            raise RuntimeError("CHT PDAS failed")
        I, J = np.flatnonzero(~outer["active"]), np.flatnonzero(outer["active"])
        B = problem.H[I][:, I].tocsr()
        b = f[I] - problem.H[I][:, J] @ np.full(len(J), bound)
        for repeat in range(repeats):
            order = ("reference", "unscaled_ritz", "scaled_ritz")
            order = order[repeat % 3:] + order[:repeat % 3]
            for method in order:
                start = time.perf_counter()
                candidates = Phi[I]
                Z = candidates[:, :rank] if method == "reference" else ritz_space(
                    B, candidates, rank, method == "scaled_ritz")
                selection_seconds = time.perf_counter()-start
                result = deflated_cg(B, b, Z, B.diagonal(), rtol=1e-10, maxiter=10000)
                rows.append({"theta": theta, "method": method, "repetition": repeat,
                    "requested_rank": rank, "effective_rank": result.rank,
                    "selection_seconds": selection_seconds, "total_seconds": time.perf_counter()-start,
                    "iterations": result.iterations, "original_residual": result.residual,
                    "status": result.status, "coarse_condition": result.coarse_condition,
                    "fallback_reason": result.fallback_reason, "n_inactive": len(I)})
    return {"n": n, "candidate_rank": pool, "retained_rank": rank, "bound": bound,
            "calibrated_activity": activity, "reference_seconds": reference_seconds,
            "mode_indices": indices, "initial_guess": "zero", "rtol": 1e-10,
            "rank_threshold": 1e-12, "coarse_condition_limit": 1e10,
            "timing_scope": "restriction, selection, solver setup, iteration, original residual",
            "rows": rows, "success": all(r["status"] == "converged" for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    positive_real(args.threads, "Threads")
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=args.threads):
        report = {"environment": environment(), "restriction": restriction_example()}
        write_report(args.output / "results.json", report)
        report["ritz"] = ritz_comparison(repeats=args.repeats)
        write_report(args.output / "results.json", report)
        if not report["ritz"]["success"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
