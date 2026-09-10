"""Small spectral examples in unscaled Euclidean coordinates."""

import numpy as np
from scipy import linalg

from .benchmark_structure import restriction_example
from .problems import build_problem
from .solvers import orthonormalize
from .spectral import analytical_reference


def angle_diagnostics(B, W, comparison=None):
    """Use equal-dimensional spaces; report rank loss and nonunique cutoffs explicitly."""
    B, W = np.asarray(B, dtype=float), np.asarray(W, dtype=float)
    if (
        B.ndim != 2
        or B.shape[0] != B.shape[1]
        or W.ndim != 2
        or W.shape[0] != B.shape[0]
        or not 0 < W.shape[1] < B.shape[0]
        or not np.all(np.isfinite(B))
        or not np.all(np.isfinite(W))
        or not np.allclose(B, B.T, rtol=1e-13, atol=1e-14)
    ):
        raise ValueError(
            "Use a finite symmetric matrix and 0 < requested rank < inactive dimension"
        )
    requested = W.shape[1]
    Z = orthonormalize(W)
    values, U = linalg.eigh(B)
    if values[0] <= 0:
        raise ValueError("The examples require a positive-definite matrix")
    T = Z.T @ B @ Z
    F = B @ Z - Z @ T
    eta = float(values[requested] - linalg.eigvalsh(T)[-1]) if Z.shape[1] else None
    row = {
        "requested_rank": requested,
        "retained_rank": Z.shape[1],
        "minimum_restricted_singular_value": float(linalg.svdvals(W)[-1]),
        "subspace_residual_frobenius": float(linalg.norm(F, "fro")),
        "separation": eta,
        "cutoff_gap": float(values[requested] - values[requested - 1]),
        "largest_angle_degrees": None,
        "actual_sine": None,
        "residual_angle_bound": None,
        "separation_status": "rank_loss",
        "bound_status": "unavailable",
    }
    if Z.shape[1] != requested:
        row["retained_rank_comparison_angle_degrees"] = float(
            np.rad2deg(max(linalg.subspace_angles(U[:, : Z.shape[1]], Z), default=0))
        )
        return row
    target = U[:, :requested] if comparison is None else comparison
    angle = float(max(linalg.subspace_angles(target, Z)))
    row.update(largest_angle_degrees=float(np.rad2deg(angle)), actual_sine=float(np.sin(angle)))
    if eta > 0:
        bound = min(1.0, row["subspace_residual_frobenius"] / eta)
        row.update(residual_angle_bound=bound, separation_status="positive")
        row["bound_status"] = "nontrivial" if bound < 1 else "trivial_one"
    else:
        row.update(separation_status="nonpositive", bound_status="unavailable")
    return row


def diffusion_examples():
    rows = []
    for n in (4, 6, 12):
        M = build_problem("diffusion", n).H.toarray()
        Phi, _ = analytical_reference(n, 2, 1)
        for q in (1, 2, 3):
            mask = np.zeros((n, n), dtype=bool)
            mask[:q, :q] = True
            I, J = np.flatnonzero(~mask.ravel()), np.flatnonzero(mask.ravel())
            B, W = M[np.ix_(I, I)], Phi[I]
            row = angle_diagnostics(B, W)
            coupling = float(linalg.norm(M[np.ix_(I, J)] @ Phi[J], "fro"))
            eta = row["separation"]
            bound = (
                min(1.0, coupling / row["minimum_restricted_singular_value"] / eta)
                if eta > 0
                else None
            )
            values = linalg.eigvalsh(B)
            condition = float(values[-1] / values[0])
            row.update(
                n=n,
                corner_width=q,
                alpha=0.001,
                active_indices=J.tolist(),
                restriction_coupling_frobenius=coupling,
                coupling_angle_bound=bound,
                coupling_bound_status="unavailable"
                if bound is None
                else "nontrivial"
                if bound < 1
                else "trivial_one",
                original_condition=condition,
                deflated_condition_upper_bound=(
                    float((np.sqrt(values[-1] / values[1]) + np.sqrt(condition) * bound) ** 2)
                    if bound is not None
                    else None
                ),
            )
            rows.append(row)
    return rows


def spectral_examples():
    M0 = np.diag([1.0, 2.0, 4.0, 8.0])
    I = np.array([1, 2, 3])
    Phi = np.eye(4)[:, :2]
    rank_loss = angle_diagnostics(M0[np.ix_(I, I)], Phi[I])
    rank_loss.update(full_operator=M0.tolist(), reference=Phi.tolist(), inactive_indices=I.tolist())
    repeated = np.diag([1.0, 1.0, 3.0])
    cutoff = angle_diagnostics(repeated, np.eye(3)[:, :1], np.eye(3)[:, 1:2])
    cutoff["comparison"] = "two exact rank-one choices within the repeated lowest cluster"
    cluster = angle_diagnostics(repeated, np.eye(3)[:, :2])
    crossing = angle_diagnostics(np.diag([1.6, 1.4]), np.eye(2)[:, :1])
    crossing.update(
        reference_operator=[[1.0, 0.0], [0.0, 2.0]],
        current_operator=[[1.6, 0.0], [0.0, 1.4]],
        reference_gap=1.0,
        perturbation_norm=0.6,
        unconditional_delta_over_gap=0.6,
        sufficient_no_crossing=False,
    )
    algebraic = restriction_example()
    for row in algebraic["rows"]:
        bound = row["sufficient_angle_bound"]
        row["separation_status"] = "positive" if row["separation"] > 0 else "nonpositive"
        row["bound_status"] = (
            "unavailable" if bound is None else "nontrivial" if bound < 1 else "trivial_one"
        )
    return {
        "coordinates": "unscaled Euclidean",
        "bound_definitions": {
            "residual_angle_bound": "min(1, norm(F, fro)/eta), eta > 0 and full retained rank",
            "coupling_angle_bound": "min(1, norm(M0_IJ Phi_J, fro)/(sigma_min(Phi_I)*eta)), eta > 0",
            "unavailable": "rank loss or nonpositive separation; null differs from the trivial bound one",
        },
        "rank_loss": rank_loss,
        "repeated_cutoff": cutoff,
        "complete_cluster": cluster,
        "eigenvalue_crossing": crossing,
        "controlled_spd": algebraic,
        "diffusion": diffusion_examples(),
    }
