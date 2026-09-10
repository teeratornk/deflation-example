"""Coarse-mesh spatial references and mode-dependent temporal factors."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import eigsh

from .spacetime_reference import SpaceTimeReference
from .study_solvers import ArrayReference
from .validation import integer


def spatial_reference(assembly, rank, *, alpha=None):
    rank = integer(rank, "Reference rank", 1)
    I = assembly.mesh.free
    if rank >= len(I):
        raise ValueError("Reference rank must be below the coarse spatial dimension")
    K = assembly.diffusion[I][:, I]
    m = assembly.mass[I]
    if alpha is not None:
        from .mesh_control import build_mesh_control
        H = build_mesh_control(assembly, alpha=alpha).H
        K, m = H.assembled(), H.diagonal()
    if len(I) < 256 or rank > len(I) // 3:
        values, vectors = linalg.eigh(K.toarray(), np.diag(m), subset_by_index=[0, rank - 1])
    else:
        values, vectors = eigsh(K, k=rank, M=sparse.diags(m), sigma=0, which="LM",
                                v0=np.random.default_rng(701).normal(size=len(I)), tol=1e-10)
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
    residual = K @ vectors - m[:, None] * vectors * values
    denominator = np.linalg.norm(K @ vectors, axis=0)
    error = np.linalg.norm(residual, axis=0) / np.maximum(denominator, np.finfo(float).tiny)
    if not np.isfinite(error).all() or error.max() > 1e-7:
        raise RuntimeError("Reference eigensolve failed its independently recomputed residual check")
    return values, vectors, error


def build_mesh_reference(problem, coarse_assembly, prolongation, rank, construction="mode_dependent",
                         spatial_policy="diffusion"):
    """Build one full-domain reference before optimization, including all time levels."""
    rank = integer(rank, "Total reference rank", 1)
    if construction not in {"mode_dependent", "tensor"}:
        raise ValueError("Unknown temporal reference construction")
    if spatial_policy not in {"diffusion", "scaled_schur"}:
        raise ValueError("Choose diffusion or scaled_schur spatial reference")
    spatial_rank = min(rank, len(coarse_assembly.mesh.free) - 1)
    values, phi, residuals = spatial_reference(coarse_assembly, spatial_rank,
                                              alpha=problem.alpha if spatial_policy == "scaled_schur" else None)
    phi = np.asarray(prolongation @ phi)
    description = {
        "construction": f"generalized_{spatial_policy}_modes_with_P1_prolongation",
        "generalized_weight": "lumped_mass" if spatial_policy == "diffusion" else "Jacobi_diagonal",
        "coarse_spatial_rank": spatial_rank,
        "coarse_nodes": len(coarse_assembly.mesh.nodes),
        "fine_nodes": len(problem.assembly.mesh.nodes),
        "generalized_eigenvalues": values.tolist(),
        "reference_relative_residuals": residuals.tolist(),
        "total_requested_rank": rank,
    }
    if not len(problem.steps):
        if spatial_rank != rank:
            raise ValueError("The coarse mesh cannot support the requested spatial rank")
        return ArrayReference(phi, description)
    steps = problem.steps
    n = len(steps)
    T = sparse.diags([1 / steps, -1 / steps[1:]], [0, -1], shape=(n, n)).toarray()
    Wt = np.diag(steps / steps.mean())
    w = problem.assembly.mass[problem.free]
    w = w / w.mean()
    AP = problem.spatial_A @ phi
    CP = problem.nodal_capacity[:, None] * phi
    factors, candidates = [], []
    common = linalg.eigh(Wt + problem.alpha * T.T @ Wt @ T)[1]
    for j in range(spatial_rank):
        mass = float(phi[:, j] @ (w * phi[:, j]))
        aa = float(AP[:, j] @ (w * AP[:, j]))
        ac = float(AP[:, j] @ (w * CP[:, j]))
        cc = float(CP[:, j] @ (w * CP[:, j]))
        H = mass * Wt + problem.alpha * (aa * Wt + ac * (Wt @ T + T.T @ Wt)
                                        + cc * T.T @ Wt @ T)
        if construction == "mode_dependent":
            score, vectors = linalg.eigh(H)
        else:
            vectors = common
            score = np.diag(vectors.T @ H @ vectors)
        factors.append(vectors)
        candidates.extend((float(s), j, k) for k, s in enumerate(score))
    selected = sorted(candidates)[:rank]
    if len(selected) != rank:
        raise ValueError("The coarse space-time pool cannot support the requested rank")
    used = sorted({j for _, j, _ in selected})
    remap = {j: i for i, j in enumerate(used)}
    return SpaceTimeReference(phi[:, used],
                              np.column_stack([factors[j][:, k] for _, j, k in selected]),
                              [remap[j] for _, j, _ in selected],
                              {**description, "temporal_construction": construction,
                               "selection": selected,
                               "temporal_operator": "one-spatial-mode compression of the weighted trajectory Hessian"})
