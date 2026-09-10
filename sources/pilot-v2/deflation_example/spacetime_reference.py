"""Compact full-domain reference modes for coupled thermal trajectories."""

import numpy as np
from scipy import linalg

from .spectral import analytical_reference
from .validation import integer, positive_real


class SpaceTimeReference:
    """Store spatial and temporal factors; materialize only restricted columns."""

    def __init__(self, spatial, temporal, spatial_columns, description):
        self.spatial = spatial
        self.temporal = temporal
        self.spatial_columns = np.asarray(spatial_columns, dtype=int)
        self.rank = len(spatial_columns)
        self.description = description

    def restrict(self, indices):
        time_rows, space_rows = np.divmod(indices, self.spatial.shape[0])
        result = np.empty((len(indices), self.rank))
        for column, spatial_column in enumerate(self.spatial_columns):
            result[:, column] = (
                self.spatial[space_rows, spatial_column] * self.temporal[time_rows, column]
            )
        return result

    def storage(self):
        return {
            "compact_reference_bytes": self.spatial.nbytes
            + self.temporal.nbytes
            + self.spatial_columns.nbytes,
            "full_reference_bytes": 0,
            "full_reference_equivalent_bytes": self.spatial.shape[0]
            * self.temporal.shape[0]
            * self.rank
            * 8,
            "spatial_factor_columns": self.spatial.shape[1],
            "total_space_time_rank": self.rank,
        }


def build_space_time_reference(
    trajectory, rank, construction="mode_dependent", reference_capacity=1.0
):
    """Compare exact homogeneous mode-dependent and common temporal constructions.

    For each spatial Laplacian eigenvalue lambda, the reference temporal matrix
    is W + alpha (lambda I + c_ref T).T W (lambda I + c_ref T). Mode-dependent
    temporal eigenvectors diagonalize this matrix. The tensor alternative uses
    the same pure-time eigenvectors for every spatial mode and orders products
    by their reference Rayleigh values. Both retain the declared total rank.
    """
    rank = integer(rank, "Total space-time rank")
    if rank >= trajectory.size:
        raise ValueError("Space-time rank must be below the full trajectory dimension")
    if construction not in {"mode_dependent", "tensor"}:
        raise ValueError("Choose mode_dependent or tensor reference construction")
    capacity = positive_real(reference_capacity, "Reference thermal capacity")
    n, slabs = trajectory.spatial.n, len(trajectory.steps)
    spatial, modes = analytical_reference(n, 3, min(rank, n**3))
    T = trajectory.time_matrix.toarray()
    W = np.diag(trajectory.steps / trajectory.steps.mean())
    alpha = trajectory.spatial.alpha
    common = (
        linalg.eigh(W + alpha * capacity**2 * T.T @ W @ T)[1] if construction == "tensor" else None
    )
    candidates = []
    vectors = []
    for j, mode in enumerate(modes):
        eigenvalue = sum(4 * (n + 1) ** 2 * np.sin(np.pi * k / (2 * (n + 1))) ** 2 for k in mode)
        F = eigenvalue * np.eye(slabs) + capacity * T
        normal = W + alpha * F.T @ W @ F
        if common is None:
            values, temporal = linalg.eigh(normal)
        else:
            temporal = common
            values = np.diag(temporal.T @ normal @ temporal)
        vectors.append(temporal)
        candidates.extend((float(value), j, ell) for ell, value in enumerate(values))
    selected = sorted(candidates)[:rank]
    used = sorted({j for _, j, _ in selected})
    mapped = {j: i for i, j in enumerate(used)}
    temporal = np.empty((slabs, rank))
    for column, (_, j, ell) in enumerate(selected):
        vector = vectors[j][:, ell]
        temporal[:, column] = vector * (1 if vector[np.argmax(np.abs(vector))] >= 0 else -1)
    return SpaceTimeReference(
        spatial[:, used],
        temporal,
        [mapped[j] for _, j, _ in selected],
        {
            "construction": construction,
            "reference_capacity": capacity,
            "spatial_modes": [modes[j] for j in used],
            "spatial_column_indices": [mapped[j] for _, j, _ in selected],
            "temporal_column_indices": [ell for _, _, ell in selected],
            "reference_rayleigh_values": [value for value, _, _ in selected],
            "total_rank": rank,
        },
    )
