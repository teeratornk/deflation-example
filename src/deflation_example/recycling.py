"""Bounded, Jacobi-scaled recycling between successive inactive systems.

Each accepted solve contributes its deployed coarse vectors and a bounded window
of new CG directions. Ritz selection takes place in Jacobi coordinates. The
selected physical vectors are stored on the previous inactive set and transferred
by zero extension. No replacement directions are added after restriction loss.
"""

from collections import deque
import time

import numpy as np
from scipy import linalg

from .validation import integer, matrix, positive_real, real_array


def inactive_indices(values):
    values = np.asarray(values)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise ValueError("Inactive indices must be a one-dimensional integer array")
    if values.size and (np.any(values < 0) or np.any(values > np.iinfo(np.int64).max)):
        raise ValueError("Inactive indices must be nonnegative signed integers")
    values = values.astype(np.int64, copy=True)
    if not np.all(values[1:] > values[:-1]) and len(np.unique(values)) != len(values):
        raise ValueError("Inactive indices must be distinct")
    return values


def transfer_basis(basis, previous, current):
    """Apply R_current R_previous.T without a full-domain dense allocation."""
    previous, current = inactive_indices(previous), inactive_indices(current)
    basis = real_array(basis, "Transferred basis")
    if basis.ndim != 2 or basis.shape[0] != len(previous) or not np.isfinite(basis).all():
        raise ValueError("Transferred basis must be finite and match the previous indices")
    transferred = np.zeros((len(current), basis.shape[1]))
    _, old_rows, new_rows = np.intersect1d(
        previous, current, assume_unique=True, return_indices=True
    )
    transferred[new_rows] = basis[old_rows]
    return transferred


def jacobi_ritz(A, diagonal, candidates, rank, tolerance=1e-12):
    """Select low Ritz vectors of D^(-1/2) A D^(-1/2), returning physical vectors."""
    A = matrix(A)
    diagonal, candidates = real_array(diagonal, "Diagonal"), real_array(candidates, "Candidates")
    rank = integer(rank, "Retained rank")
    tolerance = positive_real(tolerance, "Numerical rank tolerance")
    if (
        candidates.ndim != 2
        or A.shape != (len(diagonal), len(diagonal))
        or candidates.shape[0] != len(diagonal)
        or diagonal.ndim != 1
        or not np.isfinite(diagonal).all()
        or np.any(diagonal <= 0)
        or not np.isfinite(candidates).all()
    ):
        raise ValueError(
            "Ritz selection needs finite compatible candidates and a positive diagonal"
        )
    n = len(diagonal)
    if not rank or 0 in candidates.shape:
        return np.empty((n, 0)), {"candidate_rank": 0, "selected_rank": 0}
    sqrt_d = np.sqrt(diagonal)
    scaled = sqrt_d[:, None] * candidates
    norms = linalg.norm(scaled, axis=0)
    scaled = scaled[:, norms > 0] / norms[norms > 0]
    if scaled.shape[1] == 0:
        return np.empty((n, 0)), {"candidate_rank": 0, "selected_rank": 0}
    Q, R = linalg.qr(scaled, mode="economic")
    U, singular, _ = linalg.svd(R, full_matrices=False)
    keep = singular > tolerance * singular[0]
    Q = Q @ U[:, keep]
    physical = Q / sqrt_d[:, None]
    projected = physical.T @ (A @ physical)
    values, vectors = linalg.eigh((projected + projected.T) / 2)
    retained = min(rank, len(values))
    return physical @ vectors[:, :retained], {
        "candidate_rank": int(keep.sum()),
        "selected_rank": retained,
        "candidate_sigma_min": float(singular[keep][-1]),
        "ritz_min": float(values[0]),
        "ritz_max": float(values[retained - 1]),
    }


def gpu_jacobi_ritz(A, diagonal, candidates, rank, torch, tolerance=1e-12):
    """GPU counterpart; inputs and returned physical vectors stay on the device."""
    if not rank or 0 in candidates.shape:
        return candidates[:, :0], {"candidate_rank": 0, "selected_rank": 0}
    sqrt_d = torch.sqrt(diagonal)
    scaled = sqrt_d[:, None] * candidates
    norms = torch.linalg.vector_norm(scaled, dim=0)
    nonzero = norms > 0
    scaled = scaled[:, nonzero] / norms[nonzero]
    if scaled.shape[1] == 0:
        return candidates[:, :0], {"candidate_rank": 0, "selected_rank": 0}
    Q, R = torch.linalg.qr(scaled, mode="reduced")
    U, singular, _ = torch.linalg.svd(R, full_matrices=False, driver="gesvd")
    keep = singular > tolerance * singular[0]
    Q = Q @ U[:, keep]
    physical = Q / sqrt_d[:, None]
    projected = physical.T @ torch.mm(A, physical)
    values, vectors = torch.linalg.eigh((projected + projected.T) / 2)
    retained = min(rank, len(values))
    return physical @ vectors[:, :retained], {
        "candidate_rank": int(keep.sum()),
        "selected_rank": retained,
        "candidate_sigma_min": float(singular[keep][-1]),
        "ritz_min": float(values[0]),
        "ritz_max": float(values[retained - 1]),
    }


class RecycleSpace:
    """Own only past accepted information, with fixed retained and window limits.

    begin transfers the previous space into the new inactive set. Capture stores
    the last window new directions. Finish combines the actually deployed coarse
    space with these directions and selects at most rank vectors. The coarse
    space remains fixed throughout each CG solve, including residual restarts.
    A failed solve clears history. Returned metrics contain no basis arrays.
    """

    def __init__(self, rank, window, device="cpu", tolerance=1e-12):
        self.rank = integer(rank, "Retained rank")
        self.window = integer(window, "Direction window", 1)
        self.tolerance = positive_real(tolerance, "Numerical rank tolerance")
        if device not in {"cpu", "cuda"}:
            raise ValueError("Recycle device must be cpu or cuda")
        self.device = device
        self.clear()

    def clear(self):
        self.indices = np.empty(0, dtype=np.int64)
        self.basis = np.empty((0, 0))
        self.current = None
        self.directions = deque(maxlen=self.window)
        self.observed = 0

    def begin(self, indices):
        self.current = inactive_indices(indices)
        self.directions.clear()
        self.observed = 0
        return transfer_basis(self.basis, self.indices, self.current)

    def capture(self, direction):
        if self.current is None or direction.shape != (len(self.current),):
            raise ValueError("Begin the recycle solve before capturing matching directions")
        self.observed += 1
        if self.rank:
            self.directions.append(
                direction.detach().clone() if self.device == "cuda" else direction.copy()
            )

    def finish(self, A, diagonal, deployed, status):
        if self.current is None:
            raise ValueError("Begin the recycle solve before selecting directions")
        start = time.perf_counter()
        metrics = {
            "policy": "existing-coarse-plus-new-directions-jacobi-ritz-v1",
            "existing_columns": 0 if deployed is None else deployed.shape[1],
            "new_directions_observed": self.observed,
            "new_directions_retained": len(self.directions),
            "history_bytes_before": self.basis.nbytes,
            "direction_bytes": sum(
                p.numel() * p.element_size() if self.device == "cuda" else p.nbytes
                for p in self.directions
            ),
            "numerical_rank_tolerance": self.tolerance,
        }
        if status != "converged":
            self.clear()
            return {**metrics, "selected_rank": 0, "status": "discarded_after_failed_solve"}
        blocks = ([] if deployed is None else [deployed]) + list(self.directions)
        if self.device == "cpu":
            candidates = np.column_stack(blocks) if blocks else np.empty((len(self.current), 0))
            selected, spectral = jacobi_ritz(A, diagonal, candidates, self.rank, self.tolerance)
            selection_seconds = time.perf_counter() - start
            download_seconds = 0.0
            candidate_bytes = candidates.nbytes
        else:
            from .gpu import require_cuda

            torch = require_cuda()
            candidates = (
                torch.column_stack(blocks)
                if blocks
                else torch.empty((len(self.current), 0), device="cuda", dtype=torch.float64)
            )
            selected, spectral = gpu_jacobi_ritz(
                A, diagonal, candidates, self.rank, torch, self.tolerance
            )
            torch.cuda.synchronize()
            selection_seconds = time.perf_counter() - start
            download_start = time.perf_counter()
            selected = selected.cpu().numpy()
            torch.cuda.synchronize()
            download_seconds = time.perf_counter() - download_start
            candidate_bytes = candidates.numel() * candidates.element_size()
        self.indices, self.basis = self.current, selected
        self.current = None
        self.directions.clear()
        return {
            **metrics,
            **spectral,
            "status": "selected",
            "candidate_columns": candidates.shape[1],
            "candidate_bytes": candidate_bytes,
            "history_bytes_after": self.basis.nbytes,
            "selection_seconds": selection_seconds,
            "selection_download_seconds": download_seconds,
        }
