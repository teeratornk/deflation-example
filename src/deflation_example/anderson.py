"""Small-history, weighted Anderson acceleration with rank-revealing QR.

This module proposes iterates only. Callers accept them using fresh residuals
of their original equations. Zero weights allow auxiliary variables, such as
pressure, to follow the same affine combination without mixing physical units.
"""

import numpy as np
from scipy.linalg import qr, solve_triangular

from .validation import integer, positive_real


class Anderson:
    def __init__(self, weights, depth=3, damping=0.5, rank_tolerance=1e-12):
        self.weights = np.asarray(weights, dtype=float).copy()
        if (
            self.weights.ndim != 1
            or not np.isfinite(self.weights).all()
            or np.any(self.weights < 0)
            or not np.any(self.weights > 0)
        ):
            raise ValueError("Anderson weights must be finite, nonnegative and nonzero")
        self.root = np.sqrt(self.weights / self.weights.max())
        self.depth = integer(depth, "Anderson history", 1)
        self.damping = positive_real(damping, "Anderson damping")
        self.rank_tolerance = positive_real(rank_tolerance, "Anderson rank tolerance")
        if self.damping > 1 or self.rank_tolerance >= 1:
            raise ValueError("Damping must not exceed one; rank tolerance must be below one")
        self.reset()

    def reset(self):
        self.xs, self.fs = [], []

    def propose(self, x, mapped):
        x, mapped = np.asarray(x, dtype=float), np.asarray(mapped, dtype=float)
        if x.shape != self.weights.shape or mapped.shape != x.shape:
            raise ValueError("Anderson iterate dimensions differ from weights")
        if not np.isfinite(x).all() or not np.isfinite(mapped).all():
            raise ValueError("Anderson iterates must be finite")
        f = mapped - x
        self.xs.append(x.copy())
        self.fs.append(f.copy())
        self.xs, self.fs = self.xs[-self.depth - 1 :], self.fs[-self.depth - 1 :]
        plain = x + self.damping * f
        info = {"history_rank": 0, "proposal": "relaxed", "coefficient_l1": 0.0}
        if len(self.xs) == 1:
            return plain, info
        dx = np.diff(np.stack(self.xs), axis=0).T
        df = np.diff(np.stack(self.fs), axis=0).T
        q, r, piv = qr(self.root[:, None] * df, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(r))
        rank = int(np.sum(diagonal > self.rank_tolerance * diagonal.max(initial=0)))
        info["history_rank"] = rank
        if not rank:
            return plain, info
        chosen = piv[:rank]
        gamma = solve_triangular(r[:rank, :rank], q[:, :rank].T @ (self.root * f))
        info["coefficient_l1"] = float(np.abs(gamma).sum())
        if not np.isfinite(gamma).all() or info["coefficient_l1"] > 10:
            info["proposal"] = "coefficient_safeguard"
            self.reset()
            return plain, info
        accelerated = plain - (dx[:, chosen] + self.damping * df[:, chosen]) @ gamma
        if not np.isfinite(accelerated).all():
            self.reset()
            info["proposal"] = "nonfinite_safeguard"
            return plain, info
        info["proposal"] = "anderson"
        return accelerated, info
