"""Optional damped BFGS curvature corrections to a Gauss--Newton model.

This is a separate nonlinear pilot policy. The unmodified Gauss--Newton model
remains the default. Each construction starts from the current positive normal
operator and applies the retained secants in chronological order.
"""

import numpy as np

from .coupled_derivatives import GaussNewtonOperator


class SecantGaussNewton(GaussNewtonOperator):
    """Apply positive-curvature, Powell-damped BFGS updates matrix-free."""

    def __init__(self, base, pairs):
        super().__init__(base.jacobian, base.weights, base.alpha, base.damping)
        self.corrections = []
        self.secant_diagnostics = []
        for displacement, change in pairs:
            s, y = np.asarray(displacement), np.asarray(change)
            if s.shape != self.weights.shape or y.shape != s.shape:
                raise ValueError("Secants must match the full temperature trajectory")
            if not np.isfinite([s, y]).all():
                raise ValueError("Secants must be finite")
            q = self @ s
            curvature, observed = float(s @ q), float(s @ y)
            threshold = 64 * np.finfo(float).eps * np.linalg.norm(s) * np.linalg.norm(q)
            if curvature <= threshold:
                self.secant_diagnostics.append({"status": "negligible_model_curvature"})
                continue
            weight = (
                1.0 if observed >= 0.2 * curvature else 0.8 * curvature / (curvature - observed)
            )
            adjusted = weight * y + (1 - weight) * q
            denominator = float(s @ adjusted)
            if not np.isfinite(denominator) or denominator <= 0:
                self.secant_diagnostics.append({"status": "nonpositive_adjusted_curvature"})
                continue
            self.corrections.append((q / np.sqrt(curvature), adjusted / np.sqrt(denominator)))
            self.secant_diagnostics.append(
                {
                    "status": "retained",
                    "powell_weight": weight,
                    "curvature_ratio": observed / curvature,
                }
            )
        if self.corrections:
            self.negative = np.column_stack([q for q, _ in self.corrections])
            self.positive = np.column_stack([y for _, y in self.corrections])
        else:
            self.negative = self.positive = np.empty((len(self.weights), 0))

    def _matmat(self, vectors):
        result = super()._matmat(vectors)
        # During construction, only previously processed secants are present.
        for q, y in self.corrections:
            result -= q[:, None] * (q @ vectors)
            result += y[:, None] * (y @ vectors)
        return result

    def _matvec(self, vector):
        return self._matmat(np.asarray(vector).reshape(-1, 1)).ravel()

    def _rmatvec(self, vector):
        return self._matvec(vector)
