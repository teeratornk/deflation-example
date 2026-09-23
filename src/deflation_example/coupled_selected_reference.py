"""Fixed full-domain Ritz references selected with a nominal coupled operator.

Selection takes place in Jacobi coordinates. Candidate directions and selected
directions are mapped back to physical coordinates before any mask restriction.
The selected reference remains unchanged across subsequent nonlinear iterates.
"""

import time

import numpy as np
from scipy import linalg

from .validation import integer, positive_real


class SelectedReference:
    """A compact linear combination of an unchanged candidate reference."""

    def __init__(self, candidates, coefficients, description):
        coefficients = np.array(coefficients, dtype=float, copy=True)
        if (
            coefficients.ndim != 2
            or coefficients.shape[0] != candidates.rank
            or not np.isfinite(coefficients).all()
        ):
            raise ValueError("Finite coefficients must match the candidate reference")
        self.candidates = candidates
        self.coefficients = coefficients
        self.rank = coefficients.shape[1]
        self.description = {**candidates.description, **description}

    def restrict(self, indices):
        return self.candidates.restrict(indices) @ self.coefficients

    def storage(self):
        return {
            **self.candidates.storage(),
            "selection_coefficient_bytes": self.coefficients.nbytes,
        }


def selected_reference(
    candidates, operator, diagonal, rank, *, tolerance=1e-12, chunk=5, block_action=None
):
    """Solve the projected scaled eigenproblem, using QR and an SVD of its R.

    If C contains physical candidate columns, orthogonalize sqrt(D) C. For
    physical V with V.T D V = I, the projected scaled operator is V.T H V.
    This avoids a squared-condition-number Gram-matrix rank test. The optional
    block action must apply precisely H, including any declared corrections.
    """
    start = time.perf_counter()
    rank = integer(rank, "Requested reference rank", 1)
    chunk = integer(chunk, "Operator block width", 1)
    tolerance = positive_real(tolerance, "Relative singular-value threshold")
    diagonal = np.asarray(diagonal, dtype=float)
    if (
        diagonal.ndim != 1
        or operator.shape != (len(diagonal),) * 2
        or not np.isfinite(diagonal).all()
        or np.any(diagonal <= 0)
        or tolerance >= 1
        or rank > candidates.rank
    ):
        raise ValueError("Selection requires a positive diagonal and sufficient candidates")
    tick = time.perf_counter()
    raw = np.asarray(candidates.restrict(np.arange(len(diagonal))), dtype=float)
    if raw.shape != (len(diagonal), candidates.rank) or not np.isfinite(raw).all():
        raise ValueError("Candidate dimensions or values differ")
    materialization = time.perf_counter() - tick
    tick = time.perf_counter()
    Q, R = linalg.qr(np.sqrt(diagonal)[:, None] * raw, mode="economic", check_finite=False)
    del Q
    _, s, vh = linalg.svd(R, full_matrices=False, check_finite=False)
    keep = s > tolerance * s[0] if len(s) and s[0] else np.zeros(len(s), dtype=bool)
    transform = vh[keep].T / s[keep]
    independent = raw @ transform
    del raw
    orthogonalization = time.perf_counter() - tick
    effective = independent.shape[1]
    projected = np.empty((effective, effective))
    tick = time.perf_counter()
    action = (lambda x: operator @ x) if block_action is None else block_action
    for first in range(0, effective, chunk):
        last = min(first + chunk, effective)
        product = np.asarray(action(independent[:, first:last]))
        if product.shape != independent[:, first:last].shape or not np.isfinite(product).all():
            raise ValueError("Nonfinite or incompatible nominal operator product")
        projected[:, first:last] = independent.T @ product
    products = time.perf_counter() - tick
    skew = linalg.norm(projected - projected.T) / max(linalg.norm(projected), 1e-300)
    if not np.isfinite(skew) or skew > 1e-9:
        raise ValueError("The projected nominal operator fails symmetry verification")
    eigenvalues, vectors = linalg.eigh((projected + projected.T) / 2)
    if effective and eigenvalues[0] <= 0:
        raise ValueError("The projected nominal operator must be positive definite")
    deployed = min(rank, effective)
    coefficients = transform @ vectors[:, :deployed]
    total = time.perf_counter() - start
    return SelectedReference(
        candidates,
        coefficients,
        {
            "selection": "nominal_coupled_jacobi_ritz",
            "requested_rank": rank,
            "candidate_rank": candidates.rank,
            "independent_candidates": effective,
            "deployed_rank": deployed,
            "relative_singular_value_threshold": tolerance,
            "minimum_retained_candidate_singular_value": float(s[keep][-1]) if effective else None,
            "ritz_values": eigenvalues[:deployed].tolist(),
            "projected_relative_skew": float(skew),
            "construction_seconds": total,
            "construction_components_seconds": {
                "candidate_materialization": materialization,
                "scaled_orthogonalization": orthogonalization,
                "operator_products_and_projection": products,
                "eigensolve_and_bookkeeping": total
                - materialization
                - orthogonalization
                - products,
            },
            "scope": "Fixed nominal selection within the declared thermal candidate span; approximate coupled modes.",
        },
    )


class CoupledBlockAction:
    """One operator's GPU resources, with explicit lifetime and bounded width."""

    def __init__(self, operator):
        from .coupled_cuda import CudaControlJacobian, CudaGaussNewton

        self.jacobian = CudaControlJacobian(operator.jacobian)
        self.operator = CudaGaussNewton(
            self.jacobian,
            operator.weights,
            operator.alpha,
            operator.damping,
            corrections=getattr(operator, "corrections", ()),
        )

    def __call__(self, vectors):
        value = self.jacobian.cp.asnumpy(self.operator.apply(vectors))
        self.jacobian.cp.cuda.get_current_stream().synchronize()
        return value

    def close(self):
        self.jacobian.close()
        self.jacobian.cp.cuda.get_current_stream().synchronize()


def configured_selected_reference(problem, cfg, baseline, initial_guess=None, initial_state=None):
    """Construct once from the same initial temperature used by optimization.

    The extra nominal flow evaluation is charged to reference construction.
    It supplies no secant history or nonlinear warm-start advantage.
    """
    from .coupled_bounds import temperature_bounds
    from .coupled_derivatives import GaussNewtonOperator
    from .coupled_reference import configured_reference
    from .coupled_optimize import equation_acceptance

    if cfg.get("reference_transfer", "full") != "full" or not cfg["queries"]:
        raise ValueError("Nominal selection requires full-reference transfer and declared targets")
    if len({q["upper_K"] for q in cfg["queries"]}) != 1:
        raise ValueError("The nominal reference sequence holds the physical bound fixed")
    start = time.perf_counter()
    candidate_count = integer(cfg.get("reference_candidates", 400), "Candidate count", 1)
    candidates = configured_reference(problem, {**cfg, "rank": candidate_count}, baseline)
    construction = time.perf_counter() - start
    tick = time.perf_counter()
    bounds = temperature_bounds(cfg, upper_K=cfg["queries"][0]["upper_K"])
    lower = (
        bounds["optimization_lower_K"] - problem.temperature_offset
    ) / problem.temperature_scale
    upper = (
        bounds["optimization_upper_K"] - problem.temperature_offset
    ) / problem.temperature_scale
    state = (
        initial_guess.state
        if initial_guess is not None
        else initial_state
        if initial_state is not None
        else np.zeros(problem.size)
    )
    evaluation = problem.evaluate(np.clip(state, lower, upper), initial=initial_guess)
    if not equation_acceptance(problem.verify(evaluation), cfg):
        raise ValueError("The nominal flow evaluation fails independent verification")
    H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
    diagonal = problem.preconditioning_diagonal(evaluation, 0.0)
    evaluation_seconds = time.perf_counter() - tick
    action = None
    try:
        tick = time.perf_counter()
        if cfg["device"] in {"hybrid", "cuda"}:
            action = CoupledBlockAction(H)
        upload = time.perf_counter() - tick
        if cfg.get("reference_selection") == "preconditioned_coupled":
            from .coupled_frozen_preconditioner import frozen_preconditioner_factory
            from .coupled_preconditioned_reference import preconditioned_reference

            inverse = frozen_preconditioner_factory(problem, evaluation, sweeps=3)(
                np.arange(problem.size)
            )
            reference = preconditioned_reference(
                candidates,
                H,
                inverse,
                cfg["rank"],
                tolerance=cfg.get("reference_selection_tolerance", 1e-12),
                chunk=cfg.get("hybrid_block_max_columns", 5),
                block_action=action,
            )
            del inverse
        else:
            reference = selected_reference(
                candidates,
                H,
                diagonal,
                cfg["rank"],
                tolerance=cfg.get("reference_selection_tolerance", 1e-12),
                chunk=cfg.get("hybrid_block_max_columns", 5),
                block_action=action,
            )
    finally:
        if action is not None:
            action.close()
    reference.description.update(
        thermal_candidate_construction_seconds=construction,
        nominal_evaluation_seconds=evaluation_seconds,
        nominal_factor_upload_seconds=upload,
        total_reference_construction_seconds=time.perf_counter() - start,
        nominal_policy="Initial temperature; complete coupled Gauss-Newton; zero damping and no secants",
    )
    return reference
