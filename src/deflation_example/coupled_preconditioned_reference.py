"""Fixed full-domain Ritz directions for a declared SPD inverse preconditioner.

For SPD H and K, K H is self-adjoint in the H inner product. On a candidate
space V, its Ritz pencil is ((H V)' K (H V), V' H V). This avoids treating the
matrix of an approximate inverse as the metric of a generalized H eigenproblem.
The construction is a development control, not evidence of a runtime benefit.
"""

import time

import numpy as np
from scipy import linalg

from .study_solvers import ArrayReference
from .validation import integer, positive_real, real_array


def end_indices(size):
    """Nested ordering alternating the smallest and largest Ritz values."""
    low, high, order = 0, size - 1, []
    while low <= high:
        order.append(low)
        if low < high:
            order.append(high)
        low, high = low + 1, high - 1
    return np.asarray(order, dtype=int)


def preconditioned_reference(
    candidates,
    operator,
    inverse,
    rank,
    *,
    chunk=20,
    tolerance=1e-12,
    enrich=True,
    block_action=None,
):
    """Select a fixed reference; optional K H enrichment changes its candidate span.

    K must be one fixed symmetric positive-definite linear map. Both projected
    forms are checked for symmetry and positive definiteness. These checks on
    the candidate space do not establish global SPD for an arbitrary callable.
    """
    start = time.perf_counter()
    rank = integer(rank, "Reference rank", 1)
    chunk = integer(chunk, "Block width", 1)
    tolerance = positive_real(tolerance, "Rank threshold")
    if tolerance >= 1 or not isinstance(enrich, bool) or not callable(inverse):
        raise ValueError("Use a rank threshold below one and a callable SPD inverse")
    n = operator.shape[0]
    if operator.shape != (n, n):
        raise ValueError("The nominal operator must be square")
    raw = real_array(candidates.restrict(np.arange(n)), "Full-domain candidates")
    if raw.shape != (n, candidates.rank) or not np.isfinite(raw).all():
        raise ValueError("Candidate shape or values differ")
    action = (lambda x: operator @ x) if block_action is None else block_action
    action_columns = inverse_columns = 0

    def apply(fn, x, name):
        y = real_array(fn(x), name)
        if y.shape != x.shape or not np.isfinite(y).all():
            raise ValueError(name + " returned nonfinite or incompatible values")
        return y

    if enrich and raw.shape[1]:
        extension = np.empty_like(raw)
        for first in range(0, raw.shape[1], chunk):
            last = min(first + chunk, raw.shape[1])
            extension[:, first:last] = apply(inverse, apply(action, raw[:, first:last], "H"), "K")
            action_columns += last - first
            inverse_columns += last - first
        raw = np.column_stack((raw, extension))
    # Remove column scaling before rank revelation; do not replace lost directions.
    norms = linalg.norm(raw, axis=0)
    raw = raw[:, norms > 0] / norms[norms > 0]
    Q, R = linalg.qr(raw, mode="economic", check_finite=False)
    del raw
    U, singular, _ = linalg.svd(R, full_matrices=False, check_finite=False)
    keep = singular > tolerance * singular[0] if len(singular) else np.zeros(0, dtype=bool)
    V = Q @ U[:, keep]
    del Q, R, U
    effective = V.shape[1]
    HV = np.empty_like(V)
    for first in range(0, effective, chunk):
        last = min(first + chunk, effective)
        HV[:, first:last] = apply(action, V[:, first:last], "H")
        action_columns += last - first
    metric = V.T @ HV
    projected = np.empty((effective, effective))
    for first in range(0, effective, chunk):
        last = min(first + chunk, effective)
        projected[:, first:last] = HV.T @ apply(inverse, HV[:, first:last], "K")
        inverse_columns += last - first
    for name, form in (("energy metric", metric), ("preconditioned Ritz form", projected)):
        skew = linalg.norm(form - form.T) / max(linalg.norm(form), 1e-300)
        if not np.isfinite(skew) or skew > 1e-9:
            raise ValueError(name + " fails symmetry verification")
    if effective:
        eigenvalues, vectors = linalg.eigh((projected + projected.T) / 2, (metric + metric.T) / 2)
        if eigenvalues[0] <= 0:
            raise ValueError("The preconditioned Ritz form must be positive definite")
        chosen = end_indices(effective)[: min(rank, effective)]
        basis = V @ vectors[:, chosen]
    else:
        eigenvalues, chosen = np.empty(0), np.empty(0, dtype=int)
        basis = np.empty((n, 0))
    return ArrayReference(
        basis,
        {
            **candidates.description,
            "selection": "fixed_nominal_preconditioned_energy_ritz",
            "requested_rank": rank,
            "deployed_rank": basis.shape[1],
            "candidate_rank": candidates.rank,
            "independent_candidates": effective,
            "enrichment": "C and K H C" if enrich else "C",
            "rank_threshold": tolerance,
            "ritz_selection": "alternating_low_high",
            "ritz_values": eigenvalues[chosen].tolist(),
            "operator_columns": action_columns,
            "inverse_columns": inverse_columns,
            "construction_seconds": time.perf_counter() - start,
            "scope": "Fixed nominal full-domain reference; projected spectral information, not a convergence certificate for changing inactive systems.",
        },
    )
