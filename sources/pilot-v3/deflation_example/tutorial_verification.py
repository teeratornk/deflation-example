"""Runnable accuracy examples and summaries of every recorded CHT stopping protocol."""

from collections import Counter
from statistics import median

import numpy as np
from scipy.optimize import minimize

from .benchmark_extended_report import checked_sequences, METHODS
from .solvers import deflated_cg, independent_residual, kkt_metrics, pdas


def recurrence_example():
    """Float32 CG illustrates residual accumulation; production solvers remain float64."""
    B = np.diag(np.array([1.0, 3.0], dtype=np.float32))
    b = np.ones(2, dtype=np.float32)
    x, r, p = np.zeros(2, dtype=np.float32), b.copy(), b.copy()
    rr = r @ r
    trace = []
    tolerance = 1e-10
    for k in range(32):
        Bp = B @ p
        curvature = p @ Bp
        if not np.isfinite(curvature) or curvature <= 0:
            break
        step = rr / curvature
        x += step * p
        r -= step * Bp
        recurrence = float(np.linalg.norm(r) / np.linalg.norm(b))
        fresh = independent_residual(B.astype(float), x.astype(float), b.astype(float))
        trace.append(
            {"iteration": k + 1, "recurrence_relative": recurrence, "original_relative": fresh}
        )
        if recurrence <= tolerance:
            break
        new_rr = r @ r
        p = r + (new_rr / rr) * p
        rr = new_rr
    checked = deflated_cg(B.astype(float), b.astype(float), rtol=tolerance)
    return {
        "scope": "two-dimensional float32 teaching recurrence; separate float64 production solve",
        "matrix": B,
        "rhs": b,
        "relative_tolerance": tolerance,
        "trace": trace,
        "recurrence_candidate": trace[-1]["recurrence_relative"] <= tolerance,
        "original_acceptance": trace[-1]["original_relative"] <= tolerance,
        "production_status": checked.status,
        "production_original_residual": checked.residual,
    }


def accuracy_examples():
    H, f = np.array([[2.0, -1.0], [-1.0, 2.0]]), np.array([0.2, 2.9])
    result = pdas(H, f, 1.0, initial_active=[True, True])
    reference = minimize(
        lambda y: (0.5 * y @ H @ y - f @ y, H @ y - f),
        np.zeros(2),
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, 1.0)] * 2,
        options={"gtol": 1e-11, "ftol": 1e-15, "maxiter": 2000},
    )
    g = H @ reference.x - f
    multiplier = np.where(reference.x >= 1 - 1e-12, -g, 0)
    # The exact minimizer is zero; a small objective difference coexists with a larger gradient.
    illustrative_H, trial = np.diag([1.0, 1e6]), np.array([0.0, 1e-9])
    projection_B = np.array([[2.0, 1.0], [1.0, 2.0]])
    Z = np.eye(2)[:, :1]
    Q = Z @ np.linalg.solve(Z.T @ projection_B @ Z, Z.T)
    P = np.eye(2) - Q @ projection_B
    projection_b = projection_B @ Z[:, 0]
    corrected = deflated_cg(projection_B, projection_b, Z, projection_B.diagonal())
    return {
        "active_set_release": {
            "matrix": H,
            "rhs": f,
            "bound": 1.0,
            "initial_active": [True, True],
            "initial_multiplier": f - H @ np.ones(2),
            "solution": result["y"],
            "status": result["status"],
            "history": result["history"],
            "kkt": result["kkt"],
        },
        "independent_optimizer": {
            "method": "L-BFGS-B",
            "success": bool(reference.success),
            "termination_message": str(reference.message),
            "solution": reference.x,
            "objective_difference": abs(reference.fun - result["objective"]),
            "separately_computed_kkt": kkt_metrics(H, f, 1.0, reference.x, multiplier),
        },
        "objective_and_stationarity": {
            "scope": "controlled quadratic illustration",
            "matrix": illustrative_H,
            "exact_minimizer": [0.0, 0.0],
            "trial": trial,
            "objective_error": float(0.5 * trial @ illustrative_H @ trial),
            "gradient_inf": float(np.linalg.norm(illustrative_H @ trial, np.inf)),
        },
        "residual_recomputation": recurrence_example(),
        "projected_equation_without_coarse_correction": {
            "scope": "exact algebraic check of the corrected-initial-guess requirement",
            "matrix": projection_B,
            "rhs": projection_b,
            "basis": Z,
            "zero_guess_projected_residual": float(np.linalg.norm(P.T @ projection_b)),
            "zero_guess_original_relative_residual": independent_residual(
                projection_B, np.zeros(2), projection_b
            ),
            "corrected_solution": corrected.x,
            "corrected_status": corrected.status,
            "corrected_original_relative_residual": corrected.residual,
        },
    }


def stopping_examples(records):
    groups = []
    for version in (1, 2, 3):
        relative = f"extended/cht_v{version}/results.json"
        index = records.load(relative)
        for record in index["sequences"]:
            records.load(f"extended/cht_v{version}/" + record["path"])
        if records.load(f"extended/cht_v{version}/protocol.json") != index["specification"]:
            raise ValueError("Stored CHT protocol differs from the experiment specification")
        report, sequences = checked_sequences(records.path(relative), require_clean_source=True)
        rows, attempts = [], []
        for n in report["specification"]["controls"]["grids"]:
            for warm in report["specification"]["controls"]["warm_starts"]:
                for method in METHODS:
                    runs = [
                        s
                        for s in sequences
                        if (s["n"], s["warm_start"], s["method"]) == (n, warm, method)
                    ]
                    accepted = [s["total_seconds"] for s in runs if s["success"]]
                    failures = Counter(
                        c["status"] for s in runs for c in s["cases"] if c["status"] != "converged"
                    )
                    rows.append(
                        {
                            "n": n,
                            "warm_start": warm,
                            "method": method,
                            "passed_sequences": len(accepted),
                            "attempted_sequences": len(runs),
                            "accepted_median_seconds": median(accepted) if accepted else None,
                            "failed_target_statuses": dict(failures),
                        }
                    )
        for sequence in sequences:
            attempts.append(
                {
                    k: sequence[k]
                    for k in ("n", "warm_start", "method", "repetition", "success", "total_seconds")
                }
            )
        groups.append(
            {
                "protocol": report["specification"],
                "measurement_environment": report["environment"],
                "rows": rows,
                "attempts": attempts,
            }
        )
    return groups
