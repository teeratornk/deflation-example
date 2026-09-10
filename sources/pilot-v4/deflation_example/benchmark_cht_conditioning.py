"""Controlled CHT restriction with a conditional convergence certificate.

This diagnostic uses exact modes of the full Jacobi-scaled CHT operator. The
complete timing study uses analytical reference constructions. The distinction
is recorded: this example tests the sufficient estimate, not its applicability
to every timed reference space.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import linalg, sparse
from threadpoolctl import threadpool_limits

from .benchmark_transfer import correction_diagnostics
from .problems import build_problem
from .reporting import environment, write_report
from .tutorial_spectral import angle_diagnostics
from .validation import integer


def controlled_cht(n=6, counts=(1, 8, 32, 48, 64, 100, 120)):
    n = integer(n, "Interior grid size", 3)
    counts = [integer(k, "Removed nodes", 1) for k in counts]
    if len(set(counts)) != len(counts) or max(counts) >= n**3 - 1:
        raise ValueError("Use distinct restriction sizes leaving at least two inactive nodes")
    problem = build_problem("cht", n)
    H = problem.H.toarray()
    root = np.sqrt(H.diagonal())
    M0 = H / root[:, None] / root[None, :]
    eigenvalues, vectors = linalg.eigh(M0)
    phi = vectors[:, :1]
    G0 = M0 @ phi - phi * eigenvalues[0]
    ordering = np.argsort(np.abs(phi[:, 0]), kind="stable")
    rows = []
    for count in counts:
        J = np.sort(ordering[:count])
        I = np.setdiff1d(np.arange(n**3), J)
        B, W = M0[np.ix_(I, I)], phi[I]
        row = angle_diagnostics(B, W)
        coupling = M0[np.ix_(I, J)] @ phi[J]
        lhs = B @ W - W * eigenvalues[0]
        rhs = -coupling + G0[I]
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-13)
        sigma = row["minimum_restricted_singular_value"]
        eta = row["separation"]
        numerator = float(linalg.norm(coupling, "fro") + linalg.norm(G0[I], "fro"))
        sufficient = min(1.0, numerator / (sigma * eta)) if eta > 0 else None
        values = linalg.eigvalsh(B)
        original = float(values[-1] / values[0])
        predicted = (
            float((np.sqrt(values[-1] / values[1]) + np.sqrt(original) * sufficient) ** 2)
            if sufficient is not None
            else None
        )
        # correction_diagnostics scales its input. Use physical H_II and the
        # consistently mapped physical vector so that its scaled matrix is B.
        physical = sparse.csr_matrix(H[np.ix_(I, I)])
        exact = np.linspace(0.1, 1.0, len(I))
        measured = correction_diagnostics(
            physical, physical @ exact, phi[I] / root[I, None], 1, exact, n**3
        )
        actual = measured["effective_condition"]
        if predicted is not None and actual > predicted * (1 + 1e-9):
            raise ValueError("Measured deflated condition exceeds the stated sufficient bound")
        row.update(
            removed_nodes=count,
            active_indices=J.tolist(),
            reference_residual_frobenius=float(linalg.norm(G0, "fro")),
            restriction_identity_error=float(linalg.norm(lhs - rhs, "fro")),
            restriction_coupling_frobenius=float(linalg.norm(coupling, "fro")),
            restricted_reference_residual_frobenius=float(linalg.norm(G0[I], "fro")),
            sufficient_angle_bound=sufficient,
            sufficient_bound_status="unavailable"
            if sufficient is None
            else "nontrivial"
            if sufficient < 1
            else "trivial_one",
            original_preconditioned_condition=original,
            deflated_condition_upper_bound=predicted,
            measured_effective_condition=actual,
            certifies_conditioning_improvement=predicted is not None and predicted < original,
        )
        rows.append(row)
    return {
        "protocol": "controlled-cht-conditioning-v1",
        "n": n,
        "rank": 1,
        "alpha": problem.alpha,
        "reference": "exact lowest eigenvector of the full Jacobi-scaled heterogeneous CHT matrix; physical reference is D^(-1/2) phi",
        "restriction": "remove the declared number of nodes in ascending absolute reference amplitude, with stable index tie breaking",
        "coordinates": "Jacobi-scaled; principal restriction preserves the diagonal scaling",
        "scope": "controlled spectral experiment; analytical reference spaces used for complete timings are different constructions; a conditioning certificate does not certify a runtime improvement",
        "formula": "epsilon=min(1,(norm(M0_IJ phi_J,F)+norm(G0_I,F))/(sigma_min(phi_I)*eta)) for eta>0; kappa_upper=(sqrt(lambda_max/lambda_2)+sqrt(kappa_original)*epsilon)^2",
        "rows": rows,
    }


def run(output, n=6, counts=(1, 8, 32, 48, 64, 100, 120), threads=1):
    integer(threads, "Threads", 1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=threads):
        result = controlled_cht(n, counts)
        result["environment"] = environment()
        write_report(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 8, 32, 48, 64, 100, 120])
    parser.add_argument("--threads", type=int, default=1)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
