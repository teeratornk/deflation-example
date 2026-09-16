"""Local derivative of the segregated temperature fixed-point map.

This is an iteration spectrum, not a physical-time amplification spectrum.
For H=[F,-L; S,T], the temperature map derivative is -T^{-1} S F^{-1} L.
Computed modes with modulus above one demonstrate local amplification; a
partial spectrum wholly inside the unit circle does not certify contraction.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs, splu
from threadpoolctl import threadpool_limits

from step_diagnostics import checked_field, load_case
from deflation_example.coupled_newton_replay import criteria_met, step_equations
from deflation_example.coupled_resolution import forward_model
from deflation_example.coupled_saved import file_digest
from deflation_example.coupled_step_spectrum import step_linearization
from deflation_example.reporting import environment, write_report


def fixed_point_operator(H, nf):
    if not 0 < nf < H.shape[0] or H.shape[0] != H.shape[1]:
        raise ValueError("Invalid coupled block dimensions")
    H = sparse.csr_matrix(H)
    F, L, S, T = H[:nf, :nf], -H[:nf, nf:], H[nf:, :nf], H[nf:, nf:]
    fs = 1 / np.maximum(abs(F).max(axis=1).toarray().ravel(), np.finfo(float).tiny)
    ts = 1 / np.maximum(abs(T).max(axis=1).toarray().ravel(), np.finfo(float).tiny)
    ff, tf = splu((sparse.diags(fs) @ F).tocsc()), splu((sparse.diags(ts) @ T).tocsc())

    def action(x):
        if np.iscomplexobj(x):
            return action(x.real) + 1j * action(x.imag)
        return -tf.solve(ts * (S @ ff.solve(fs * (L @ x))))

    operator = LinearOperator(T.shape, matvec=action, dtype=float)
    rng = np.random.default_rng(794)
    rhs = rng.normal(size=F.shape[0])
    delta = ff.solve(fs * rhs)
    residual = np.linalg.norm(fs * (F @ delta - rhs)) / np.linalg.norm(fs * rhs)
    return operator, {
        "flow_block_dimension": F.shape[0],
        "thermal_block_dimension": T.shape[0],
        "flow_factor_nonzeros": ff.L.nnz + ff.U.nnz,
        "thermal_factor_nonzeros": tf.L.nnz + tf.U.nnz,
        "flow_test_linear_relative_residual": float(residual),
    }


def spectrum(H, nf, count=4):
    operator, report = fixed_point_operator(H, nf)
    if operator.shape[0] < count + 2:
        raise ValueError("Too few thermal unknowns for requested sparse spectrum")
    try:
        values, vectors = eigs(
            operator,
            k=count,
            which="LM",
            tol=1e-6,
            maxiter=300,
            v0=np.random.default_rng(321).normal(size=operator.shape[0]),
        )
        status = "converged"
    except ArpackNoConvergence as error:
        values, vectors = error.eigenvalues, error.eigenvectors
        status = "eigensolver_iteration_cap"
    rows = []
    for value, vector in zip(values, vectors.T, strict=True):
        image = operator @ vector
        residual = np.linalg.norm(image - value * vector) / max(
            np.linalg.norm(image) + abs(value) * np.linalg.norm(vector), 1e-30
        )
        rows.append(
            {
                "real": float(value.real),
                "imaginary": float(value.imag),
                "modulus": float(abs(value)),
                "eigen_residual": float(residual),
                "relaxed_moduli": {str(w): float(abs(1 - w + w * value)) for w in (0.25, 0.5, 1.0)},
            }
        )
    return {**report, "status": status, "requested_modes": count, "modes": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "fine-baseline", "optimization", "trajectory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repair", type=Path)
    parser.add_argument("--slab", type=int)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = {
        "status": "running",
        "environment": environment(),
        "script_sha256": file_digest(Path(__file__)),
        "scope": __doc__,
    }
    write_report(args.output / "record.json", result)
    try:
        with threadpool_limits(args.threads):
            problem, source, previous, previous_flow, state, flow, n, record = load_case(args)
            result.update(
                slab_zero_based=n,
                time_s=record["steps"][n]["time_s"],
                trajectory_record_sha256=file_digest(args.trajectory / "record.json"),
                field_sha256=record["steps"][n]["field_sha256"],
            )
            if args.repair:
                repair = json.loads((args.repair / "record.json").read_text())
                if not (
                    repair["status"] == "complete"
                    and repair["verified"]
                    and repair["trajectory_record_sha256"] == result["trajectory_record_sha256"]
                    and repair["initial_field_sha256"] == result["field_sha256"]
                    and repair["slab_zero_based"] == n
                ):
                    raise ValueError("Repair must verify the same unchanged step")
                state, flow = checked_field(
                    args.repair,
                    {"fields": "fields.npz", "field_sha256": repair["field_sha256"]},
                    problem,
                )
                result.update(
                    repair_record_sha256=file_digest(args.repair / "record.json"),
                    field_sha256=repair["field_sha256"],
                )
            _, checks = step_equations(
                problem, forward_model(problem), state, flow, source, previous, previous_flow, n
            )
            if args.repair and not criteria_met(checks, 1e-12):
                raise ValueError("Repaired field fails independent original equations")
            result.update(equations=checks, at_verified_root=bool(criteria_met(checks, 1e-12)))
            write_report(args.output / "record.json", result)
            H = step_linearization(problem, state[problem.free], flow.velocity, n)[0]
            result.update(spectrum=spectrum(H, len(problem.flow_free)), status="complete")
    except Exception as error:
        result.update(status="execution_failure", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result["seconds"] = time.perf_counter() - start
        write_report(args.output / "record.json", result)


if __name__ == "__main__":
    main()
