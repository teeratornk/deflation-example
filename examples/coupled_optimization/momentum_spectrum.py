"""Inspect slow modes of the linearized discrete momentum equations.

For M dv/dt + J dv = 0, nonzero eigenvalues mu of J^(-1) M give
decay rates lambda=1/mu. A negative real part indicates growth in this
linearized discretization. The test does not certify a physical instability.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs, splu
from threadpoolctl import threadpool_limits

from deflation_example.assess_transformer import transformer_boundaries
from deflation_example.axisymmetric_flow import AxisymmetricFlow
from deflation_example.coupled_pilot import transformer_inputs
from deflation_example.oil_properties import momentum_reference
from deflation_example.reporting import environment, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fields", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(4):
        record = json.loads((args.baseline / "record.json").read_text())
        mesh, _, _, _, _, _ = transformer_inputs(record["configuration"]["level"])
        flow = AxisymmetricFlow(
            mesh,
            momentum_reference()["kinematic_viscosity_m2_s"],
            convection_form=record["configuration"]["convection_form"],
            grad_div=record.get("grad_div_coefficient_m2_s", 0),
        )
        fixed, _ = transformer_boundaries(flow, record["inlet_velocity_m_s"])
        free = np.setdiff1d(np.arange(flow.size), np.r_[fixed, flow.nv + fixed])
        with np.load(args.fields or args.baseline / "baseline.npz", allow_pickle=False) as data:
            velocity = data["velocity"].copy()
        J = (flow.operator(velocity) + flow.convection_derivative(velocity))[free][:, free].tocsc()
        M = sparse.block_diag(
            (flow.mass, flow.mass, sparse.csr_matrix((flow.np, flow.np))), format="csr"
        )[free][:, free]
        factor = splu(J)
        operator = LinearOperator(J.shape, matvec=lambda x: factor.solve(M @ x), dtype=float)
        mu, vectors = eigs(
            operator,
            k=6,
            which="LM",
            tol=1e-9,
            maxiter=2000,
            v0=np.random.default_rng(702).normal(size=len(free)),
        )
        rates = 1 / mu
        rows = []
        for rate, vector in zip(rates, vectors.T, strict=True):
            Jv, Mv = J @ vector, M @ vector
            rows.append(
                {
                    "decay_rate_real_per_s": float(rate.real),
                    "decay_rate_imag_per_s": float(rate.imag),
                    "relative_pencil_residual": float(
                        np.linalg.norm(Jv - rate * Mv)
                        / (np.linalg.norm(Jv) + abs(rate) * np.linalg.norm(Mv))
                    ),
                }
            )
        write_report(
            args.output,
            {
                "environment": environment(),
                "modes": rows,
                "kinetic_diagnostics": flow.kinetic_diagnostics(velocity),
                "scope": "Six inverse-pencil modes nearest zero; this is not a complete stability spectrum.",
            },
        )


if __name__ == "__main__":
    main()
