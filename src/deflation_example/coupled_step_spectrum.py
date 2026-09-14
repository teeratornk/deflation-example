"""Linearized one-step amplification at saved coupled states.

The pencil C z = mu H z describes the current fully coupled backward-Euler
linearization. Eigenvalues mu of H^{-1} C are local propagation factors with the
heat source fixed. They do not determine the amplification of a product of
different time-step maps or certify a physical instability.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, ArpackNoConvergence, eigs, splu
from threadpoolctl import threadpool_limits

from .coupled_derivatives import thermal_velocity_jacobian
from .coupled_optimize import load_problem
from .coupled_saved import load_saved_solution, require_matching_baseline
from .reporting import environment, write_report
from .validation import integer


def step_linearization(problem, state, velocity, slab):
    """Return exact current-state and previous-state derivatives of one step."""
    slab = integer(slab, "Slab index", 0)
    if not len(problem.physical_steps) or slab >= problem.slabs:
        raise ValueError("Select an existing transient time slab")
    state = np.asarray(state, dtype=float)
    if state.shape != (problem.spatial_size,) or not np.isfinite(state).all():
        raise ValueError("The current state must match the free temperature degrees of freedom")
    dt = float(problem.physical_steps[slab])
    full = problem.full_temperature(state)
    assembly = problem.assemble(velocity)
    flow_jacobian = problem.flow.operator(velocity, time_step=dt)
    flow_jacobian += problem.flow.convection_derivative(velocity)
    flow_jacobian = flow_jacobian[problem.flow_free][:, problem.flow_free]
    thermal_jacobian = thermal_velocity_jacobian(
        problem.flow, velocity, full, problem.capacity, problem.conductivity, problem.velocity_scale
    )[problem.free][:, problem.flow_free]
    thermal_mass = sparse.diags(assembly.capacity[problem.free] / problem.steps[slab])
    thermal = assembly.stiffness[problem.free][:, problem.free] + thermal_mass
    H = sparse.bmat(
        [[flow_jacobian, -problem.load_derivative], [thermal_jacobian, thermal]], format="csc"
    )
    C = sparse.block_diag((problem.momentum_mass / dt, thermal_mass), format="csr")
    return H, C, thermal.tocsc(), thermal_mass.tocsr()


def amplification_spectrum(H, C, modes=4):
    modes = min(integer(modes, "Requested modes", 1), H.shape[0] - 2)
    if modes < 1:
        raise ValueError("The pencil is too small for this sparse diagnostic")
    factor = splu(H)
    action = LinearOperator(H.shape, matvec=lambda x: factor.solve(C @ x), dtype=float)
    try:
        values, vectors = eigs(
            action,
            k=modes,
            which="LM",
            tol=1e-8,
            maxiter=1000,
            v0=np.random.default_rng(413).normal(size=H.shape[0]),
        )
        status = "converged"
    except ArpackNoConvergence as failure:
        values, vectors = failure.eigenvalues, failure.eigenvectors
        status = "eigensolver_iteration_cap"
    rows = []
    for value, vector in zip(values, vectors.T, strict=True):
        Cv, Hv = C @ vector, H @ vector
        rows.append(
            {
                "amplification_real": float(value.real),
                "amplification_imag": float(value.imag),
                "amplification_modulus": float(abs(value)),
                "pencil_relative_residual": float(
                    np.linalg.norm(Cv - value * Hv)
                    / max(np.linalg.norm(Cv) + abs(value) * np.linalg.norm(Hv), 1e-30)
                ),
            }
        )
    return {
        "status": status,
        "requested_modes": modes,
        "modes": rows,
        "factor_nonzeros": factor.L.nnz + factor.U.nnz,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("reference", "jacobi", "recycling"), default="reference"
    )
    parser.add_argument("--slabs", type=int, nargs="+", default=[0, 15, 27])
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        record, cfg, fields, digest = load_saved_solution(args.optimization, args.method)
        problem, baseline = load_problem({**cfg, "baseline_directory": str(args.baseline)})
        require_matching_baseline(record, baseline)
        state = fields["state"].reshape(problem.slabs, problem.spatial_size)
        with np.load(args.optimization / (args.method + "-fields.npz"), allow_pickle=False) as data:
            velocity = data["velocity"].copy()
        args.output.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema": "coupled-step-amplification-v1",
            "environment": environment(),
            "optimization_source": record["environment"]["git_head"],
            "optimization_field_sha256": digest,
            "scope": "Local linearized fixed-source propagation factors; no full-trajectory or physical stability certificate.",
        }
        rows = []
        write_report(args.output / "record.json", {**metadata, "status": "running", "rows": rows})
        for n in args.slabs:
            H, C, B, T = step_linearization(problem, state[n], velocity[n], n)
            rows.append(
                {
                    "slab_zero_based": n,
                    "coupled": amplification_spectrum(H, C, args.modes),
                    "frozen_velocity_thermal": amplification_spectrum(B, T, args.modes),
                }
            )
            write_report(
                args.output / "record.json", {**metadata, "status": "running", "rows": rows}
            )
        write_report(args.output / "record.json", {**metadata, "status": "complete", "rows": rows})


if __name__ == "__main__":
    main()
