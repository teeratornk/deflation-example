"""Self-contained small coupled verification; no physical performance claim."""

import argparse
from pathlib import Path

import numpy as np

from .axisymmetric_flow import AxisymmetricFlow
from .coupled_control import CoupledControlProblem
from .coupled_derivatives import GaussNewtonOperator
from .coupled_optimize import adjoint_acceptance, equation_acceptance
from .coupled_trust import minimize_trust
from .meshes import ThermalMesh
from .reporting import environment, write_arrays, write_report
from .study_solvers import ArrayReference, StudySolver


def verification_problem():
    """Two time slabs on an annular test mesh with nondimensional coefficients."""
    n = 3
    r, z = np.meshgrid(np.linspace(1, 2, n + 1), np.linspace(0, 1, n + 1))
    nodes = np.column_stack((r.ravel(), z.ravel()))
    a = (np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]).ravel()
    cells = np.vstack(
        (np.column_stack((a, a + 1, a + n + 2)), np.column_stack((a, a + n + 2, a + n + 1)))
    )
    mesh = ThermalMesh(nodes, cells, np.zeros(len(cells), dtype=int), np.arange(n + 1), True)
    flow = AxisymmetricFlow(mesh, 0.1)
    velocity = np.column_stack((np.zeros(flow.nv), np.full(flow.nv, 0.02)))
    baseline = flow.solve(
        np.zeros_like(flow.quadrature_points),
        flow.boundary,
        velocity[flow.boundary],
        pressure_gauge=(0, 0),
        method="newton",
    )
    if baseline.status != "converged":
        raise RuntimeError("The verification baseline failed")
    count = len(mesh.cells)
    return CoupledControlProblem(
        flow,
        np.tile(np.diag([0.07, 0.12]), (count, 1, 1)),
        np.ones(count),
        np.linspace(0, 0.02, count),
        flow.boundary,
        velocity[flow.boundary],
        baseline,
        alpha=0.003,
        expansion=0.003,
        temperature_scale=2.0,
        physical_steps=[0.2, 0.35],
        time_scale=3.0,
        velocity_scale=5.0,
        pressure_gauge=(0, 0),
        initial_temperature=np.full(len(mesh.free), 0.04),
        thermal_boundary=0.03,
        flow_tolerance=1e-11,
    )


def run(device="cpu"):
    if device not in {"cpu", "cuda", "hybrid"}:
        raise ValueError("Choose cpu, cuda or hybrid")
    problem = verification_problem()
    desired = np.linspace(-0.2, 0.4, problem.size)
    initial = np.full(problem.size, 0.04)
    evaluation = problem.evaluate(initial)
    H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
    dense = H @ np.eye(problem.size)
    _, vectors = np.linalg.eigh((dense + dense.T) / 2)
    reference = ArrayReference(vectors[:, :3], {"construction": "small exact initial reference"})
    baseline = StudySolver("jacobi", rtol=1e-10, residual_policy="refine")
    options = dict(reference=reference, rank=3, rtol=1e-10, residual_policy="refine")
    if device == "cuda":
        from .coupled_cuda_solver import CudaCoupledSolver

        tested = CudaCoupledSolver("reference", frozen_layout="block_diagonal", **options)
    elif device == "hybrid":
        from .coupled_hybrid_solver import HybridCoupledSolver

        tested = HybridCoupledSolver(
            "reference", block_device="cpu", coarse_device="cuda", **options
        )
    else:
        tested = StudySolver("reference", **options)
    try:
        expected = minimize_trust(
            problem, desired, -0.05, 0.15, baseline, initial=initial, max_iterations=80
        )
        actual = minimize_trust(
            problem,
            desired,
            -0.05,
            0.15,
            tested,
            initial=initial,
            max_iterations=80,
            qp_solver="projected",
            accuracy="adaptive_projected",
        )
        equations = problem.verify(actual.evaluation)
        adjoint = problem.verify_adjoint(actual.evaluation, desired)
        difference = float(np.max(np.abs(actual.evaluation.state - expected.evaluation.state)))
        verified = (
            actual.status == expected.status == "converged"
            and max(actual.kkt.values()) <= 1e-8
            and difference <= 2e-7
            and equation_acceptance(equations, {})
            and adjoint_acceptance(adjoint)
        )
        return {
            "schema": "coupled-projected-verification-v1",
            "device": device,
            "status": "verified" if verified else "verification_failed",
            "baseline_status": expected.status,
            "projected_status": actual.status,
            "kkt": actual.kkt,
            "equations": equations,
            "adjoint": adjoint,
            "state_maximum_difference": difference,
            "scope": "Small discrete verification with prescribed test coefficients; no timing or physical-resolution claim.",
        }, {
            "state": actual.evaluation.state,
            "baseline_state": expected.evaluation.state,
            "control": actual.evaluation.control,
            "desired": desired,
        }
    finally:
        tested.close()
        baseline.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "hybrid"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report, fields = run(args.device)
    write_report(args.output / "record.json", {**report, "environment": environment()})
    write_arrays(args.output / "fields.npz", **fields)
    if report["status"] != "verified":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
