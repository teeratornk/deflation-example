"""Separate finite-element operator and fixed-source changes under refinement."""

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from deflation_example.assess_transformer import load_controls
from deflation_example.mesh_refinement import refine, refine_quadratic_flow
from deflation_example.mesh_showcases import _assembly
from deflation_example.reporting import environment, write_report


def diagnose(inputs, levels):
    if len(inputs.ends):
        raise ValueError("This diagnostic requires a steady control")
    mesh, velocity, background = inputs.mesh, inputs.velocity, inputs.background
    coarse = _assembly(mesh, inputs.parameters, velocity, background)
    control, state = inputs.control[0], inputs.state[0]
    coarse_load = coarse.mass * control + coarse.load
    scale = np.linalg.norm(coarse_load[mesh.free])
    if scale == 0:
        raise ValueError("A nonzero thermal load is required")
    P = sparse.eye(len(mesh.nodes), format="csr")
    physical = inputs.parameters["physical"]
    source_factor = (
        physical["oil_conductivity_W_m_K"]
        * physical["temperature_scale_K"]
        / physical["length_scale_m"] ** 2
    )
    rows = []
    for level in range(levels + 1):
        assembly = _assembly(mesh, inputs.parameters, velocity, background)
        u, y0 = P @ control, P @ state
        rhs = assembly.mass * u + assembly.load
        I, J = mesh.free, mesh.dirichlet
        temperature = np.zeros(len(mesh.nodes))
        temperature[I] = spsolve(assembly.stiffness[I][:, I], rhs[I])
        if not np.isfinite(temperature).all():
            raise FloatingPointError("Nonfinite thermal solution")
        residual = assembly.stiffness @ temperature - rhs
        components, matrices = {}, {}
        for name in ("diffusion", "transport", "stabilization"):
            reference = getattr(coarse, name)
            difference = P.T @ getattr(assembly, name) @ P - reference
            components[name] = (difference @ state)[coarse.mesh.free]
            matrices[name] = float(
                sparse.linalg.norm(difference) / max(sparse.linalg.norm(reference), 1e-30)
            )
        components["source_loading"] = -(P.T @ (assembly.mass * u) - coarse.mass * control)[
            coarse.mesh.free
        ]
        components["background_loading"] = -(P.T @ assembly.load - coarse.load)[coarse.mesh.free]
        total = sum(components.values())
        old_residual = coarse.stiffness @ state - coarse_load
        projected = (P.T @ (assembly.stiffness @ y0 - rhs) - old_residual)[coarse.mesh.free]
        fluid = np.unique(mesh.cells[mesh.materials == 0])
        T = physical["inlet_temperature_K"] + physical["temperature_scale_K"] * temperature
        weights = assembly.mass[I]
        rows.append(
            {
                "level": level,
                "nodes": len(mesh.nodes),
                "thermal_original_relative_residual": float(
                    np.linalg.norm(residual[I]) / np.linalg.norm(rhs[I])
                ),
                "peak_temperature_K": float(T[I].max()),
                "minimum_temperature_K": float(T.min()),
                "fluid_temperature_range_K": [float(T[fluid].min()), float(T[fluid].max())],
                "bound_violation_K": float(max(0, T[I].max() - inputs.manifest["bound_K"])),
                "weighted_difference_from_interpolated_saved_state_K": float(
                    physical["temperature_scale_K"]
                    * np.sqrt(weights @ (temperature[I] - y0[I]) ** 2 / weights.sum())
                ),
                "source_range_W_m3": [
                    float(u.min() * source_factor),
                    float(u.max() * source_factor),
                ],
                "source_integral_difference_W": float(
                    (assembly.mass @ u - coarse.mass @ control) * source_factor
                ),
                "galerkin_matrix_relative_changes": matrices,
                "projected_residual_components_relative_norm": {
                    k: float(np.linalg.norm(v) / scale) for k, v in components.items()
                },
                "projected_residual_change_relative_norm": float(np.linalg.norm(projected) / scale),
                "decomposition_identity_relative_error": float(
                    np.linalg.norm(total - projected) / scale
                ),
                "boundary_values_K": np.unique(T[J]).tolist(),
            }
        )
        if level < levels:
            fine, interpolation, parent = refine(mesh)
            velocity = refine_quadratic_flow(mesh, fine, parent, velocity)
            background = background[parent]
            P = interpolation @ P
            mesh = fine
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-directory", type=Path, required=True)
    parser.add_argument("--role", choices=["nominal", "active", "demanding"], default="nominal")
    parser.add_argument("--levels", type=int, choices=range(1, 3), default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(1):
        inputs = load_controls(args.controls_directory, args.role, 0)
        report = {
            "scope": "Fixed-source, prescribed-flow thermal refinement diagnostic. Temperatures are retained without clipping. Values outside the published oil range are diagnostic outputs, with physical applicability unresolved.",
            "environment": environment(),
            "selection": inputs.selection,
            "identity": "The projected residual change equals the sum of diffusion, transport, stabilization, source-loading and background-loading changes. Component norms do not add and do not separately predict temperature changes.",
            "rows": diagnose(inputs, args.levels),
        }
        write_report(args.output, report)
    print(args.output.name)


if __name__ == "__main__":
    main()
