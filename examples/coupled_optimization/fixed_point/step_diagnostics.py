"""Local conservation and transport diagnostics on immutable trajectory steps.

These diagnostics leave the equations, controls and trajectory records unchanged.
Cellwise divergence and field roughness are indicators, not error estimates or
proofs of spurious oscillations. Optional Newton checks use the original equations.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from deflation_example.axisymmetric_flow import FlowResult
from deflation_example.coupled_newton_diagnostic import inspect_step
from deflation_example.coupled_newton_replay import step_equations
from deflation_example.coupled_resolution import forward_model
from deflation_example.coupled_saved import file_digest
from deflation_example.fixed_point_study import prepare
from deflation_example.meshes import simplex_geometry
from deflation_example.reporting import environment, write_arrays, write_report


def summary(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Diagnostics require nonempty finite values")
    return dict(
        zip(
            ("minimum", "median", "p95", "maximum"),
            np.quantile(values, [0, 0.5, 0.95, 1]).tolist(),
            strict=True,
        )
    )


def cell_mass_diagnostics(flow, velocity):
    """Cylindrical divergence integrated with the existing flow quadrature."""
    samples = flow.sampled_velocity(velocity)
    gradient = np.einsum("eia,eqib->eqab", velocity[flow.p2], flow.grad)
    divergence = gradient[:, :, 0, 0] + gradient[:, :, 1, 1] + samples[:, :, 0] / flow.radius
    measure = flow.measure.sum(axis=1)
    signed = np.sum(flow.measure * divergence, axis=1)
    rms = np.sqrt(np.sum(flow.measure * divergence**2, axis=1) / measure)
    boundary = flow.boundary_flux(velocity)
    throughput = max(np.abs(boundary).sum() / 2, 1e-30)
    return {
        "global_boundary_relative_imbalance": float(abs(boundary.sum()) / throughput),
        "sum_absolute_cell_flux_over_boundary_throughput": float(np.abs(signed).sum() / throughput),
        "maximum_absolute_cell_net_flux_m3_s": float(np.abs(signed).max()),
        "cell_divergence_rms_s_inverse": summary(rms),
        "divergence_theorem_absolute_defect_m3_s": float(abs(signed.sum() - boundary.sum())),
        "kinetic": flow.kinetic_diagnostics(velocity),
    }, {"cell_net_flux_m3_s": signed, "cell_divergence_rms_s_inverse": rms}


def thermal_diagnostics(problem, state, flow):
    mesh = problem.mesh
    vertices = mesh.nodes[mesh.cells]
    h = np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=3).max(axis=(1, 2))
    sampled_speed = np.linalg.norm(problem.flow.sampled_velocity(flow.velocity), axis=2).max(axis=1)
    speed = np.zeros(len(mesh.cells))
    speed[problem.flow.fluid_cells] = sampled_speed * problem.velocity_scale
    kmin = np.linalg.eigvalsh(problem.conductivity)[:, 0]
    peclet = problem.capacity * speed * h / (2 * kmin)
    grad, _ = simplex_geometry(mesh)
    temperature = problem.temperature_offset + problem.temperature_scale * state
    thermal_gradient = np.einsum("ei,eid->ed", temperature[mesh.cells], grad)
    pressure_gradient = np.einsum(
        "ei,eid->ed", flow.pressure[problem.flow.p1], grad[problem.flow.fluid_cells]
    )
    assembly = problem.assemble(flow.velocity)
    stiffness = assembly.stiffness[problem.free][:, problem.free].tocoo()
    off_diagonal = stiffness.data[stiffness.row != stiffness.col]
    tolerance = 1e-12 * max(float(np.abs(stiffness.data).max()), 1e-30)
    return {
        "temperature_K": summary(temperature),
        "temperature_cell_gradient_K_per_m": summary(np.linalg.norm(thermal_gradient, axis=1)),
        "fluid_cell_peclet_sampled_maximum_speed": summary(peclet[problem.flow.fluid_cells]),
        "positive_off_diagonal_fraction": float(np.mean(off_diagonal > tolerance)),
        "positive_off_diagonal_threshold": tolerance,
        "pressure_m2_s2": summary(flow.pressure),
        "pressure_cell_gradient_m_s2": summary(np.linalg.norm(pressure_gradient, axis=1)),
        "scope": "Positive off-diagonal entries indicate failure of the usual Z-matrix sign condition, not proof of oscillations. Signed heat sources allow physical interior extrema.",
    }, {"temperature_K": temperature, "cell_peclet": peclet}


def checked_field(directory, row, problem):
    name = row["fields"]
    if Path(name).name != name:
        raise ValueError("Field archive must be a local basename")
    path = directory / name
    if path.resolve().parent != directory.resolve() or file_digest(path) != row["field_sha256"]:
        raise ValueError("Field identity differs")
    with np.load(path, allow_pickle=False) as arrays:
        y, v, p = (arrays[key].copy() for key in ("state", "velocity", "pressure"))
    if (
        y.shape != (problem.spatial_size,)
        or v.shape != (problem.flow.nv, 2)
        or p.shape != (problem.flow.np,)
    ):
        raise ValueError("Field dimensions differ")
    if not all(np.isfinite(a).all() for a in (y, v, p)):
        raise ValueError("Nonfinite saved field")
    return problem.full_temperature(y), FlowResult(v, p, "saved", [])


def load_case(args):
    record = json.loads((args.trajectory / "record.json").read_text())
    if record["status"] not in {"converged", "numerical_failure"}:
        raise ValueError("Inspect a terminal immutable trajectory")
    if record["time_scheme"] != "backward_euler":
        raise ValueError("This diagnostic currently supports backward Euler only")
    args.policy, args.family = record["policy"], "forward"
    args.repetition, args.target_position = record["repetition"], None
    problem, controls, metadata = prepare(args, record["subdivision"])
    for key in (
        "optimization_field_sha256",
        "baseline_sha256",
        "fine_baseline_sha256",
        "slabs",
        "spatial_state_dofs",
        "configuration",
    ):
        if record[key] != metadata[key]:
            raise ValueError("Trajectory identity differs: " + key)
    n = len(record["steps"]) - 1 if args.slab is None else args.slab
    if not 0 <= n < len(record["steps"]):
        raise ValueError("Select a saved step")
    for i, row in enumerate(record["steps"][: n + 1]):
        if row["slab_zero_based"] != i or not np.isclose(
            row["time_s"], np.sum(problem.physical_steps[: i + 1]), rtol=0, atol=1e-9
        ):
            raise ValueError("Trajectory time ordering differs")
        if i < n and not row["verified"]:
            raise ValueError("The preceding trajectory must be verified")
    state, flow = checked_field(args.trajectory, record["steps"][n], problem)
    previous, previous_flow = (
        checked_field(args.trajectory, record["steps"][n - 1], problem)
        if n
        else (problem.full_temperature(problem.initial), problem.initial_flow)
    )
    source = np.zeros(len(problem.mesh.nodes))
    source[problem.free] = controls[n]
    return problem, source, previous, previous_flow, state, flow, n, record


def plot_fields(problem, flow, arrays, output, physical_time):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    mesh = problem.mesh
    tri = mtri.Triangulation(*mesh.nodes.T, mesh.cells)
    fluid = mtri.Triangulation(*mesh.nodes[problem.flow.vertices].T, problem.flow.p1)
    figure, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    specs = [
        (tri, arrays["temperature_K"], "Temperature (K)", False),
        (fluid, flow.pressure, "Kinematic pressure (m²/s²)", False),
        (fluid, arrays["cell_divergence_rms_s_inverse"], "Cell divergence RMS (s⁻¹)", True),
        (tri, arrays["cell_peclet"], "Cell Péclet number", True),
    ]
    for ax, (triangles, values, title, cellwise) in zip(axes.flat, specs, strict=True):
        artist = (
            ax.tripcolor(triangles, facecolors=values, shading="flat")
            if cellwise
            else ax.tripcolor(triangles, values, shading="gouraud")
        )
        figure.colorbar(artist, ax=ax)
        ax.set(title=title, xlabel="r (m)", ylabel="z (m)")
    figure.suptitle(
        f"Saved fields at t = {physical_time:g} s; radial and axial display scales differ"
    )
    figure.savefig(output / "fields.png", dpi=180)
    figure.savefig(output / "fields.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "fine-baseline", "optimization", "trajectory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--slab", type=int)
    parser.add_argument("--newton-check", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    result = {
        "status": "running",
        "environment": environment(),
        "diagnostic_sha256": file_digest(Path(__file__)),
        "scope": "Read-only saved-step diagnosis; no numerical repair or complete-trajectory result.",
    }
    write_report(args.output / "record.json", result)
    try:
        with threadpool_limits(args.threads):
            problem, source, previous, previous_flow, state, flow, n, record = load_case(args)
            result.update(
                trajectory_record_sha256=file_digest(args.trajectory / "record.json"),
                field_sha256=record["steps"][n]["field_sha256"],
                slab_zero_based=n,
                time_s=record["steps"][n]["time_s"],
                saved_status=record["steps"][n]["status"],
            )
            _, checks = step_equations(
                problem, forward_model(problem), state, flow, source, previous, previous_flow, n
            )
            mass, mass_arrays = cell_mass_diagnostics(problem.flow, flow.velocity)
            thermal, thermal_arrays = thermal_diagnostics(problem, state, flow)
            result.update(equations=checks, mass=mass, thermal=thermal)
            arrays = {**mass_arrays, **thermal_arrays}
            write_arrays(args.output / "indicators.npz", **arrays)
            result["indicator_sha256"] = file_digest(args.output / "indicators.npz")
            write_report(args.output / "record.json", result)
            plot_fields(problem, flow, arrays, args.output, result["time_s"])
            if args.newton_check:
                result["newton_check"] = inspect_step(
                    problem, source, previous, previous_flow, state, flow, n
                )
            result["status"] = "complete"
    except Exception as error:
        result.update(status="execution_failure", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result["seconds"] = time.perf_counter() - began
        write_report(args.output / "record.json", result)


if __name__ == "__main__":
    main()
