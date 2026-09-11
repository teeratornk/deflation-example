"""Reproducible operating-point pilots for coupled transformer optimization."""

import hashlib
import json
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from .assess_transformer import transformer_boundaries
from .axisymmetric_flow import AxisymmetricFlow, FlowResult
from .mesh_refinement import refine, refine_quadratic_flow
from .meshes import ThermalMesh
from .oil_properties import momentum_reference
from .reporting import environment, write_fields, write_report
from .validation import integer, positive_real


def transformer_inputs(level=0):
    directory = Path(__file__).parent / "data" / "transformer_2d"
    parameters = json.loads((directory / "parameters.json").read_text())
    mesh = ThermalMesh.load(directory / "mesh.npz")
    with np.load(directory / "inputs.npz", allow_pickle=False) as data:
        source = data["source_W_m3"].copy()
    for _ in range(integer(level, "Refinement level")):
        mesh, _, parent = refine(mesh)
        source = source[parent]
    p = parameters["physical"]
    k = np.tile(np.eye(2), (len(mesh.cells), 1, 1))
    k[mesh.materials == 1] = (
        np.diag(p["winding_conductivity_rz_W_m_K"]) / p["oil_conductivity_W_m_K"]
    )
    k[mesh.materials == 2] *= p["baffle_conductivity_W_m_K"] / p["oil_conductivity_W_m_K"]
    c = np.array(p["capacity_J_m3_K"])[mesh.materials] / p["capacity_J_m3_K"][0]
    q = source * p["length_scale_m"] ** 2 / (p["oil_conductivity_W_m_K"] * p["temperature_scale_K"])
    hashes = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(directory.iterdir())
        if f.is_file()
    }
    return mesh, parameters, k, c, q, hashes


def prolong_flow(coarse_flow, fine_flow, prolongation, parent, result):
    """Transfer P2 velocity and P1 pressure to one uniformly refined mesh."""
    local = refine_quadratic_flow(
        coarse_flow.mesh, fine_flow.mesh, parent, coarse_flow.thermal_velocity(result.velocity)
    )[fine_flow.fluid_cells]
    velocity = np.zeros((fine_flow.nv, 2))
    count = np.zeros(fine_flow.nv)
    np.add.at(velocity, fine_flow.p2.ravel(), local.reshape(-1, 2))
    np.add.at(count, fine_flow.p2.ravel(), 1)
    velocity /= count[:, None]
    pressure = np.zeros(len(coarse_flow.mesh.nodes))
    pressure[coarse_flow.vertices] = result.pressure
    return FlowResult(
        velocity, np.asarray(prolongation @ pressure)[fine_flow.vertices], "initial", []
    )


def run(config):
    cfg = OmegaConf.to_container(config, resolve=True)
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    threads = integer(cfg["threads"], "Thread count", 1)
    factor = positive_real(cfg["inlet_factor"], "Inlet speed factor")
    if factor not in (1.0, 0.5, 0.25, 0.125):
        raise ValueError("The declared inlet pilot factors are 1, 0.5, 0.25 and 0.125")
    with threadpool_limits(threads):
        start = time.perf_counter()
        mesh, parameters, _, _, _, hashes = transformer_inputs(cfg["level"])
        properties = momentum_reference()
        speed = parameters["physical"]["inlet_velocity_m_s"] * factor
        grad_div_scale = float(cfg["grad_div_scale"])
        if not np.isfinite(grad_div_scale) or grad_div_scale < 0:
            raise ValueError("Grad-div scale must be finite and nonnegative")
        grad_div = grad_div_scale * speed * np.ptp(mesh.nodes[:, 0])
        flow = AxisymmetricFlow(
            mesh,
            properties["kinematic_viscosity_m2_s"],
            convection_form=cfg["convection_form"],
            grad_div=grad_div,
        )
        fixed, values = transformer_boundaries(flow, speed)
        force = np.zeros_like(flow.quadrature_points)
        metadata = {
            "schema": "coupled-optimization-baseline-v1",
            "environment": environment(),
            "configuration": {
                k: v for k, v in cfg.items() if k not in {"output", "seed_directory"}
            },
            "input_sha256": hashes,
            "properties": properties,
            "inlet_velocity_m_s": speed,
            "grad_div_coefficient_m2_s": grad_div,
            "thermal_nodes": len(mesh.nodes),
            "flow_unknowns": flow.size,
            "scope": "Isothermal nonlinear flow initialization; no optimization or speedup result.",
        }
        write_report(output / "record.json", {**metadata, "status": "running"})

        def checkpoint(stages, candidate):
            write_report(
                output / "progress.json",
                {
                    "stages": stages,
                    "seconds": time.perf_counter() - start,
                },
            )
            if stages[-1]["status"] == "converged":
                write_fields(
                    output / "baseline-checkpoint.npz",
                    velocity=candidate.velocity,
                    pressure=candidate.pressure,
                )

        if cfg.get("seed_directory"):
            directory = Path(cfg["seed_directory"])
            seed = json.loads((directory / "record.json").read_text())
            field = directory / "baseline.npz"
            if (
                seed["status"] != "converged"
                or seed["properties"] != properties
                or seed["input_sha256"] != hashes
                or seed["configuration"]["level"] + 1 != cfg["level"]
                or seed["inlet_velocity_m_s"] != speed
                or seed["configuration"]["convection_form"] != cfg["convection_form"]
                or seed.get("grad_div_coefficient_m2_s", 0.0) != grad_div
                or hashlib.sha256(field.read_bytes()).hexdigest() != seed["baseline_sha256"]
            ):
                raise ValueError("The prolongation seed must be a verified matching coarse model")
            coarse_mesh, *_ = transformer_inputs(cfg["level"] - 1)
            refined, interpolation, parent = refine(coarse_mesh)
            if not (
                np.array_equal(refined.nodes, mesh.nodes)
                and np.array_equal(refined.cells, mesh.cells)
            ):
                raise ValueError("The refinement does not match the declared mesh")
            coarse_flow = AxisymmetricFlow(
                coarse_mesh,
                flow.viscosity,
                convection_form=flow.convection_form,
                grad_div=flow.grad_div,
            )
            with np.load(field, allow_pickle=False) as data:
                coarse_result = FlowResult(
                    data["velocity"].copy(), data["pressure"].copy(), "converged", []
                )
            coarse_fixed, coarse_values = transformer_boundaries(coarse_flow, speed)
            seed_checks = coarse_flow.verify(
                coarse_result,
                np.zeros_like(coarse_flow.quadrature_points),
                coarse_fixed,
                coarse_values,
            )
            if (
                not np.isfinite(list(seed_checks.values())).all()
                or max(seed_checks.values()) > 1e-8
            ):
                raise ValueError("The coarse initialization fails the original flow equations")
            initial = prolong_flow(coarse_flow, flow, interpolation, parent, coarse_result)
            baseline = flow.solve(
                force,
                fixed,
                values,
                initial=initial,
                method="newton",
                tolerance=cfg["flow_tolerance"],
                max_iterations=cfg["flow_cap"],
            )
            metadata["initialization"] = (
                "P2/P1 prolongation from verified coarse physical-viscosity flow"
            )
            metadata["seed_baseline_sha256"] = seed["baseline_sha256"]
        else:
            baseline = flow.steady_initialization(
                force,
                fixed,
                values,
                tolerance=cfg["flow_tolerance"],
                max_iterations=cfg["flow_cap"],
                callback=checkpoint,
            )
        residuals = flow.verify(baseline, force, fixed, values)
        flux = flow.boundary_flux(baseline.velocity)
        mass = float(abs(flux.sum()) / max(np.abs(flux).sum() / 2, 1e-30))
        write_fields(
            output / "baseline.npz", velocity=baseline.velocity, pressure=baseline.pressure
        )
        write_report(
            output / "record.json",
            {
                **metadata,
                "status": baseline.status,
                "stages": baseline.history,
                "residuals": residuals,
                "relative_mass_imbalance": mass,
                "peak_velocity_m_s": float(np.linalg.norm(baseline.velocity, axis=1).max()),
                "kinetic_diagnostics": flow.kinetic_diagnostics(baseline.velocity),
                "seconds": time.perf_counter() - start,
                "baseline_sha256": hashlib.sha256(
                    (output / "baseline.npz").read_bytes()
                ).hexdigest(),
            },
        )
    print(json.dumps({"inlet_factor": factor, "status": baseline.status, "mass": mass}), flush=True)


@hydra.main(version_base="1.3", config_path="conf", config_name="coupled_pilot")
def main(config):
    run(config)


if __name__ == "__main__":
    main()
