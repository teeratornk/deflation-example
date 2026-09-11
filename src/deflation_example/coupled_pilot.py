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
from .axisymmetric_flow import AxisymmetricFlow
from .mesh_refinement import refine
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
            "configuration": {k: v for k, v in cfg.items() if k != "output"},
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
