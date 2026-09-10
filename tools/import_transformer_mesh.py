"""Export a numeric-only transformer mesh and prescribed Stokes field.

The owner supplies the source directory explicitly. Only whitelisted thermal
parameters and geometry arrays enter the public bundle. Raw measurements,
OpenFOAM dictionaries, machine paths, and application logs are excluded.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from deflation_example.meshes import ThermalMesh, orient_cells
from deflation_example.reporting import write_fields, write_report


def export(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    mesh_path = source / "fattuesday_2d_physics_g1.npz"
    flow_path = source / "fattuesday_2d_stokes_field.npz"
    with np.load(mesh_path, allow_pickle=False) as m, np.load(flow_path, allow_pickle=False) as f:
        nodes = m["nodes"]
        cells = orient_cells(nodes, m["triangles"])
        materials = np.where(m["cell_baffle"], 2, np.where(m["cell_solid"], 1, 0))
        mesh = ThermalMesh(nodes, cells, materials, m["boundary_inlet"], axisymmetric=True)
        mesh.save(destination / "mesh.npz")
        locations = np.concatenate(
            (
                nodes[cells],
                ((nodes[cells[:, 0]] + nodes[cells[:, 1]]) / 2)[:, None],
                ((nodes[cells[:, 0]] + nodes[cells[:, 2]]) / 2)[:, None],
                ((nodes[cells[:, 1]] + nodes[cells[:, 2]]) / 2)[:, None],
            ),
            axis=1,
        )
        distance, indices = cKDTree(f["coords"]).query(locations.reshape(-1, 2))
        if distance.max() > 1e-10:
            raise ValueError("The frozen P2 flow does not match the exported mesh")
        velocity = np.column_stack((f["u"], f["v"]))[indices].reshape(len(cells), 6, 2)
        if np.max(np.abs(velocity[materials != 0])) > 1e-10:
            raise ValueError("The prescribed flow must vanish in solid and baffle cells")
        reference_K = float(m["T_ref_K"])
        reference_C = reference_K - 273.15
        oil_k = float(np.interp(reference_C, m["oil_T_C"], m["oil_kappa"]))
        oil_capacity = float(
            np.interp(reference_C, m["oil_T_C"], m["oil_rho"])
            * np.interp(reference_C, m["oil_T_C"], m["oil_cp"])
        )
        physical = {
            "reference_temperature_K": reference_K,
            "inlet_temperature_K": float(m["T_inlet_K"]),
            "inlet_velocity_m_s": float(m["u_inlet"]),
            "oil_conductivity_W_m_K": oil_k,
            "winding_conductivity_rz_W_m_K": m["kappa_solid_rz"].tolist(),
            "baffle_conductivity_W_m_K": float(m["kappa_baffle"]),
            "capacity_J_m3_K": [oil_capacity, float(m["rhoCv_solid"]), float(m["rhoCv_baffle"])],
            "temperature_scale_K": 20.0,
            "length_scale_m": 1.0,
            "time_scale_s": oil_capacity / oil_k,
            "control_units": "source density divided by oil_conductivity*temperature_scale/length_scale^2",
            "property_policy": "Material properties are frozen at reference_temperature_K.",
            "flow_policy": "Prescribed axisymmetric P2 Stokes velocity, frozen for all queries and time slabs.",
        }
        write_fields(
            destination / "inputs.npz",
            velocity_P2_m_s=velocity,
            source_W_m3=m["cell_q_density"],
            boundary_outlet=m["boundary_outlet"],
            boundary_wall=m["boundary_wall"],
        )
        # Boundary quadrature uses P2 velocities, preserving the supplied flow.
        fluxes = []
        tree = cKDTree(f["coords"])
        for key in ("boundary_inlet", "boundary_outlet"):
            points = nodes[m[key]]
            points = points[np.argsort(points[:, 0])]
            mid = (points[:-1] + points[1:]) / 2
            _, idx = tree.query(points)
            _, imid = tree.query(mid)
            fluxes.append(
                float(
                    np.sum(
                        2
                        * np.pi
                        * np.diff(points[:, 0])
                        / 6
                        * (
                            points[:-1, 0] * f["v"][idx[:-1]]
                            + 4 * mid[:, 0] * f["v"][imid]
                            + points[1:, 0] * f["v"][idx[1:]]
                        )
                    )
                )
            )
        imbalance = abs(fluxes[1] - fluxes[0]) / max(abs(fluxes[0]), np.finfo(float).tiny)
        if imbalance > 1e-5:
            raise ValueError("Frozen-flow inlet/outlet imbalance exceeds 1e-5")
        write_report(
            destination / "parameters.json",
            {
                "schema": "mesh-cht-input-v1",
                "geometry": "transformer_2d",
                "source_files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (mesh_path, flow_path)
                },
                "source_revision": "35161d0afc0c0115d7ef03d8474d451885d3b75d",
                "coordinates": "axisymmetric (r,z), metres",
                "materials": {"0": "oil", "1": "winding", "2": "baffle"},
                "physical": physical,
                "flow_verification": {
                    "inlet_outlet_relative_imbalance": imbalance,
                    "inlet_outlet_flux_m3_s": fluxes,
                    "coordinate_match_max_m": float(distance.max()),
                },
                "scope": "Self-built geometry with prescribed flow; thermal optimization uses a separate P1 discretization.",
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
