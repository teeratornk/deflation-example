"""Generate the documented bore-in-block geometry with a conforming tetrahedral mesh."""

import argparse
from pathlib import Path

import numpy as np

from .meshes import ThermalMesh, orient_cells
from .reporting import write_report


def generate(output, size=0.12):
    import gmsh

    if not np.isfinite(size) or size <= 0:
        raise ValueError("Mesh size must be positive and finite")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("engine_bore")
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 1, 1, 1)
        bore = occ.addCylinder(0.5, 0.5, 0.12, 0, 0, 0.66, 0.3)
        fragments, mapping = occ.fragment([(3, box)], [(3, bore)])
        occ.synchronize()
        gas = {tag for dim, tag in mapping[1] if dim == 3}
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = coords.reshape(-1, 3)
        lookup = {int(t): i for i, t in enumerate(tags)}
        cells, materials = [], []
        for dim, tag in fragments:
            if dim != 3:
                continue
            types, _, element_nodes = gmsh.model.mesh.getElements(3, tag)
            for kind, conn in zip(types, element_nodes, strict=True):
                if kind != 4:
                    raise ValueError("Only linear tetrahedra are supported")
                block = np.array([lookup[int(t)] for t in conn]).reshape(-1, 4)
                cells.append(block)
                materials.extend([0 if tag in gas else 1] * len(block))
        boundary = np.flatnonzero(
            np.any(np.isclose(nodes, 0, atol=1e-12) | np.isclose(nodes, 1, atol=1e-12), axis=1)
        )
        mesh = ThermalMesh(
            nodes, orient_cells(nodes, np.vstack(cells)), np.asarray(materials), boundary
        )
        mesh.save(output / "mesh.npz")
        write_report(
            output / "parameters.json",
            {
                "schema": "mesh-cht-input-v1",
                "geometry": "engine_3d",
                "generator": "deflation_example.generate_engine_mesh",
                "gmsh": gmsh.__version__,
                "mesh_size": size,
                "mesher_threads": 1,
                "coordinates": "(x1,x2,x3) divided by block length",
                "geometry_parameters": {
                    "block": [1.0, 1.0, 1.0],
                    "bore_center": [0.5, 0.5],
                    "bore_radius": 0.3,
                    "floor": 0.12,
                    "deck": 0.78,
                },
                "materials": {"0": "gas", "1": "metal"},
                "physical": {
                    "conductivity_ratio": 1000.0,
                    "metal_conductivity_W_m_K": 55.0,
                    "inward_interface_flux_W_m2": 500000.0,
                    "length_scale_m": 0.1,
                    "coolant_temperature_K": 360.0,
                    "temperature_scale_K": 500000.0 * 0.1 / 55.0,
                    "dimensionless_interface_flux": 1000.0,
                    "dimensionless_capacity": [1.0, 1.0],
                    "capacity_policy": "Unit capacity in both materials, matching the documented dimensionless energy equation.",
                    "flow_policy": "Prescribed smooth azimuthal circulation in the bore; zero velocity in metal.",
                    "velocity_max": 50.0,
                    "time_units": "fluid diffusion time L^2/thermal_diffusivity",
                    "control_units": "dimensionless distributed thermal source",
                },
                "source": {
                    "manuscript_section": "03c_engine.tex",
                    "repository": "https://github.com/teeratornk/gnn_solver_manuscript",
                    "scope": "Geometry, conductivity, interface flux and cooling boundary follow the documented case. The optimization prescribes circulation independently of temperature.",
                },
            },
        )
    finally:
        gmsh.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=float, default=0.12)
    args = parser.parse_args()
    generate(args.output, args.size)


if __name__ == "__main__":
    main()
