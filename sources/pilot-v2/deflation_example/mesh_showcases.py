"""Self-contained mesh presets with explicit nondimensionalization."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .meshes import ThermalMesh, assemble_thermal, add_facet_load
from .mesh_refinement import refine, refine_quadratic_flow
from .validation import integer


@dataclass
class MeshShowcase:
    assembly: object
    coarse_assembly: object
    prolongation: sparse.csr_matrix
    parameters: dict
    preparation: dict


def engine_velocity(points, speed=50.):
    """Smooth divergence-free azimuthal flow, zero on the analytic bore boundary."""
    x, y, z = points.T
    dx, dy = x - .5, y - .5
    radius2 = dx**2 + dy**2
    axial = np.maximum((z - .12) * (.78 - z), 0)**2
    radial = np.maximum(.3**2 - radius2, 0)**2
    # max r*(R^2-r^2)^2 occurs at r=R/sqrt(5).
    rmax = .3 / np.sqrt(5)
    norm = rmax * (.3**2 - rmax**2)**2 * (.66**2/4)**2
    factor = speed * axial * radial / norm
    return np.column_stack((-dy*factor, dx*factor, np.zeros_like(x)))


def _assembly(mesh, parameters, velocity=None, source=None):
    nc, dim = len(mesh.cells), mesh.dimension
    p = parameters["physical"]
    k = np.tile(np.eye(dim), (nc, 1, 1))
    if parameters["geometry"] == "transformer_2d":
        conductivity_scale = p["oil_conductivity_W_m_K"]
        k[mesh.materials == 1] = np.diag(p["winding_conductivity_rz_W_m_K"]) / conductivity_scale
        k[mesh.materials == 2] *= p["baffle_conductivity_W_m_K"] / conductivity_scale
        capacity = np.array(p["capacity_J_m3_K"])[mesh.materials] / p["capacity_J_m3_K"][0]
        velocity = velocity * p["time_scale_s"] / p["length_scale_m"]
        source = source * p["length_scale_m"]**2 / (conductivity_scale * p["temperature_scale_K"])
        return assemble_thermal(mesh, k, capacity, velocity, source, streamline=True)
    k[mesh.materials == 1] *= p["conductivity_ratio"]
    center = mesh.nodes[mesh.cells].mean(axis=1)
    velocity = engine_velocity(center, p["velocity_max"])
    velocity[mesh.materials != 0] = 0
    capacity = np.array(p["dimensionless_capacity"])[mesh.materials]
    result = assemble_thermal(mesh, k, capacity, velocity, streamline=True)
    # An interface source contributes once to the single conforming temperature equation.
    faces = np.concatenate([mesh.cells[:, face] for face in ((0, 1, 2), (0, 1, 3),
                                                            (0, 2, 3), (1, 2, 3))])
    material = np.tile(mesh.materials, 4)
    unique, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                        return_inverse=True, return_counts=True)
    gas_incidence = np.bincount(inverse, weights=material == 0, minlength=len(unique))
    interface = unique[(counts == 2) & (gas_incidence == 1)]
    if not len(interface):
        raise ValueError("The engine mesh must contain a gas-metal interface")
    add_facet_load(result, interface, p["dimensionless_interface_flux"])
    return result


def build_showcase(geometry="engine_3d", level=0, data_directory=None):
    import time

    start = time.perf_counter()
    if geometry not in {"engine_3d", "transformer_2d"}:
        raise ValueError("Choose transformer_2d or engine_3d")
    level = integer(level, "Mesh refinement level")
    directory = (Path(__file__).parent / "data" / geometry
                 if data_directory is None else Path(data_directory))
    parameters = json.loads((directory / "parameters.json").read_text())
    if parameters["geometry"] != geometry or parameters["schema"] != "mesh-cht-input-v1":
        raise ValueError("The mesh input manifest does not match the requested geometry")
    mesh = ThermalMesh.load(directory / "mesh.npz")
    base = mesh
    velocity, source = None, None
    if geometry == "transformer_2d":
        with np.load(directory / "inputs.npz", allow_pickle=False) as inputs:
            velocity = inputs["velocity_P2_m_s"]
            source = inputs["source_W_m3"]
    coarse = _assembly(mesh, parameters, velocity, source)
    P = sparse.eye(len(mesh.nodes), format="csr")
    for _ in range(level):
        fine, interpolation, parent = refine(mesh)
        if velocity is not None:
            velocity = refine_quadratic_flow(mesh, fine, parent, velocity)
            source = source[parent]
        mesh = fine
        P = interpolation @ P
    current = coarse if not level else _assembly(mesh, parameters, velocity, source)
    return MeshShowcase(current, coarse, P[mesh.free][:, base.free], parameters,
                        {"mesh_and_assembly_seconds": time.perf_counter() - start,
                         "refinement": "nested simplex subdivision with inherited materials",
                         "level": level, "nodes": len(mesh.nodes), "cells": len(mesh.cells),
                         "state_degrees_of_freedom": len(mesh.free),
                         "weighted_volume": float(current.mass.sum()),
                         "coarse_weighted_volume": float(coarse.mass.sum())})


def desired_temperature(problem, query, count):
    """Distinct moving targets in physical space, with pulses along physical time."""
    integer(query, "Query index")
    integer(count, "Query count", 1)
    if query >= count:
        raise ValueError("Query index must be smaller than query count")
    x = problem.assembly.mesh.nodes[problem.free]
    lower, extent = problem.assembly.mesh.nodes.min(axis=0), np.ptp(problem.assembly.mesh.nodes, axis=0)
    x = (x - lower) / extent
    theta = (query + .25) / count
    times = np.cumsum(problem.steps) if len(problem.steps) else np.array([0.])
    fields = []
    for t in times:
        phase = 0 if not len(problem.steps) else t / times[-1]
        center = np.array([.5 + .24*np.cos(2*np.pi*theta + phase),
                           .5 + .27*np.sin(2*np.pi*theta + .7*phase)])
        if x.shape[1] == 3:
            center = np.r_[center, .28 + .42*theta + .06*np.sin(2*np.pi*phase)]
        width = .15 if x.shape[1] == 2 else .24
        gaussian = np.exp(-np.sum(((x - center)/width)**2, axis=1))
        second = np.exp(-np.sum(((x - (.75 - .45*center))/(width*.8))**2, axis=1))
        pulse = 1 if not len(problem.steps) else .25 + .75*np.sin(np.pi*phase)**2
        fields.append(.1 + pulse*((1 + .15*theta)*gaussian + .35*second))
    return np.concatenate(fields)
