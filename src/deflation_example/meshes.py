"""Validated simplex meshes and conforming P1 thermal assembly.

Axisymmetric meshes use coordinates (r, z) and the measure 2*pi*r dr dz.
Temperature and distributed control use the same positive lumped mass weights.
"""

from dataclasses import dataclass
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
from scipy import sparse

from .validation import real_array


@dataclass(frozen=True)
class ThermalMesh:
    nodes: np.ndarray
    cells: np.ndarray
    materials: np.ndarray
    dirichlet: np.ndarray
    axisymmetric: bool = False

    def __post_init__(self):
        nodes = real_array(self.nodes, "Mesh coordinates").copy()
        if nodes.ndim != 2 or nodes.shape[1] not in (2, 3) or not np.isfinite(nodes).all():
            raise ValueError("Mesh coordinates must be finite with dimension 2 or 3")
        d = nodes.shape[1]
        if self.axisymmetric and (d != 2 or np.any(nodes[:, 0] < 0)):
            raise ValueError("Axisymmetric meshes require nonnegative (r,z) coordinates")
        cells = np.asarray(self.cells)
        if cells.ndim != 2 or cells.shape[1] != d + 1 or cells.dtype.kind not in "iu":
            raise ValueError("Cells must contain integer simplex connectivity")
        cells = cells.astype(np.int64, copy=True)
        if not len(cells) or cells.min() < 0 or cells.max() >= len(nodes):
            raise ValueError("Cell indices must reference existing nodes")
        if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
            raise ValueError("A cell cannot repeat a node")
        if len(np.unique(np.sort(cells, axis=1), axis=0)) != len(cells):
            raise ValueError("Duplicate cells are invalid")
        if len(np.unique(nodes, axis=0)) != len(nodes):
            raise ValueError("Coincident nodes must share an index")
        if len(np.unique(cells)) != len(nodes):
            raise ValueError("Every mesh node must belong to a cell")
        material = np.asarray(self.materials)
        if material.shape != (len(cells),) or material.dtype.kind not in "iu":
            raise ValueError("Material labels must be one integer per cell")
        if np.any(material < 0):
            raise ValueError("Material labels must be nonnegative")
        pins = np.asarray(self.dirichlet)
        if pins.ndim != 1 or pins.dtype.kind not in "iu":
            raise ValueError("Dirichlet indices must be a one-dimensional integer array")
        if not len(pins) or pins.min() < 0 or pins.max() >= len(nodes):
            raise ValueError("At least one valid Dirichlet node is required")
        jac = nodes[cells[:, 1:]] - nodes[cells[:, :1]]
        determinant = np.linalg.det(jac)
        scale = np.max(np.linalg.norm(jac, axis=2), axis=1) ** d
        if np.any(determinant <= 64 * np.finfo(float).eps * scale):
            raise ValueError("Cells must have positive, nondegenerate orientation")
        faces = np.concatenate([cells[:, f] for f in combinations(range(d + 1), d)])
        _, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("A conforming mesh facet has at most two incident cells")
        graph = sparse.coo_matrix(
            (np.ones(len(cells) * d), (np.repeat(cells[:, 0], d), cells[:, 1:].ravel())),
            shape=(len(nodes), len(nodes)),
        ).tocsr()
        _, labels = sparse.csgraph.connected_components(graph, directed=False)
        if len(np.unique(labels[pins])) != len(np.unique(labels)):
            raise ValueError("Every connected component needs a Dirichlet boundary")
        for key, array in (
            ("nodes", nodes),
            ("cells", cells),
            ("materials", material.astype(np.int64, copy=True)),
            ("dirichlet", np.unique(pins).astype(np.int64)),
        ):
            array.flags.writeable = False
            object.__setattr__(self, key, array)

    @property
    def dimension(self):
        return self.nodes.shape[1]

    @property
    def free(self):
        return np.setdiff1d(np.arange(len(self.nodes)), self.dirichlet, assume_unique=True)

    def facets(self):
        """Unique facets, incident-cell counts, and one owning cell per facet."""
        local = list(combinations(range(self.dimension + 1), self.dimension))
        faces = np.concatenate([self.cells[:, f] for f in local])
        facets, first, counts = np.unique(
            np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
        )
        return facets, counts, first % len(self.cells)

    def save(self, path):
        """Create a portable numeric-only mesh; existing paths are protected."""
        with Path(path).open("xb") as stream:
            np.savez_compressed(
                stream,
                nodes=self.nodes,
                cells=self.cells,
                materials=self.materials,
                dirichlet=self.dirichlet,
                axisymmetric=np.array(self.axisymmetric),
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                *(data[k] for k in ("nodes", "cells", "materials", "dirichlet")),
                axisymmetric=bool(data["axisymmetric"].item()),
            )


def orient_cells(nodes, cells):
    """Normalize imported orientation explicitly before constructing a mesh."""
    cells = np.array(cells, dtype=np.int64, copy=True)
    negative = np.linalg.det(nodes[cells[:, 1:]] - nodes[cells[:, :1]]) < 0
    cells[negative, :2] = cells[negative, 1::-1]
    return cells


@dataclass
class ThermalAssembly:
    mesh: ThermalMesh
    mass: np.ndarray
    capacity: np.ndarray
    diffusion: sparse.csr_matrix
    transport: sparse.csr_matrix
    stabilization: sparse.csr_matrix
    load: np.ndarray

    @property
    def stiffness(self):
        return self.diffusion + self.transport + self.stabilization


def simplex_geometry(mesh):
    vertices = mesh.nodes[mesh.cells]
    jac = vertices[:, 1:] - vertices[:, :1]
    volume = np.linalg.det(jac) / factorial(mesh.dimension)
    gradients = np.empty((len(mesh.cells), mesh.dimension + 1, mesh.dimension))
    gradients[:, 1:] = np.linalg.inv(jac).transpose(0, 2, 1)
    gradients[:, 0] = -gradients[:, 1:].sum(axis=1)
    if mesh.axisymmetric:
        radii = vertices[:, :, 0]
        # Exact integral of r*N_i. Positive on every nondegenerate cell.
        local_mass = 2 * np.pi * volume[:, None] * (radii.sum(axis=1)[:, None] + radii) / 12
    else:
        local_mass = np.repeat((volume / (mesh.dimension + 1))[:, None], mesh.dimension + 1, axis=1)
    return gradients, local_mass


def assemble_thermal(mesh, conductivity, capacity, velocity=None, source=None, streamline=False):
    """Assemble diffusion, nonconservative transport, and optional streamline diffusion.

    Conductivity is a symmetric positive-definite tensor per cell. Capacity and
    velocity are frozen per cell. Streamline diffusion adds a declared symmetric
    artificial-diffusion term to the discrete state equation; no SUPG right-hand
    side is implied. The exact transpose of the resulting equation is used.
    """
    nc, d = len(mesh.cells), mesh.dimension
    k = real_array(conductivity, "Conductivity")
    c = real_array(capacity, "Capacity")
    v = np.zeros((nc, d)) if velocity is None else real_array(velocity, "Velocity")
    q = np.zeros(nc) if source is None else real_array(source, "Source")
    if k.shape != (nc, d, d) or not np.isfinite(k).all():
        raise ValueError("Conductivity must have shape (cells,dimension,dimension)")
    if not np.allclose(k, k.transpose(0, 2, 1), rtol=1e-13, atol=0):
        raise ValueError("Conductivity tensors must be symmetric")
    eigenvalues = np.linalg.eigvalsh(k)
    if np.any(eigenvalues <= 0):
        raise ValueError("Conductivity tensors must be positive definite")
    if c.shape != (nc,) or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("Capacity must be finite and positive in every cell")
    quadratic_flow = d == 2 and v.shape == (nc, 6, 2)
    if (v.shape != (nc, d) and not quadratic_flow) or not np.isfinite(v).all():
        raise ValueError("Velocity must be finite with one vector per cell")
    if q.shape != (nc,) or not np.isfinite(q).all():
        raise ValueError("Source must be finite with one value per cell")
    grad, lump = simplex_geometry(mesh)
    measure = lump.sum(axis=1)
    diffusion = measure[:, None, None] * np.einsum("eid,edf,ejf->eij", grad, k, grad)
    if quadratic_flow:
        # Degree-five triangle quadrature integrates r*N_i*v_P2 exactly.
        bary, quadrature_weights = triangle_quadrature()
        shape = np.column_stack(
            (
                bary * (2 * bary - 1),
                4 * bary[:, 0] * bary[:, 1],
                4 * bary[:, 0] * bary[:, 2],
                4 * bary[:, 1] * bary[:, 2],
            )
        )
        samples = np.einsum("qa,ead->eqd", shape, v)
        volume = np.linalg.det(mesh.nodes[mesh.cells[:, 1:]] - mesh.nodes[mesh.cells[:, :1]]) / 2
        qmeasure = volume[:, None] * quadrature_weights
        if mesh.axisymmetric:
            qmeasure = qmeasure * 2 * np.pi * (mesh.nodes[mesh.cells, 0] @ bary.T)
        moments = np.einsum("eq,qi,eqd->eid", qmeasure, bary, samples)
        transport = c[:, None, None] * np.einsum("eid,ejd->eij", moments, grad)
        v = (-v[:, :3].sum(axis=1) + 4 * v[:, 3:].sum(axis=1)) / 9
    else:
        transport_gradient = c[:, None] * np.einsum("ed,eid->ei", v, grad)
        transport = lump[:, :, None] * transport_gradient[:, None, :]
    transport_gradient = c[:, None] * np.einsum("ed,eid->ei", v, grad)
    stabilization = np.zeros_like(diffusion)
    if streamline:
        verts = mesh.nodes[mesh.cells]
        h = np.max(np.linalg.norm(verts[:, :, None] - verts[:, None, :], axis=3), axis=(1, 2))
        speed = c * np.linalg.norm(v, axis=1)
        tau = np.minimum(
            h / (2 * np.maximum(speed, np.finfo(float).tiny)), h**2 / (12 * eigenvalues[:, 0])
        )
        stabilization = (measure * tau)[:, None, None] * (
            transport_gradient[:, :, None] * transport_gradient[:, None, :]
        )
    size = len(mesh.nodes)
    rows = np.repeat(mesh.cells, d + 1, axis=1).ravel()
    cols = np.tile(mesh.cells, (1, d + 1)).ravel()

    def matrix(values):
        return sparse.coo_matrix((values.ravel(), (rows, cols)), shape=(size, size)).tocsr()

    def vector(values):
        return np.bincount(mesh.cells.ravel(), weights=values.ravel(), minlength=size)

    return ThermalAssembly(
        mesh,
        vector(lump),
        vector(c[:, None] * lump),
        matrix(diffusion),
        matrix(transport),
        matrix(stabilization),
        vector(q[:, None] * lump),
    )


def triangle_quadrature():
    """Seven-point degree-five rule, normalized to unit total weight."""
    a, b = 0.059715871789770, 0.470142064105115
    c, d = 0.797426985353087, 0.101286507323456
    return (
        np.array(
            [
                [1 / 3, 1 / 3, 1 / 3],
                [a, b, b],
                [b, a, b],
                [b, b, a],
                [c, d, d],
                [d, c, d],
                [d, d, c],
            ]
        ),
        np.array([0.225, *([0.132394152788506] * 3), *([0.125939180544827] * 3)]),
    )


def add_facet_load(assembly, facets, flux):
    """Add an inward thermal flux once per supplied facet, including interfaces."""
    mesh = assembly.mesh
    faces = np.asarray(facets)
    if faces.ndim != 2 or faces.shape[1] != mesh.dimension or faces.dtype.kind not in "iu":
        raise ValueError("Flux facets must contain integer facet connectivity")
    if not len(faces):
        return
    if faces.min() < 0 or faces.max() >= len(mesh.nodes):
        raise ValueError("Flux facets must reference existing nodes")
    known, _, _ = mesh.facets()
    supplied = np.sort(faces, axis=1)
    if len(np.unique(supplied, axis=0)) != len(faces):
        raise ValueError("A flux facet must be supplied once")
    if not set(map(tuple, supplied)).issubset(set(map(tuple, known))):
        raise ValueError("Flux facets must belong to the mesh")
    q = np.broadcast_to(real_array(flux, "Boundary flux"), (len(faces),))
    if not np.isfinite(q).all():
        raise ValueError("Boundary flux must be finite")
    x = mesh.nodes[faces]
    if mesh.dimension == 2:
        area = np.linalg.norm(x[:, 1] - x[:, 0], axis=1)
        if mesh.axisymmetric:
            weights = 2 * np.pi * area[:, None] * (x[:, :, 0].sum(axis=1)[:, None] + x[:, :, 0]) / 6
        else:
            weights = np.repeat((area / 2)[:, None], 2, axis=1)
    else:
        area = np.linalg.norm(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=1) / 2
        weights = np.repeat((area / 3)[:, None], 3, axis=1)
    assembly.load += np.bincount(
        faces.ravel(), weights=(q[:, None] * weights).ravel(), minlength=len(mesh.nodes)
    )
