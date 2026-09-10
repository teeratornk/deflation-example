"""Nested simplex refinement with inherited materials and P1 prolongation."""

from itertools import combinations

import numpy as np
from scipy import sparse

from .meshes import ThermalMesh, orient_cells


def refine(mesh):
    """Split triangles into four and tetrahedra into eight conforming children."""
    d, cells, nodes = mesh.dimension, mesh.cells, mesh.nodes
    edges = list(combinations(range(d + 1), 2))
    pairs = np.concatenate([cells[:, e] for e in edges])
    unique, inverse = np.unique(np.sort(pairs, axis=1), axis=0, return_inverse=True)
    midpoints = nodes[unique].mean(axis=1)
    new_nodes = np.vstack((nodes, midpoints))
    local = np.column_stack((cells, len(nodes) + inverse.reshape(len(edges), -1).T))
    if d == 2:
        template = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2], [3, 5, 4]])
    else:
        # Local edge order: 01,02,03,12,13,23. The central octahedron
        # uses diagonal 01--23. Every shared face has the same subdivision.
        template = np.array([[0, 4, 5, 6], [1, 4, 7, 8], [2, 5, 7, 9], [3, 6, 8, 9],
                             [4, 9, 5, 6], [4, 9, 6, 8], [4, 9, 8, 7], [4, 9, 7, 5]])
    children = orient_cells(new_nodes, local[:, template].reshape(-1, d + 1))
    parent = np.repeat(np.arange(len(cells)), len(template))
    # Only midpoint edges lying on a Dirichlet boundary facet inherit the BC.
    facets, counts, _ = mesh.facets()
    boundary_facets = facets[(counts == 1) & np.all(np.isin(facets, mesh.dirichlet), axis=1)]
    boundary_edges = {tuple(sorted(f[e] for e in pair))
                      for f in boundary_facets for pair in combinations(range(d), 2)}
    added_pins = [len(nodes) + i for i, e in enumerate(unique) if tuple(e) in boundary_edges]
    result = ThermalMesh(new_nodes, children, mesh.materials[parent],
                         np.r_[mesh.dirichlet, np.asarray(added_pins, dtype=int)],
                         mesh.axisymmetric)
    rows = np.r_[np.arange(len(nodes)), np.repeat(np.arange(len(nodes), len(new_nodes)), 2)]
    cols = np.r_[np.arange(len(nodes)), unique.ravel()]
    data = np.r_[np.ones(len(nodes)), np.full(2 * len(unique), 0.5)]
    P = sparse.coo_matrix((data, (rows, cols)), shape=(len(new_nodes), len(nodes))).tocsr()
    return result, P, parent


def refine_quadratic_flow(coarse, fine, parent, velocity):
    """Evaluate the original P2 polynomial at each child P2 node."""
    if coarse.dimension != 2 or velocity.shape != (len(coarse.cells), 6, 2):
        raise ValueError("Quadratic flow refinement requires triangle P2 vectors")
    vertices = fine.nodes[fine.cells]
    points = np.concatenate((vertices,
                             ((vertices[:, 0] + vertices[:, 1]) / 2)[:, None],
                             ((vertices[:, 0] + vertices[:, 2]) / 2)[:, None],
                             ((vertices[:, 1] + vertices[:, 2]) / 2)[:, None]), axis=1)
    old = coarse.nodes[coarse.cells[parent]]
    local = np.einsum("eqd,edj->eqj", points - old[:, :1],
                      np.linalg.inv(old[:, 1:] - old[:, :1]))
    bary = np.concatenate((1 - local.sum(axis=2, keepdims=True), local), axis=2)
    shape = np.concatenate((bary * (2 * bary - 1),
                             (4 * bary[:, :, 0] * bary[:, :, 1])[:, :, None],
                             (4 * bary[:, :, 0] * bary[:, :, 2])[:, :, None],
                             (4 * bary[:, :, 1] * bary[:, :, 2])[:, :, None]), axis=2)
    return np.einsum("eqa,ead->eqd", shape, velocity[parent])
