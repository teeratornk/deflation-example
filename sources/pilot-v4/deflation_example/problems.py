"""Cartesian benchmark definitions matching the manuscript's checked quadratic cases.

All coordinates are interior nodes, flattened in C order. In the thermal
preset the discrete transport is v1 * d/dx2 + v2 * d/dx1, as specified in
the accompanying paper. This convention is explicit to avoid exchanging axes.
"""

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from .validation import finite_real, integer, positive_real


def laplacian(n, dim=2):
    """Positive Dirichlet Laplacian on the unit square/cube."""
    n = integer(n, "Grid size", 2)
    dim = integer(dim, "Dimension", 2)
    if dim not in (2, 3):
        raise ValueError("Use at least two interior nodes per axis and dimension 2 or 3")
    T = sparse.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    T *= (n + 1) ** 2
    L = sparse.csr_matrix((n**dim, n**dim))
    for axis in range(dim):
        term = sparse.csr_matrix([[1.0]])
        for j in range(dim):
            term = sparse.kron(term, T if axis == j else sparse.eye(n), format="csr")
        L += term
    return L


def thermal_convection(n, coords):
    h = 1 / (n + 1)
    derivative = sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [-1, 1]) / (2 * h)
    d1 = sparse.kron(derivative, sparse.eye(n), format="csr")
    d2 = sparse.kron(sparse.eye(n), derivative, format="csr")
    x1, x2 = coords
    v1 = 100 * np.sin(np.pi * x1) * np.cos(np.pi * x2)
    v2 = -100 * np.cos(np.pi * x1) * np.sin(np.pi * x2)
    return (sparse.diags(v1) @ d2 + sparse.diags(v2) @ d1).tocsr()


def cht_operator(n, coords):
    """Harmonic face diffusion, solid conductivity 100, prescribed fluid Re=50."""
    N, h = n**3, 1 / (n + 1)
    indices = np.arange(N)
    ijk = np.array(np.unravel_index(indices, (n, n, n)))
    kappa = np.where(coords[2] < 0.5, 100.0, 1.0)
    diagonal = np.zeros(N)
    rows, cols, values = [], [], []
    # Missing neighbors have zero Dirichlet data.
    for axis, stride in enumerate([n * n, n, 1]):
        for sign in [-1, 1]:
            valid = (ijk[axis] + sign >= 0) & (ijk[axis] + sign < n)
            src = indices[valid]
            dst = src + sign * stride
            coeff = 2 * kappa[src] * kappa[dst] / (kappa[src] + kappa[dst]) / h**2
            rows.append(src)
            cols.append(dst)
            values.append(-coeff)
            diagonal[src] += coeff
            diagonal[~valid] += kappa[~valid] / h**2
    rows.append(indices)
    cols.append(indices)
    values.append(diagonal)
    D = sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)
    )
    velocity = np.where(coords[2] >= 0.5, 50 * 4 * coords[2] * (1 - coords[2]), 0.0)
    rows, cols, values = [], [], []
    for sign in [1, -1]:
        valid = (ijk[0] + sign >= 0) & (ijk[0] + sign < n) & (velocity != 0)
        src = indices[valid]
        rows.append(src)
        cols.append(src + sign * n * n)
        values.append(sign * velocity[valid] / (2 * h))
    C = sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)
    )
    return D, C


@dataclass
class Problem:
    name: str
    n: int
    dim: int
    alpha: float
    coordinates: np.ndarray
    A: sparse.csr_matrix
    H: sparse.csr_matrix
    reference: sparse.csr_matrix

    def target(self, theta):
        theta = finite_real(theta, "Target angle")
        if self.dim == 2:
            angles = 5 * np.pi / 4 + np.arange(4) * np.pi / 2 + theta
            centers = 0.5 + np.sqrt(0.08) * np.column_stack([np.cos(angles), np.sin(angles)])
            sigma = 0.1
        else:
            centers = np.array([[0.3, 0.3, 0.3], [0.3, 0.7, 0.3], [0.7, 0.3, 0.7], [0.7, 0.7, 0.7]])
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            centers[:, :2] = (centers[:, :2] - 0.5) @ rotation.T + 0.5
            sigma = 0.12
        return sum(
            np.exp(-np.sum((self.coordinates - c[:, None]) ** 2, axis=0) / (2 * sigma**2))
            for c in centers
        )


def build_problem(name="diffusion", n=None, alpha=1e-3):
    """Create a quadratic problem; 'thermal' fixes Ra=100 and gamma=0."""
    if not isinstance(name, str) or name not in {"diffusion", "thermal", "cht"}:
        raise ValueError("Problem must be diffusion, thermal or cht")
    alpha = positive_real(alpha, "Regularization")
    dim = 3 if name == "cht" else 2
    n = (12 if dim == 3 else 32) if n is None else n
    L = laplacian(n, dim)
    h = 1 / (n + 1)
    coords = np.array(np.meshgrid(*([np.linspace(h, 1 - h, n)] * dim), indexing="ij")).reshape(
        dim, -1
    )
    if name == "thermal":
        A = L + thermal_convection(n, coords)
    elif name == "cht":
        D, C = cht_operator(n, coords)
        A = D + C
    else:
        A = L
    H = (sparse.eye(n**dim) + alpha * (A.T @ A)).tocsr()
    reference = (sparse.eye(n**dim) + alpha * (L.T @ L)).tocsr() if name == "cht" else H
    return Problem(name, n, dim, alpha, coords, A.tocsr(), H, reference)


def sine_modes(n, dim, rank):
    """Low tensor sine modes; stable sorting fixes a choice inside repeated clusters."""
    n = integer(n, "Grid size", 2)
    dim = integer(dim, "Dimension", 2)
    if dim not in (2, 3):
        raise ValueError("Dimension must be 2 or 3")
    rank = integer(rank, "Reference rank")
    if rank > n**dim:
        raise ValueError("Reference rank must be an integer between zero and the full dimension")
    mu = 4 * (n + 1) ** 2 * np.sin(np.pi * np.arange(1, n + 1) / (2 * (n + 1))) ** 2
    ordered = np.argsort(sum(np.meshgrid(*([mu] * dim), indexing="ij")).ravel(), kind="stable")[
        :rank
    ]
    V = np.sqrt(2 / (n + 1)) * np.sin(
        np.pi * np.outer(np.arange(1, n + 1), np.arange(1, n + 1)) / (n + 1)
    )
    Z = np.empty((n**dim, rank))
    for j, mode in enumerate(np.array(np.unravel_index(ordered, (n,) * dim)).T):
        v = np.array([1.0])
        for axis in mode:
            v = np.multiply.outer(v, V[:, axis])
        Z[:, j] = v.ravel()
    return Z


def reference_modes(problem, rank=20):
    """Build once per sequence: sine modes for diffusion/CHT, eigsh for thermal."""
    N = problem.H.shape[0]
    rank = integer(rank, "Reference rank", 1)
    if rank >= N:
        raise ValueError("Reference rank must be an integer between 1 and N-1")
    if problem.name != "thermal":
        return sine_modes(problem.n, problem.dim, rank)
    values, vectors = eigsh(
        problem.reference, k=rank, sigma=0, which="LM", v0=np.random.default_rng(13).normal(size=N)
    )
    return vectors[:, np.argsort(values)]
