"""Deterministic reference modes for the GPU benchmark."""

import heapq
import numpy as np
from .validation import integer


def lowest_modes(n, dimension, rank):
    n, rank = integer(n, "Grid", 1), integer(rank, "Rank")
    if dimension not in (2, 3) or rank > n**dimension:
        raise ValueError("Invalid reference dimension or rank")
    values = 4 * (n + 1)**2 * np.sin(np.pi * np.arange(1, n + 1) / (2 * (n + 1)))**2
    def value(mode):
        return float(sum(values[j - 1] for j in mode))
    first = (1,) * dimension
    heap, seen, result = [(value(first), first)], {first}, []
    while len(result) < rank:
        _, mode = heapq.heappop(heap)
        result.append(mode)
        for axis in range(dimension):
            if mode[axis] < n:
                neighbor = list(mode)
                neighbor[axis] += 1
                neighbor = tuple(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    heapq.heappush(heap, (value(neighbor), neighbor))
    return result


def analytical_reference(n, dimension, rank):
    """Match the manuscript benchmark's eigenvalue/lexicographic ordering."""
    modes = lowest_modes(n, dimension, rank)
    coordinate = np.arange(1, n + 1) / (n + 1)
    one_d = np.sqrt(2 / (n + 1)) * np.sin(
        np.pi * np.arange(1, n + 1)[:, None] * coordinate
    )
    basis = np.empty((n**dimension, rank))
    for column, mode in enumerate(modes):
        vector = one_d[mode[0] - 1]
        for index in mode[1:]:
            vector = np.multiply.outer(vector, one_d[index - 1])
        basis[:, column] = vector.ravel()
    return basis, modes
