"""Device block products are chunked into bounded, contiguous column ranges."""

import pytest

from deflation_example.coupled_hybrid_solver import chunk_bounds
from deflation_example.coupled_optimize import solver_options


@pytest.mark.parametrize(
    ("columns", "width", "expected"),
    [
        (3, 100, [(0, 3)]),
        (100, 100, [(0, 100)]),
        (400, 100, [(0, 100), (100, 200), (200, 300), (300, 400)]),
        (150, 100, [(0, 75), (75, 150)]),
        (7, 2, [(0, 2), (2, 4), (4, 5), (5, 7)]),
        (0, 5, [(0, 0)]),
    ],
)
def test_chunks_cover_the_columns_in_order_within_the_width(columns, width, expected):
    bounds = chunk_bounds(columns, width)
    assert bounds == expected
    assert all(b - a <= width for a, b in bounds)
    assert bounds[0][0] == 0 and bounds[-1][1] == columns
    assert all(bounds[k][1] == bounds[k + 1][0] for k in range(len(bounds) - 1))


def test_runner_forwards_the_chunk_width_only_to_the_hybrid_backend():
    class HybridCoupledSolver:
        pass

    class StudySolver:
        pass

    cfg = {"inner_refresh": 3, "hybrid_block_max_columns": 7, "hybrid_block_min_columns": 4}
    hybrid = solver_options(cfg, HybridCoupledSolver)
    assert hybrid == {
        "refresh": 3,
        "block_min_columns": 4,
        "block_max_columns": 7,
        "coarse_device": "cpu",
    }
    assert solver_options(cfg, StudySolver) == {"refresh": 3}
    with pytest.raises(ValueError):
        solver_options({"hybrid_block_max_columns": 0}, HybridCoupledSolver)
