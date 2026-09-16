"""Check map signs, relaxation and complex mode verification independently."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

directory = Path(__file__).parents[1] / "examples/coupled_optimization/fixed_point"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location(
    "coupling_spectrum", directory / "coupling_spectrum.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.pop(0)


def test_fixed_point_block_sign_and_complex_action():
    rng = np.random.default_rng(910)
    F = np.array([[2.0, 1.0], [1.0, 0.0]])
    L, S = rng.normal(size=(2, 8)), rng.normal(size=(8, 2))
    T = np.diag(np.arange(1.0, 9.0))
    H = sparse.csr_matrix(np.block([[F, -L], [S, T]]))
    operator, report = module.fixed_point_operator(H, 2)
    exact = -np.linalg.solve(T, S @ np.linalg.solve(F, L))
    x = rng.normal(size=8) + 1j * rng.normal(size=8)
    np.testing.assert_allclose(operator @ x, exact @ x, atol=1e-13)
    assert report["flow_test_linear_relative_residual"] < 1e-13


def test_known_unstable_map_with_indefinite_flow_block():
    diagonal = np.arange(1.0, 9.0) / 4
    F = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    H = sparse.csr_matrix(np.block([[F, -F], [-np.diag(diagonal), np.eye(8)]]))
    result = module.spectrum(H, 8, count=3)
    assert result["status"] == "converged"
    np.testing.assert_allclose(
        sorted(r["real"] for r in result["modes"]), [1.5, 1.75, 2.0], atol=1e-8
    )
    assert max(r["eigen_residual"] for r in result["modes"]) < 1e-8
    assert all(r["relaxed_moduli"]["0.5"] > 1 for r in result["modes"])
