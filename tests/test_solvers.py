"""Numerical checks independent of stored iteration or timing targets."""

import unittest
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from deflation_example.solvers import (
    deflated_cg,
    pdas,
    restricted_normal_operator,
    kkt_metrics,
    calibrate_bound,
)


class CoreTests(unittest.TestCase):
    def test_wrong_active_node_must_leave(self):
        H = np.array([[2.0, -1.0], [-1.0, 2.0]])
        result = pdas(H, [0.2, 2.9], 1.0, initial_active=[True, True])
        self.assertEqual(result["status"], "converged")
        np.testing.assert_allclose(result["y"], [0.6, 1.0], atol=1e-12)
        self.assertGreater(sum(r["left"] for r in result["history"]), 0)
        self.assertLess(max(result["kkt"].values()), 1e-12)

    def test_pdas_independent_qp(self):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            B = rng.normal(size=(12, 12))
            H, f = B.T @ B + np.eye(12), rng.normal(size=12)
            bound = rng.uniform(-0.1, 0.2, 12)
            result = pdas(H, f, bound)
            reference = minimize(
                lambda y: (0.5 * y @ H @ y - f @ y, H @ y - f),
                np.minimum(0, bound),
                jac=True,
                method="L-BFGS-B",
                bounds=[(None, b) for b in bound],
                options={"gtol": 1e-11, "ftol": 1e-15, "maxiter": 2000},
            )
            self.assertEqual(result["status"], "converged")
            self.assertLess(abs(result["objective"] - reference.fun), 1e-9)
            self.assertLess(max(result["kkt"].values()), 1e-9)

    def test_coarse_correction_returns_original_solution(self):
        H = np.array([[2.0, 1.0], [1.0, 2.0]])
        result = deflated_cg(H, [0.0, 1.0], np.array([[1.0], [0.0]]), np.diag(H))
        self.assertEqual(result.status, "converged")
        np.testing.assert_allclose(result.x, [-1 / 3, 2 / 3], atol=1e-12)

    def test_deflation_dependent_zero_and_full_bases(self):
        rng = np.random.default_rng(11)
        B = rng.normal(size=(20, 20))
        H = B.T @ B + np.eye(20)
        f = rng.normal(size=20)
        U = np.linalg.eigh(H)[1]
        for Z in [None, np.zeros((20, 2)), U, np.column_stack([U[:, :3], U[:, :3]])]:
            result = deflated_cg(H, f, Z, np.diag(H), refresh=7)
            self.assertEqual(result.status, "converged")
            self.assertLess(result.residual, 1e-10)
            np.testing.assert_allclose(result.x, np.linalg.solve(H, f), rtol=1e-8, atol=1e-10)

    def test_zero_rhs_and_iteration_cap(self):
        result = deflated_cg(np.eye(3), np.zeros(3), x0=np.ones(3))
        self.assertEqual(result.status, "converged")
        self.assertLess(result.residual, 1e-10)
        self.assertEqual(deflated_cg(np.eye(3), np.ones(3), maxiter=0).status, "maxiter")

    def test_non_spd_and_coarse_fallback(self):
        self.assertEqual(deflated_cg(-np.eye(2), np.ones(2)).status, "breakdown")
        H = np.diag([1.0, 100.0])
        result = deflated_cg(H, np.ones(2), np.eye(2), np.diag(H), condition_limit=10)
        self.assertEqual(result.rank, 0)
        self.assertEqual(result.status, "converged")

    def test_matrix_free_restriction(self):
        rng = np.random.default_rng(2)
        A = sparse.csr_matrix(rng.normal(size=(8, 8)))
        I = np.array([0, 2, 3, 7])
        H = sparse.eye(8) + 0.1 * (A.T @ A)
        op = restricted_normal_operator(A, 0.1, I)
        np.testing.assert_allclose(op @ np.eye(4), H.tocsr()[I][:, I].toarray())
        wrong = np.eye(4) + 0.1 * (A[I][:, I].T @ A[I][:, I]).toarray()
        self.assertGreater(np.linalg.norm(op @ np.eye(4) - wrong), 0.1)

    def test_bad_multiplier_detected(self):
        metrics = kkt_metrics(
            np.array([[2.0, -1.0], [-1.0, 2.0]]),
            np.array([0.2, 2.9]),
            np.ones(2),
            np.ones(2),
            np.array([-0.8, 1.9]),
        )
        self.assertGreater(metrics["dual"], 0.2)

    def test_actual_activity_calibration(self):
        from deflation_example.problems import laplacian

        A = laplacian(8)
        H = (sparse.eye(64) + 0.001 * (A.T @ A)).tocsr()
        f = np.linspace(0.2, 1.0, 64)
        bound, actual = calibrate_bound(H, f, 0.2)
        result = pdas(H, f, bound)
        self.assertEqual(result["status"], "converged")
        self.assertAlmostEqual(actual, result["active"].mean())
        self.assertLess(abs(actual - 0.2), 0.04)
