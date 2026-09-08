"""Run with: uv run --locked --no-dev python examples/basic.py"""

import numpy as np
from threadpoolctl import threadpool_limits
from deflation_example import build_problem, reference_modes, pdas, deflated_cg
from deflation_example.solvers import calibrate_bound


def main():
    with threadpool_limits(limits=1):
        problem = build_problem("diffusion", n=24)
        modes = reference_modes(problem, rank=20)
        bound, activity = calibrate_bound(problem.H, problem.target(np.pi / 4), 0.2)
        print(f"Bound: {bound:.6g}; calibrated activity: {activity:.1%}")

        # The modes are reused at every step and every target angle.
        def inner(B, b, inactive):
            tolerance = min(1e-10, 1e-10 / max(np.linalg.norm(b), 1.0))
            return deflated_cg(B, b, modes[inactive], B.diagonal(), rtol=tolerance)

        for angle in [0.0, np.pi / 4, np.pi / 2]:
            result = pdas(problem.H, problem.target(angle), bound, linear_solver=inner)
            print(f"angle={angle:.3f}: {result['status']}, KKT={max(result['kkt'].values()):.2e}")
            if result["status"] != "converged":
                raise RuntimeError("Optimization failed; inspect the returned KKT metrics")


if __name__ == "__main__":
    main()
