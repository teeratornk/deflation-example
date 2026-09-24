"""Exact affine thermal control with a fixed computed velocity.

Source and storage actions retain the thermal discretization of the coupled
model. Only temperature-to-flow feedback is removed. The complete trajectory
is one constrained quadratic problem, with a block lower-triangular state map.
"""

import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .coupled_control import CoupledEvaluation
from .coupled_derivatives import ControlJacobian, GaussNewtonOperator
from .coupled_optimizer import box_kkt, box_quadratic
from .validation import real_array


class FixedFlowProblem:
    """Thermal trajectory model sharing the declared physical data and weights."""

    def __init__(self, physical):
        if physical.expansion != 0:
            raise ValueError("The fixed-flow model requires feedback_multiplier=0")
        self.physical = physical
        self.assembly = physical.assembly
        self.evaluation_callback = None
        self.evaluation_count = 0
        free, n, count = physical.free, physical.spatial_size, physical.slabs
        assembly = self.assembly
        source = assembly.source_action[free][:, free].tocsc()
        factor = splu(source) if assembly.consistent else None
        mass = assembly.mass[free]
        inverse_mass = sparse.diags(1 / mass)
        K = assembly.stiffness[free][:, free].tocsr()
        boundary = physical.full_temperature(np.zeros(n))
        load = (assembly.stiffness @ boundary - assembly.load)[free]
        blocks = [[None] * count for _ in range(count)]
        offsets = []
        for level in range(count):
            block, forcing = K, load.copy()
            if len(physical.steps):
                storage = (assembly.storage[free][:, free] / physical.steps[level]).tocsr()
                block = block + storage
                if level:
                    blocks[level][level - 1] = -storage if factor else -inverse_mass @ storage
                else:
                    forcing -= storage @ physical.initial
            blocks[level][level] = block if factor else inverse_mass @ block
            offsets.append(factor.solve(forcing) if factor else forcing / mass)
        thermal = sparse.bmat(blocks, format="csr")
        momentum_size = len(physical.flow_free)
        zero_action = sparse.csr_matrix((n, momentum_size))
        zero_history = sparse.csr_matrix((momentum_size, momentum_size))
        self.jacobian = ControlJacobian(
            thermal,
            [zero_action] * count,
            physical.load_derivative,
            [None] * count,
            [zero_history] * count,
            source_factors=[factor] * count if factor else None,
        )
        self.offset = np.concatenate(offsets)
        self.frozen_operator = (
            sparse.diags(np.tile(1 / mass, count)) @ thermal if factor else thermal
        ).tocsr()
        self.H = GaussNewtonOperator(self.jacobian, self.weights, self.alpha)

    def __getattr__(self, name):
        return getattr(self.physical, name)

    def evaluate(self, state, initial=None):
        """Apply the affine map; saved flow guesses never alter the prescribed flow."""
        start = time.perf_counter()
        state = real_array(state, "Temperature trajectory").copy()
        if state.shape != (self.size,) or not np.isfinite(state).all():
            raise ValueError("Temperature must be a finite full trajectory")
        self.evaluation_count += 1
        evaluation = CoupledEvaluation(
            state,
            self.jacobian @ state + self.offset,
            (self.physical.initial_flow,) * self.slabs,
            self.jacobian,
            self.frozen_operator,
            (),
            time.perf_counter() - start,
        )
        if self.evaluation_callback is not None:
            self.evaluation_callback(
                {
                    "evaluation": self.evaluation_count,
                    "completed_slabs": self.slabs,
                    "total_slabs": self.slabs,
                    "status": "complete",
                    "seconds": evaluation.seconds,
                    "physics": "prescribed_flow",
                    "momentum_solves": 0,
                }
            )
        return evaluation

    def verify(self, evaluation, *, local_mass=False):
        return self.physical.verify(evaluation, local_mass=local_mass)

    def verify_adjoint(self, evaluation, desired):
        """Reassemble source and thermal transpose equations independently."""
        assembly = self.physical.assemble(self.physical.initial_flow.velocity)
        free = self.free
        S = assembly.source_action[free][:, free].tocsc()
        factor = splu(S)
        K = assembly.stiffness[free][:, free].tocsr()
        rhs = (self.alpha * self.weights * evaluation.control).reshape(self.slabs, -1)
        adjoint = factor.solve(rhs.T, trans="T").T
        source_residuals = []
        gradient = (self.weights * (evaluation.state - desired)).reshape(self.slabs, -1).copy()
        for level in range(self.slabs):
            defect = S.T @ adjoint[level] - rhs[level]
            norm = np.linalg.norm(rhs[level])
            source_residuals.append(
                float(np.linalg.norm(defect) / norm if norm else np.linalg.norm(defect))
            )
            gradient[level] += K.T @ adjoint[level]
            if len(self.steps):
                storage = assembly.storage[free][:, free] / self.steps[level]
                gradient[level] += storage.T @ adjoint[level]
                if level:
                    gradient[level - 1] -= storage.T @ adjoint[level]
        reference = self.objective_gradient(evaluation, desired)[1]
        scaled = reference / self.weights
        scale = max(
            1.0,
            np.max(np.abs(evaluation.state - desired)),
            np.max(np.abs(scaled - (evaluation.state - desired))),
        )
        return {
            "procedure": "independent-fixed-flow-source-and-thermal-transpose-v1",
            "momentum_adjoint_applicable": False,
            "maximum_momentum_adjoint_relative_residual": 0.0,
            "maximum_source_adjoint_relative_residual": max(source_residuals),
            "gradient_relative_difference": float(
                np.linalg.norm(gradient.ravel() - reference)
                / max(np.linalg.norm(reference), np.finfo(float).tiny)
            ),
            "gradient_weight_normalized_difference": float(
                np.max(np.abs((gradient.ravel() - reference) / self.weights)) / scale
            ),
        }


def weighted_kkt(problem, state, gradient, desired, lower, upper):
    scaled = gradient / problem.weights
    scale = max(1.0, np.max(np.abs(state - desired)), np.max(np.abs(scaled - (state - desired))))
    return box_kkt(state, scaled, lower, upper, scale)


def solve_trajectory(problem, desired, lower, upper, solver, cfg, initial):
    """Direct PDAS for the exact quadratic, with independent final optimality."""
    from .coupled_frozen_preconditioner import frozen_preconditioner_factory

    evaluation = problem.evaluate(initial)
    diagonal = problem.preconditioning_diagonal(evaluation)
    g = -problem.weights * desired + problem.alpha * (
        problem.jacobian.T @ (problem.weights * problem.offset)
    )
    factory = None
    if cfg["inner_preconditioner"] == "frozen":
        factory = frozen_preconditioner_factory(problem, evaluation, sweeps=cfg["frozen_sweeps"])
    result = box_quadratic(
        problem.H,
        g,
        diagonal,
        lower,
        upper,
        solver,
        initial=initial,
        tolerance=cfg["nonlinear_tolerance"],
        max_steps=cfg["qp_cap"],
        preconditioner_factory=factory,
        kkt_evaluator=lambda state, gradient: weighted_kkt(
            problem, state, gradient, desired, lower, upper
        ),
    )
    final = problem.evaluate(result.x)
    objective, gradient = problem.objective_gradient(final, desired)
    kkt = weighted_kkt(problem, final.state, gradient, desired, lower, upper)
    return final, result, objective, kkt
