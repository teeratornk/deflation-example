"""Temperature-based elimination for steady and transient buoyant CHT control.

Every evaluation solves the nonlinear momentum equations for the trial
temperature and recovers the distributed source from the coupled thermal
equations. A differentiable, locally unique flow branch is required. Frozen
Jacobian factors define the exact discrete tangent and adjoint at that point.
"""

from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .coupled_derivatives import ControlJacobian, buoyancy_jacobian, thermal_velocity_jacobian
from .coupled_flow_solve import solve_momentum
from .mesh_control import WeightedReducedOperator
from .meshes import assemble_thermal
from .solvers import relative_norm
from .validation import integer, positive_real


class FlowEvaluationError(RuntimeError):
    """An inadmissible nonlinear trial, with the complete failed flow record."""

    def __init__(self, slab, result, metrics, temperature=None):
        self.slab, self.result, self.metrics = slab, result, metrics
        self.temperature = None if temperature is None else np.array(temperature, copy=True)
        super().__init__(f"Momentum solve at slab {slab} terminated with {result.status}")


@dataclass
class CoupledEvaluation:
    state: np.ndarray
    control: np.ndarray
    flows: tuple
    jacobian: ControlJacobian
    frozen_operator: sparse.csr_matrix
    flow_metrics: tuple
    seconds: float


class CoupledControlProblem:
    def __init__(
        self,
        flow,
        conductivity,
        capacity,
        source,
        boundary_indices,
        boundary_values,
        initial_flow,
        *,
        alpha=1e-3,
        expansion=0.0,
        temperature_scale=1.0,
        temperature_offset=300.0,
        buoyancy_reference=300.0,
        velocity_scale=1.0,
        time_scale=1.0,
        physical_steps=None,
        initial_temperature=None,
        thermal_boundary=0.0,
        pressure_gauge=None,
        acceleration=None,
        flow_tolerance=1e-11,
        flow_cap=100,
        flow_continuation=False,
    ):
        self.flow, self.mesh = flow, flow.mesh
        self.free = self.mesh.free.copy()
        self.spatial_size = len(self.free)
        self.conductivity = np.asarray(conductivity, dtype=float).copy()
        self.capacity = np.asarray(capacity, dtype=float).copy()
        self.source = np.asarray(source, dtype=float).copy()
        self.boundary_indices = np.asarray(boundary_indices).copy()
        self.boundary_values = np.asarray(boundary_values, dtype=float).copy()
        self.initial_flow = initial_flow
        self.alpha = positive_real(alpha, "Regularization")
        self.expansion = float(expansion)
        self.temperature_scale = positive_real(temperature_scale, "Temperature scale")
        self.temperature_offset = float(temperature_offset)
        self.buoyancy_reference = float(buoyancy_reference)
        if (
            not np.isfinite(
                [self.expansion, self.temperature_offset, self.buoyancy_reference]
            ).all()
            or self.expansion < 0
        ):
            raise ValueError("Finite temperatures and nonnegative expansion are required")
        self.velocity_scale = positive_real(velocity_scale, "Velocity scale")
        self.time_scale = positive_real(time_scale, "Time scale")
        self.physical_steps = np.asarray(
            [] if physical_steps is None else physical_steps, dtype=float
        )
        if (
            self.physical_steps.ndim != 1
            or not np.isfinite(self.physical_steps).all()
            or np.any(self.physical_steps <= 0)
        ):
            raise ValueError("Physical time steps must be a positive finite vector")
        self.slabs = max(1, len(self.physical_steps))
        self.steps = self.physical_steps / self.time_scale
        self.size = self.slabs * self.spatial_size
        self.initial = (
            np.zeros(self.spatial_size)
            if initial_temperature is None
            else np.asarray(initial_temperature, dtype=float).copy()
        )
        if self.initial.shape != (self.spatial_size,) or not np.isfinite(self.initial).all():
            raise ValueError("Initial temperature must match the spatial unknowns")
        self.thermal_boundary = np.broadcast_to(
            thermal_boundary, (len(self.mesh.dirichlet),)
        ).copy()
        if not np.isfinite(self.thermal_boundary).all():
            raise ValueError("Thermal boundary values must be finite")
        self.pressure_gauge = pressure_gauge
        self.acceleration = (
            np.zeros_like(flow.quadrature_points)
            if acceleration is None
            else np.asarray(acceleration, dtype=float).copy()
        )
        flow.load(self.acceleration)
        self.flow_tolerance = positive_real(flow_tolerance, "Flow tolerance")
        self.flow_cap = integer(flow_cap, "Flow iteration cap", 1)
        if not isinstance(flow_continuation, bool):
            raise ValueError("Flow continuation must be Boolean")
        self.flow_continuation = flow_continuation
        self.assembly = self.assemble(initial_flow.velocity)
        mass = self.assembly.mass[self.free]
        self.weights = mass / mass.mean()
        self.objective_scale = float(mass.mean())
        if len(self.steps):
            self.weights = np.kron(self.steps / self.steps.mean(), self.weights)
            self.objective_scale *= float(self.steps.mean())
        fixed = np.r_[self.boundary_indices, flow.nv + self.boundary_indices]
        if pressure_gauge is not None:
            fixed = np.r_[fixed, 2 * flow.nv + pressure_gauge[0]]
        self.flow_free = np.setdiff1d(np.arange(flow.size), fixed)
        self.load_derivative = buoyancy_jacobian(flow, self.expansion, self.temperature_scale)[
            self.flow_free
        ][:, self.free].tocsr()
        self.momentum_mass = sparse.block_diag(
            (flow.mass, flow.mass, sparse.csr_matrix((flow.np, flow.np))), format="csr"
        )[self.flow_free][:, self.flow_free]

    def assemble(self, velocity):
        return assemble_thermal(
            self.mesh,
            self.conductivity,
            self.capacity,
            self.velocity_scale * self.flow.thermal_velocity(velocity),
            self.source,
            streamline=True,
        )

    def full_temperature(self, state):
        full = np.empty(len(self.mesh.nodes))
        full[self.free] = state
        full[self.mesh.dirichlet] = self.thermal_boundary
        return full

    def evaluate(self, state, initial=None):
        start = time.perf_counter()
        state = np.asarray(state, dtype=float).copy()
        if state.shape != (self.size,) or not np.isfinite(state).all():
            raise ValueError("State must be a finite complete temperature trajectory")
        Y = state.reshape(self.slabs, self.spatial_size)
        previous_flow, previous_temperature = self.initial_flow, self.initial
        factors, history, velocity_actions, diagonals, lower = [], [], [], [], []
        controls, flows, metrics = [], [], []
        for n, yn in enumerate(Y):
            full = self.full_temperature(yn)
            acceleration = self.acceleration + self.flow.buoyancy(
                self.temperature_offset + self.temperature_scale * full,
                self.buoyancy_reference,
                self.expansion,
            )
            dt = float(self.physical_steps[n]) if len(self.physical_steps) else None
            result = solve_momentum(
                self.flow,
                acceleration,
                self.boundary_indices,
                self.boundary_values,
                initial=previous_flow if initial is None else initial.flows[n],
                previous=previous_flow.velocity if dt is not None else None,
                time_step=dt,
                pressure_gauge=self.pressure_gauge,
                tolerance=self.flow_tolerance,
                max_iterations=self.flow_cap,
                continuation=self.flow_continuation,
            )
            checks = self.flow.verify(
                result,
                acceleration,
                self.boundary_indices,
                self.boundary_values,
                previous=previous_flow.velocity if dt is not None else None,
                time_step=dt,
                pressure_gauge=self.pressure_gauge,
            )
            if (
                result.status != "converged"
                or not np.isfinite(list(checks.values())).all()
                or max(checks.values()) > self.flow_tolerance
            ):
                raise FlowEvaluationError(n, result, checks, full)
            assembly = self.assemble(result.velocity)
            mass = assembly.mass[self.free]
            inverse_mass = sparse.diags(1 / mass)
            A = (inverse_mass @ assembly.stiffness[self.free][:, self.free]).tocsr()
            control = (assembly.stiffness @ full - assembly.load)[self.free] / mass
            if dt is not None:
                C = assembly.capacity[self.free] / mass / self.steps[n]
                control += C * (yn - previous_temperature)
                A += sparse.diags(C)
                if n:
                    lower.append(-sparse.diags(C))
            derivative = thermal_velocity_jacobian(
                self.flow,
                result.velocity,
                full,
                self.capacity,
                self.conductivity,
                self.velocity_scale,
            )
            velocity_actions.append(
                (inverse_mass @ derivative[self.free][:, self.flow_free]).tocsr()
            )
            J = self.flow.operator(result.velocity, time_step=dt) + self.flow.convection_derivative(
                result.velocity
            )
            factors.append(splu(J[self.flow_free][:, self.flow_free].tocsc()))
            history.append(
                self.momentum_mass / dt if dt else sparse.csr_matrix(self.momentum_mass.shape)
            )
            diagonals.append(A)
            controls.append(control)
            flows.append(result)
            metrics.append(checks)
            previous_flow, previous_temperature = result, yn
        blocks = [[None] * self.slabs for _ in range(self.slabs)]
        for n, A in enumerate(diagonals):
            blocks[n][n] = A
            if n:
                blocks[n][n - 1] = lower[n - 1]
        thermal = sparse.bmat(blocks, format="csr")
        jacobian = ControlJacobian(
            thermal, velocity_actions, self.load_derivative, factors, history
        )
        return CoupledEvaluation(
            state,
            np.concatenate(controls),
            tuple(flows),
            jacobian,
            thermal,
            tuple(metrics),
            time.perf_counter() - start,
        )

    def objective_gradient(self, evaluation, desired):
        desired = np.asarray(desired, dtype=float)
        if desired.shape != (self.size,) or not np.isfinite(desired).all():
            raise ValueError("Desired temperature must match the complete problem")
        error = evaluation.state - desired
        value = 0.5 * np.sum(self.weights * (error**2 + self.alpha * evaluation.control**2))
        gradient = self.weights * error + self.alpha * (
            evaluation.jacobian.T @ (self.weights * evaluation.control)
        )
        # The common positive objective scale does not affect the optimizer.
        return float(value), np.asarray(gradient)

    def preconditioning_diagonal(self, evaluation, damping=0.0):
        """Positive frozen-velocity approximation, common to all inner solvers."""
        return (
            WeightedReducedOperator(evaluation.frozen_operator, self.weights, self.alpha).diagonal()
            + damping * self.weights
        )

    def verify(self, evaluation):
        """Rebuild momentum and thermal equations at the same retained fields.

        The energy check uses independently integrated boundary advection and
        discrete Dirichlet reactions. It requires constant fluid heat capacity,
        as in the transformer model; material jumps in the solid are retained.
        """
        fluid_capacity = self.capacity[self.flow.fluid_cells]
        if not np.allclose(fluid_capacity, fluid_capacity[0], rtol=1e-14, atol=0):
            raise ValueError("Boundary energy verification requires constant fluid capacity")
        previous_y = self.full_temperature(self.initial)
        previous_v = self.initial_flow.velocity
        rows = []
        for n, (yn, un, flow_result) in enumerate(
            zip(
                evaluation.state.reshape(self.slabs, -1),
                evaluation.control.reshape(self.slabs, -1),
                evaluation.flows,
                strict=True,
            )
        ):
            y = self.full_temperature(yn)
            u = np.zeros(len(self.mesh.nodes))
            u[self.free] = un
            dt = self.physical_steps[n] if len(self.physical_steps) else None
            assembly = self.assemble(flow_result.velocity)
            storage = (
                np.zeros_like(y)
                if dt is None
                else assembly.capacity * (y - previous_y) / self.steps[n]
            )
            rhs = assembly.load + assembly.mass * u
            reaction = assembly.stiffness @ y + storage - rhs
            acceleration = self.acceleration + self.flow.buoyancy(
                self.temperature_offset + self.temperature_scale * y,
                self.buoyancy_reference,
                self.expansion,
            )
            flow_checks = self.flow.verify(
                flow_result,
                acceleration,
                self.boundary_indices,
                self.boundary_values,
                previous=previous_v if dt is not None else None,
                time_step=dt,
                pressure_gauge=self.pressure_gauge,
            )
            flux = self.flow.boundary_flux(flow_result.velocity)
            components = {
                "storage": float(storage.sum()),
                "boundary_advection": float(
                    self.velocity_scale
                    * fluid_capacity[0]
                    * self.flow.boundary_flux(flow_result.velocity, scalar=y).sum()
                ),
                "control_input": float(assembly.mass @ u),
                "background_input": float(assembly.load.sum()),
                "dirichlet_supply": float(reaction[self.mesh.dirichlet].sum()),
            }
            defect = (
                components["storage"]
                + components["boundary_advection"]
                - components["control_input"]
                - components["background_input"]
                - components["dirichlet_supply"]
            )
            scale = max(
                sum(abs(v) for v in components.values()),
                float(np.abs(reaction[self.mesh.dirichlet]).sum()),
                1e-30,
            )
            rows.append(
                {
                    "slab": n,
                    **flow_checks,
                    "thermal_relative_residual": relative_norm(reaction[self.free], rhs[self.free]),
                    "mass_relative_imbalance": float(
                        abs(flux.sum()) / max(np.abs(flux).sum() / 2, 1e-30)
                    ),
                    "energy": {
                        **components,
                        "defect": defect,
                        "relative_defect": abs(defect) / scale,
                    },
                }
            )
            previous_y, previous_v = y, flow_result.velocity
        return rows
