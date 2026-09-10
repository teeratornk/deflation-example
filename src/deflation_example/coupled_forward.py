"""Verified segregated flow--temperature steps with a fixed source field.

The thermal builder retains the caller's coefficients and returns a P1 thermal
assembly for the current P2 velocity. Every accepted step checks both equations
at the same final temperature and velocity. No optimization is performed here.
"""

from dataclasses import dataclass
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, MatrixRankWarning

from .axisymmetric_flow import FlowResult
from .validation import real_array, positive_real, integer


@dataclass
class CoupledResult:
    state: np.ndarray
    flow: FlowResult
    status: str
    history: list
    seconds: float


class CoupledForward:
    """One steady solve or one coupled backward-Euler time step.

    Thermal state y means T = temperature_offset + temperature_scale*y.
    Physical dt is divided by time_scale for the caller's thermal equation.
    Momentum always uses seconds and SI velocities. Control uses the thermal
    assembly's declared source units and original nodal mass-lumped loading.
    """

    def __init__(
        self,
        flow,
        thermal_builder,
        velocity_boundary,
        velocity_values,
        *,
        expansion,
        temperature_offset=0.0,
        temperature_scale=1.0,
        time_scale=1.0,
        buoyancy_reference=0.0,
        thermal_boundary=0.0,
        transport_factor=1.0,
        acceleration=None,
        pressure_gauge=None,
        flow_method="picard",
    ):
        self.flow = flow
        self.thermal_builder = thermal_builder
        self.velocity_boundary = np.asarray(velocity_boundary).copy()
        self.velocity_values = real_array(velocity_values, "Velocity boundary").copy()
        self.expansion = float(expansion)
        if not np.isfinite(self.expansion) or self.expansion < 0:
            raise ValueError("Thermal expansion must be finite and nonnegative")
        self.temperature_offset = float(temperature_offset)
        self.temperature_scale = positive_real(temperature_scale, "Temperature scale")
        self.time_scale = positive_real(time_scale, "Time scale")
        self.buoyancy_reference = float(buoyancy_reference)
        if not np.isfinite([self.temperature_offset, self.buoyancy_reference]).all():
            raise ValueError("Reference temperatures must be finite")
        self.transport_factor = positive_real(transport_factor, "Thermal transport factor")
        self.J = flow.mesh.dirichlet
        self.I = flow.mesh.free
        self.boundary = np.broadcast_to(
            real_array(thermal_boundary, "Thermal boundary"), (len(self.J),)
        ).copy()
        if not np.isfinite(self.boundary).all():
            raise ValueError("Thermal boundary must be finite")
        self.acceleration = (
            np.zeros_like(flow.quadrature_points)
            if acceleration is None
            else real_array(acceleration, "External acceleration").copy()
        )
        flow.load(self.acceleration)
        self.pressure_gauge = pressure_gauge
        if flow_method not in {"picard", "newton"}:
            raise ValueError("Unknown flow iteration")
        self.flow_method = flow_method

    def _force(self, state, feedback):
        T = self.temperature_offset + self.temperature_scale * state
        return self.acceleration + self.flow.buoyancy(
            T,
            self.buoyancy_reference,
            self.expansion if feedback else 0.0,
        )

    def _thermal_system(self, velocity, control, previous, time_step):
        assembly = self.thermal_builder(self.flow.thermal_velocity(velocity))
        if assembly.mesh is not self.flow.mesh:
            raise ValueError("Flow and thermal assembly must share one mesh")
        K = assembly.stiffness
        rhs = assembly.load + assembly.mass * control
        if time_step is not None:
            C = assembly.capacity / (time_step / self.time_scale)
            K = K + sparse.diags(C)
            rhs = rhs + C * previous
        return assembly, K.tocsr(), rhs

    def solve(
        self,
        control,
        initial_state,
        initial_flow,
        *,
        feedback=True,
        previous_state=None,
        previous_velocity=None,
        time_step=None,
        tolerance=1e-8,
        max_iterations=100,
        relaxation=0.5,
        flow_cap=100,
        mass_tolerance=1e-6,
        energy_tolerance=1e-6,
        callback=None,
    ):
        start = time.perf_counter()
        tolerance = positive_real(tolerance, "Coupled tolerance")
        mass_tolerance = positive_real(mass_tolerance, "Mass balance tolerance")
        energy_tolerance = positive_real(energy_tolerance, "Energy balance tolerance")
        max_iterations = integer(max_iterations, "Coupling iteration cap", 1)
        relaxation = positive_real(relaxation, "Coupling relaxation")
        if relaxation > 1:
            raise ValueError("Coupling relaxation must not exceed one")
        source = real_array(control, "Saved source").copy()
        source.flags.writeable = False
        state = real_array(initial_state, "Initial thermal iterate").copy()
        n = len(self.flow.mesh.nodes)
        if source.shape != (n,) or state.shape != (n,) or not np.isfinite([source, state]).all():
            raise ValueError("Source and initial thermal iterate must match all mesh nodes")
        state[self.J] = self.boundary
        if (time_step is None) != (previous_state is None) or (time_step is None) != (
            previous_velocity is None
        ):
            raise ValueError(
                "Transient coupling requires time step, previous temperature and velocity"
            )
        if time_step is not None:
            time_step = positive_real(time_step, "Physical time step")
            previous_state = real_array(previous_state, "Previous thermal state").copy()
            if previous_state.shape != (n,) or not np.isfinite(previous_state).all():
                raise ValueError("Previous temperature must match the mesh")
        flow = initial_flow
        history, status = [], "coupling_iteration_cap"
        for step in range(1, max_iterations + 1):
            flow = self.flow.solve(
                self._force(state, feedback),
                self.velocity_boundary,
                self.velocity_values,
                initial=flow,
                previous=previous_velocity,
                time_step=time_step,
                pressure_gauge=self.pressure_gauge,
                tolerance=tolerance * 0.1,
                max_iterations=flow_cap,
                method=self.flow_method,
            )
            if flow.status != "converged":
                history.append(
                    {
                        "coupling_iteration": step,
                        "flow_status": flow.status,
                        "flow_history": flow.history,
                    }
                )
                status = "flow_" + flow.status
                break
            assembly, K, rhs = self._thermal_system(
                flow.velocity, source, previous_state, time_step
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    proposal = spsolve(
                        K[self.I][:, self.I], rhs[self.I] - K[self.I][:, self.J] @ self.boundary
                    )
            except (MatrixRankWarning, RuntimeError):
                status = "thermal_linear_failure"
                break
            if not np.isfinite(proposal).all():
                status = "thermal_nonfinite_iterate"
                break
            factor = relaxation if feedback and self.expansion else 1.0
            state[self.I] += factor * (proposal - state[self.I])
            residual = K @ state - rhs
            thermal_scale = np.linalg.norm(rhs[self.I]) + np.linalg.norm(
                K[self.I][:, self.J] @ self.boundary
            )
            thermal_residual = np.linalg.norm(residual[self.I]) / max(thermal_scale, 1e-30)
            # The new temperature changes buoyancy. Verify the momentum equation
            # again with that temperature before accepting the coupled solution.
            metrics = self.flow.verify(
                flow,
                self._force(state, feedback),
                self.velocity_boundary,
                self.velocity_values,
                previous=previous_velocity,
                time_step=time_step,
                pressure_gauge=self.pressure_gauge,
            )
            balance = self.energy_balance(
                assembly, state, flow.velocity, source, previous_state, time_step
            )
            flux = self.flow.boundary_flux(flow.velocity)
            mass_error = abs(flux.sum()) / max(np.abs(flux).sum() / 2, 1e-30)
            row = {
                "coupling_iteration": step,
                "flow_status": flow.status,
                "flow_iterations": len(flow.history),
                "flow_history": flow.history,
                "thermal_relative_residual": float(thermal_residual),
                **metrics,
                "mass_relative_imbalance": float(mass_error),
                "energy": balance,
            }
            history.append(row)
            if callback is not None:
                callback(row)
            criteria = [
                thermal_residual,
                metrics["momentum_relative_residual"],
                metrics["continuity_relative_residual"],
            ]
            if (
                np.isfinite(criteria).all()
                and max(criteria) <= tolerance
                and mass_error <= mass_tolerance
                and balance["relative_defect"] <= energy_tolerance
            ):
                status = "converged"
                break
        return CoupledResult(state, flow, status, history, time.perf_counter() - start)

    def energy_balance(self, assembly, state, velocity, control, previous, time_step):
        """Independent boundary advection plus discrete Dirichlet reactions.

        Components use the thermal assembly's energy-rate units. Boundary supply
        includes the finite-element diffusion and streamline-stabilization
        reactions. This is a discrete conservation check, distinct from a
        resolution study of physical conductive boundary fluxes.
        """
        storage_vector = np.zeros_like(state)
        if time_step is not None:
            storage_vector = assembly.capacity * (state - previous) / (time_step / self.time_scale)
        load = assembly.load + assembly.mass * control
        reaction = assembly.stiffness @ state + storage_vector - load
        advective_boundary = self.transport_factor * float(
            self.flow.boundary_flux(velocity, scalar=state).sum()
        )
        supply = float(reaction[self.J].sum())
        boundary_absolute_sum = float(np.abs(reaction[self.J]).sum())
        source = float(assembly.mass @ control)
        background = float(assembly.load.sum())
        storage = float(storage_vector.sum())
        defect = storage + advective_boundary - source - background - supply
        scale = max(
            abs(storage)
            + abs(advective_boundary)
            + abs(source)
            + abs(background)
            + boundary_absolute_sum,
            1e-30,
        )
        return {
            "storage": storage,
            "boundary_advection": advective_boundary,
            "fixed_control": source,
            "background_source": background,
            "discrete_dirichlet_supply": supply,
            "boundary_reaction_absolute_sum": boundary_absolute_sum,
            "defect": defect,
            "relative_defect": abs(defect) / scale,
        }
