"""Taylor--Hood discretization of axisymmetric flow without swirl.

Velocity has radial and axial components. Pressure is kinematic pressure (m^2/s^2).
The viscous form is nu*(grad(v):grad(w) + v_r*w_r/r^2). The cylindrical
divergence is d_r(v_r) + v_r/r + d_z(v_z). All integrals contain 2*pi*r.
Open boundaries use the natural condition nu*d_n(v) - pressure*n = 0.
The optional skew form adds 0.5*div(v)*v to convection, preserving its
kinetic-energy flux identity with weakly divergence-free velocities.
See Charnyi et al., J. Comput. Phys. 337 (2017), 289--308,
https://doi.org/10.1016/j.jcp.2017.02.039.
Application validation is separate from the numerical tests of these operators.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .meshes import simplex_geometry
from .validation import positive_real, integer


@dataclass
class FlowResult:
    velocity: np.ndarray
    pressure: np.ndarray
    status: str
    history: list


def flow_quadrature():
    """Positive Duffy rule exact through total polynomial degree eight.

    Axisymmetric P2 convection contains degree-six polynomial products.
    The radial viscous hoop term is rational and converges under refinement.
    Weights are normalized to unit sum, for multiplication by triangle area.
    """
    x, w = np.polynomial.legendre.leggauss(5)
    x, w = (x + 1) / 2, w / 2
    a, b = np.meshgrid(x, x, indexing="ij")
    wa, wb = np.meshgrid(w, w, indexing="ij")
    bary = np.column_stack(((1 - a).ravel() * (1 - b).ravel(), a.ravel(), ((1 - a) * b).ravel()))
    return bary, (2 * wa * wb * (1 - a)).ravel()


class AxisymmetricFlow:
    """P2 velocity and P1 pressure on the fluid triangles of a thermal mesh."""

    def __init__(
        self, mesh, viscosity, fluid_material=0, convection_form="advective", grad_div=0.0
    ):
        if mesh.dimension != 2 or not mesh.axisymmetric or np.min(mesh.nodes[:, 0]) <= 0:
            raise ValueError("Flow requires an axisymmetric triangular mesh bounded away from r=0")
        self.mesh = mesh
        self.viscosity = positive_real(viscosity, "Kinematic viscosity")
        if convection_form not in {"advective", "skew"}:
            raise ValueError("Unknown convection form")
        self.convection_form = convection_form
        self.grad_div = float(grad_div)
        if not np.isfinite(self.grad_div) or self.grad_div < 0:
            raise ValueError("Grad-div coefficient must be finite and nonnegative")
        self.fluid_cells = np.flatnonzero(mesh.materials == fluid_material)
        if not len(self.fluid_cells):
            raise ValueError("At least one fluid cell is required")
        cells = mesh.cells[self.fluid_cells]
        self.vertices, local = np.unique(cells, return_inverse=True)
        self.p1 = local.reshape(-1, 3)
        edges = np.sort(cells[:, [(0, 1), (0, 2), (1, 2)]], axis=2)
        self.edges, inverse, counts = np.unique(
            edges.reshape(-1, 2), axis=0, return_inverse=True, return_counts=True
        )
        self.p2 = np.column_stack((self.p1, len(self.vertices) + inverse.reshape(-1, 3)))
        self.points = np.vstack((mesh.nodes[self.vertices], mesh.nodes[self.edges].mean(axis=1)))
        self.boundary_edges = np.flatnonzero(counts == 1)
        boundary_vertices = np.searchsorted(
            self.vertices, np.unique(self.edges[self.boundary_edges])
        )
        self.boundary = np.unique(
            np.r_[boundary_vertices, len(self.vertices) + self.boundary_edges]
        )
        self.nv, self.np = len(self.points), len(self.vertices)
        self.size = 2 * self.nv + self.np
        bary, weights = flow_quadrature()
        self.bary = bary
        self.shape = np.column_stack(
            (
                bary * (2 * bary - 1),
                4 * bary[:, 0] * bary[:, 1],
                4 * bary[:, 0] * bary[:, 2],
                4 * bary[:, 1] * bary[:, 2],
            )
        )
        grad, _ = simplex_geometry(mesh)
        grad = grad[self.fluid_cells]
        vertices = mesh.nodes[cells]
        area = np.linalg.det(vertices[:, 1:] - vertices[:, :1]) / 2
        self.quadrature_points = np.einsum("qi,eid->eqd", bary, vertices)
        self.radius = self.quadrature_points[:, :, 0]
        self.measure = area[:, None] * weights[None, :] * 2 * np.pi * self.radius
        self.grad = np.concatenate(
            (
                (4 * bary[None, :, :, None] - 1) * grad[:, None, :, :],
                np.stack(
                    [
                        4
                        * (
                            bary[None, :, a, None] * grad[:, None, b, :]
                            + bary[None, :, b, None] * grad[:, None, a, :]
                        )
                        for a, b in ((0, 1), (0, 2), (1, 2))
                    ],
                    axis=2,
                ),
            ),
            axis=2,
        )
        self.div_basis = self.grad.copy()
        self.div_basis[:, :, :, 0] += self.shape[None, :, :] / self.radius[:, :, None]
        local_mass = np.einsum("eq,qi,qj->eij", self.measure, self.shape, self.shape)
        local_diff = np.einsum("eq,eqid,eqjd->eij", self.measure, self.grad, self.grad)
        local_hoop = np.einsum(
            "eq,qi,qj->eij", self.measure / self.radius**2, self.shape, self.shape
        )
        self.mass = self._matrix(local_mass, self.p2, self.p2, (self.nv, self.nv))
        self.diffusion = self._matrix(local_diff, self.p2, self.p2, (self.nv, self.nv))
        self.hoop = self._matrix(local_hoop, self.p2, self.p2, (self.nv, self.nv))
        radial = self.grad[:, :, :, 0] + self.shape[None, :, :] / self.radius[:, :, None]
        self.divergence = []
        for gradient in (radial, self.grad[:, :, :, 1]):
            local_div = np.einsum("eq,qi,eqj->eij", self.measure, bary, gradient)
            self.divergence.append(self._matrix(local_div, self.p1, self.p2, (self.np, self.nv)))
        self.grad_div_operator = None
        if self.grad_div:
            blocks = []
            for a in range(2):
                blocks.append([])
                for b in range(2):
                    local = np.einsum(
                        "eq,eqi,eqj->eij",
                        self.measure,
                        self.div_basis[:, :, :, a],
                        self.div_basis[:, :, :, b],
                    )
                    blocks[-1].append(self._matrix(local, self.p2, self.p2, (self.nv, self.nv)))
                blocks[-1].append(sparse.csr_matrix((self.nv, self.np)))
            blocks.append(
                [
                    sparse.csr_matrix((self.np, self.nv)),
                    sparse.csr_matrix((self.np, self.nv)),
                    sparse.csr_matrix((self.np, self.np)),
                ]
            )
            self.grad_div_operator = sparse.bmat(blocks, format="csr")

    @staticmethod
    def _matrix(values, rows, columns, shape):
        i = np.repeat(rows, columns.shape[1], axis=1).ravel()
        j = np.tile(columns, (1, rows.shape[1])).ravel()
        return sparse.coo_matrix((values.ravel(), (i, j)), shape=shape).tocsr()

    def sampled_velocity(self, velocity):
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (self.nv, 2) or not np.isfinite(velocity).all():
            raise ValueError("Velocity must contain finite P2 radial and axial components")
        return np.einsum("qi,eid->eqd", self.shape, velocity[self.p2])

    def convection(self, velocity):
        samples = self.sampled_velocity(velocity)
        local = np.einsum("eq,qi,eqd,eqjd->eij", self.measure, self.shape, samples, self.grad)
        if self.convection_form == "skew":
            gradient = np.einsum("eia,eqib->eqab", velocity[self.p2], self.grad)
            divergence = (
                gradient[:, :, 0, 0] + gradient[:, :, 1, 1] + samples[:, :, 0] / self.radius
            )
            local += 0.5 * np.einsum(
                "eq,qi,qj,eq->eij", self.measure, self.shape, self.shape, divergence
            )
        return self._matrix(local, self.p2, self.p2, (self.nv, self.nv))

    def convection_derivative(self, velocity):
        """Jacobian term (delta v . grad) v in the momentum equation."""
        self.sampled_velocity(velocity)
        gradient = np.einsum("eia,eqib->eqab", velocity[self.p2], self.grad)
        blocks = []
        for a in range(2):
            row = []
            for b in range(2):
                local = np.einsum(
                    "eq,qi,qj,eq->eij", self.measure, self.shape, self.shape, gradient[:, :, a, b]
                )
                if self.convection_form == "skew":
                    divergence_basis = self.div_basis[:, :, :, b]
                    sample_a = np.einsum("qi,ei->eq", self.shape, velocity[self.p2, a])
                    local += 0.5 * np.einsum(
                        "eq,qi,eqj,eq->eij", self.measure, self.shape, divergence_basis, sample_a
                    )
                row.append(self._matrix(local, self.p2, self.p2, (self.nv, self.nv)))
            row.append(sparse.csr_matrix((self.nv, self.np)))
            blocks.append(row)
        blocks.append(
            [
                sparse.csr_matrix((self.np, self.nv)),
                sparse.csr_matrix((self.np, self.nv)),
                sparse.csr_matrix((self.np, self.np)),
            ]
        )
        return sparse.bmat(blocks, format="csr")

    def nonlinear_force(self, velocity):
        """Integrate the quadratic momentum term without assembling a matrix."""
        samples = self.sampled_velocity(velocity)
        gradient = np.einsum("eia,eqib->eqab", velocity[self.p2], self.grad)
        force = np.einsum("eqb,eqab->eqa", samples, gradient)
        if self.convection_form == "skew":
            divergence = (
                gradient[:, :, 0, 0] + gradient[:, :, 1, 1] + samples[:, :, 0] / self.radius
            )
            force += 0.5 * divergence[:, :, None] * samples
        return self.load(force)

    def load(self, acceleration):
        """Integrate supplied acceleration samples at the declared quadrature points."""
        force = np.asarray(acceleration, dtype=float)
        if force.shape != self.quadrature_points.shape or not np.isfinite(force).all():
            raise ValueError("Acceleration samples must match the fluid quadrature points")
        local = np.einsum("eq,qi,eqd->eid", self.measure, self.shape, force)
        return np.column_stack(
            [
                np.bincount(self.p2.ravel(), weights=local[:, :, d].ravel(), minlength=self.nv)
                for d in range(2)
            ]
        )

    def buoyancy(self, temperature, reference_temperature, expansion, gravity=(0.0, -9.81)):
        """Boussinesq acceleration -beta*(T-Tref)*g after hydrostatic lifting."""
        temperature = np.asarray(temperature, dtype=float)
        gravity = np.asarray(gravity, dtype=float)
        if (
            temperature.shape != (len(self.mesh.nodes),)
            or not np.isfinite(temperature).all()
            or not np.isfinite(reference_temperature)
            or gravity.shape != (2,)
            or not np.isfinite(gravity).all()
            or not np.isfinite(expansion)
            or expansion < 0
        ):
            raise ValueError(
                "Finite temperatures, nonnegative expansion and a gravity vector are required"
            )
        samples = temperature[self.mesh.cells[self.fluid_cells]] @ self.bary.T
        return -expansion * (samples - reference_temperature)[:, :, None] * gravity

    def thermal_velocity(self, velocity):
        self.sampled_velocity(velocity)
        result = np.zeros((len(self.mesh.cells), 6, 2))
        result[self.fluid_cells] = velocity[self.p2]
        return result

    def operator(self, velocity, time_step=None, convection=True):
        K = self.viscosity * self.diffusion
        if convection:
            K = K + self.convection(velocity)
        if time_step is not None:
            K = K + self.mass / positive_real(time_step, "Physical time step")
        Br, Bz = self.divergence
        result = sparse.bmat(
            [
                [K + self.viscosity * self.hoop, None, -Br.T],
                [None, K, -Bz.T],
                [Br, Bz, None],
            ],
            format="csr",
        )
        if self.grad_div_operator is not None:
            result += self.grad_div * self.grad_div_operator
        return result

    def solve(
        self,
        acceleration,
        boundary_indices,
        boundary_values,
        *,
        initial=None,
        previous=None,
        time_step=None,
        pressure_gauge=None,
        convection=True,
        tolerance=1e-9,
        max_iterations=100,
        relaxation=1.0,
        method="picard",
        callback=None,
    ):
        """Picard or damped Newton iteration with independent equation checks.

        The pressure gauge is (P1 index, kinematic pressure); use it for a fully
        Dirichlet velocity problem. An open natural boundary determines pressure.
        Return every iteration and an explicit termination status.
        """
        tolerance = positive_real(tolerance, "Flow tolerance")
        max_iterations = integer(max_iterations, "Flow iteration cap", 1)
        relaxation = positive_real(relaxation, "Flow relaxation")
        if relaxation > 1:
            raise ValueError("Flow relaxation must not exceed one")
        if method not in {"picard", "newton"}:
            raise ValueError("Unknown flow iteration")
        fixed = np.asarray(boundary_indices)
        values = np.asarray(boundary_values, dtype=float)
        if (
            fixed.ndim != 1
            or fixed.dtype.kind not in "iu"
            or not len(fixed)
            or fixed.min() < 0
            or fixed.max() >= self.nv
            or len(np.unique(fixed)) != len(fixed)
            or values.shape != (len(fixed), 2)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Distinct velocity boundary indices and matching values are required")
        constrained = np.r_[fixed, self.nv + fixed]
        prescribed = np.r_[values[:, 0], values[:, 1]]
        if pressure_gauge is not None:
            index, value = pressure_gauge
            integer(index, "Pressure gauge index")
            if index >= self.np or not np.isfinite(value):
                raise ValueError("Invalid pressure gauge")
            constrained = np.r_[constrained, 2 * self.nv + index]
            prescribed = np.r_[prescribed, value]
        free = np.setdiff1d(np.arange(self.size), constrained)
        if initial is None:
            x = np.zeros(self.size)
        else:
            if initial.velocity.shape != (self.nv, 2) or initial.pressure.shape != (self.np,):
                raise ValueError("Initial flow dimensions differ from this discretization")
            x = np.r_[initial.velocity[:, 0], initial.velocity[:, 1], initial.pressure].copy()
            if not np.isfinite(x).all():
                raise ValueError("Initial flow must be finite")
        x[constrained] = prescribed
        force = self.load(acceleration)
        if (time_step is None) != (previous is None):
            raise ValueError("Transient momentum requires both a time step and previous velocity")
        if previous is not None:
            self.sampled_velocity(previous)
            force += (self.mass @ previous) / positive_real(time_step, "Physical time step")
        rhs = np.r_[force[:, 0], force[:, 1], np.zeros(self.np)]
        history, status = [], "iteration_cap"
        for iteration in range(1, max_iterations + 1):
            velocity = np.column_stack((x[: self.nv], x[self.nv : 2 * self.nv]))
            A = self.operator(velocity, time_step, convection)
            if method == "newton" and convection:
                linear_operator = A + self.convection_derivative(velocity)
                linear_rhs = (rhs - A @ x)[free]
            else:
                linear_operator = A
                linear_rhs = rhs[free] - A[free][:, constrained] @ prescribed
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    proposal = spsolve(linear_operator[free][:, free], linear_rhs)
            except (MatrixRankWarning, RuntimeError):
                status = "linear_failure"
                break
            if not np.isfinite(proposal).all():
                status = "nonfinite_iterate"
                break
            linear_residual = float(
                np.linalg.norm(linear_operator[free][:, free] @ proposal - linear_rhs)
                / max(np.linalg.norm(linear_rhs), 1e-30)
            )
            damping = relaxation
            if method == "newton" and convection:
                original = x.copy()
                original_residual = (A @ x - rhs)[free]
                original_norm = np.linalg.norm(original_residual)
                full_step = np.zeros_like(x)
                full_step[free] = proposal
                linear_change = (linear_operator @ full_step)[free]
                step_velocity = np.column_stack(
                    (full_step[: self.nv], full_step[self.nv : 2 * self.nv])
                )
                nonlinear = self.nonlinear_force(step_velocity)
                quadratic_change = np.r_[nonlinear[:, 0], nonlinear[:, 1], np.zeros(self.np)][free]
                for backtrack in range(24):
                    x[free] = original[free] + damping * proposal
                    # The Navier-Stokes residual is quadratic in velocity.
                    # Reassemble it independently after choosing the step.
                    trial_norm = np.linalg.norm(
                        original_residual
                        + damping * linear_change
                        + damping * damping * quadratic_change
                    )
                    if (
                        np.isfinite(trial_norm)
                        and trial_norm <= (1 - 1e-4 * damping) * original_norm
                    ):
                        break
                    damping *= 0.5
                else:
                    x = original
                    metrics = self._residual_metrics(A, x, rhs, constrained, prescribed)
                    history.append(
                        {
                            "iteration": iteration,
                            "method": method,
                            "damping": 0.0,
                            "linear_relative_residual": linear_residual,
                            "newton_step_norm": float(np.linalg.norm(proposal)),
                            **metrics,
                        }
                    )
                    if callback is not None:
                        callback(history[-1])
                    status = (
                        "converged" if max(metrics.values()) <= tolerance else "line_search_failed"
                    )
                    break
            else:
                x[free] = damping * proposal + (1 - damping) * x[free]
            velocity = np.column_stack((x[: self.nv], x[self.nv : 2 * self.nv]))
            # Rebuild convection at the updated velocity; the lagged linear
            # residual alone cannot establish a nonlinear solution.
            updated = self.operator(velocity, time_step, convection)
            metrics = self._residual_metrics(updated, x, rhs, constrained, prescribed)
            history.append(
                {
                    "iteration": iteration,
                    "method": method,
                    "damping": damping,
                    "linear_relative_residual": linear_residual,
                    **metrics,
                }
            )
            if callback is not None:
                callback(history[-1])
            if max(metrics.values()) <= tolerance:
                status = "converged"
                break
            if method == "newton" and len(history) >= 6:
                window = history[-6:]
                values_window = [h["momentum_relative_residual"] for h in window]
                if max(h["damping"] for h in window) < 1e-4 and min(values_window) >= 0.99 * max(
                    values_window
                ):
                    status = "stagnation"
                    break
        return FlowResult(
            np.column_stack((x[: self.nv], x[self.nv : 2 * self.nv])),
            x[2 * self.nv :].copy(),
            status,
            history,
        )

    def _residual_metrics(self, A, x, rhs, constrained, prescribed):
        residual = A @ x - rhs
        momentum = np.setdiff1d(np.arange(2 * self.nv), constrained)
        scale = np.linalg.norm(rhs[momentum]) + np.linalg.norm(
            A[momentum][:, constrained] @ prescribed
        )
        vr, vz = x[: self.nv], x[self.nv : 2 * self.nv]
        Br, Bz = self.divergence
        divergence = Br @ vr + Bz @ vz
        div_scale = max(
            np.linalg.norm(Br @ vr) + np.linalg.norm(Bz @ vz),
            np.max(np.abs(x[: 2 * self.nv])) * self.measure.sum() ** (2 / 3),
            1e-30,
        )
        return {
            "momentum_relative_residual": float(
                np.linalg.norm(residual[momentum]) / max(scale, 1e-30)
            ),
            "continuity_relative_residual": float(np.linalg.norm(divergence) / div_scale),
        }

    def steady_initialization(
        self,
        acceleration,
        boundary_indices,
        boundary_values,
        *,
        initial=None,
        tolerance=1e-9,
        max_iterations=100,
        callback=None,
        initial_factor=64.0,
        iteration_callback=None,
    ):
        """Viscosity continuation ending at the unchanged physical viscosity.

        Intermediate equations initialize Newton's method. Only the last solve
        uses the assessment viscosity. Failure at any stage remains explicit.
        """
        physical_viscosity = self.viscosity
        stages = []
        result = initial
        initial_factor = positive_real(initial_factor, "Initial viscosity factor")
        if initial_factor < 1:
            raise ValueError("Viscosity continuation must start at or above the physical value")
        factor, last_factor, log_step = initial_factor, None, np.log(2.0)
        status = "continuation_stage_cap"
        try:
            for _ in range(80):
                self.viscosity = physical_viscosity * factor
                candidate = self.solve(
                    acceleration,
                    boundary_indices,
                    boundary_values,
                    initial=result,
                    tolerance=tolerance,
                    max_iterations=min(max_iterations, 30),
                    method="newton",
                    callback=(lambda row: iteration_callback({"viscosity_factor": factor, **row}))
                    if iteration_callback is not None
                    else None,
                )
                stages.append(
                    {
                        "viscosity_factor": factor,
                        "status": candidate.status,
                        "history": candidate.history,
                    }
                )
                if callback is not None:
                    callback(stages, candidate)
                if candidate.status != "converged":
                    log_step *= 0.5
                    if last_factor is None or log_step < 1e-3:
                        result = candidate
                        status = "continuation_" + candidate.status
                        break
                    factor = max(1.0, last_factor * np.exp(-log_step))
                    continue
                result, last_factor = candidate, factor
                if factor == 1.0:
                    status = "converged"
                    break
                log_step = min(np.log(2.0), log_step * 1.25)
                factor = max(1.0, factor * np.exp(-log_step))
        finally:
            self.viscosity = physical_viscosity
        return FlowResult(result.velocity, result.pressure, status, stages)

    def pseudo_time_initialization(
        self,
        acceleration,
        boundary_indices,
        boundary_values,
        *,
        initial=None,
        tolerance=1e-9,
        max_iterations=100,
        callback=None,
        iteration_callback=None,
    ):
        """Pseudo-transient initialization with a separate steady residual test.

        Artificial time steps initialize the steady equations at the physical
        viscosity. Their convergence alone never accepts a steady baseline.
        """
        if initial is None:
            initial = FlowResult(np.zeros((self.nv, 2)), np.zeros(self.np), "initial", [])
        result = initial
        dt, stages, status = 0.001, [], "pseudo_time_step_cap"
        for _ in range(200):
            candidate = self.solve(
                acceleration,
                boundary_indices,
                boundary_values,
                initial=result,
                previous=result.velocity,
                time_step=dt,
                tolerance=tolerance * 0.1,
                max_iterations=min(max_iterations, 12),
                method="newton",
                callback=(lambda row: iteration_callback({"pseudo_time_step_s": dt, **row}))
                if iteration_callback is not None
                else None,
            )
            steady = self.verify(candidate, acceleration, boundary_indices, boundary_values)
            norm = max(steady.values())
            stages.append(
                {
                    "pseudo_time_step_s": dt,
                    "status": candidate.status,
                    "steady_residuals": steady,
                    "history": candidate.history,
                }
            )
            if callback is not None:
                callback(stages, candidate)
            if candidate.status != "converged":
                dt *= 0.25
                if dt < 1e-6:
                    status = "pseudo_time_" + candidate.status
                    break
                continue
            result = candidate
            if norm <= tolerance:
                status = "converged"
                break
            # Temporal evolution can increase the steady residual temporarily.
            # Use nonlinear work to choose the next pseudo-time increment.
            factor = (
                1.5 if len(candidate.history) <= 5 else 1.2 if len(candidate.history) <= 9 else 1.0
            )
            dt = min(1e4, dt * factor)
        return FlowResult(result.velocity, result.pressure, status, stages)

    def verify(
        self,
        result,
        acceleration,
        boundary_indices,
        boundary_values,
        *,
        previous=None,
        time_step=None,
        pressure_gauge=None,
        convection=True,
    ):
        """Reassemble the complete residual for externally updated thermal forcing."""
        self.sampled_velocity(result.velocity)
        if result.pressure.shape != (self.np,) or not np.isfinite(result.pressure).all():
            raise ValueError("Pressure must match the P1 space")
        fixed = np.asarray(boundary_indices)
        values = np.asarray(boundary_values, dtype=float)
        if (
            fixed.ndim != 1
            or fixed.dtype.kind not in "iu"
            or not len(fixed)
            or fixed.min() < 0
            or fixed.max() >= self.nv
            or len(np.unique(fixed)) != len(fixed)
            or values.shape != (len(fixed), 2)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Invalid velocity boundary data")
        constrained = np.r_[fixed, self.nv + fixed]
        prescribed = np.r_[values[:, 0], values[:, 1]]
        if pressure_gauge is not None:
            index, value = pressure_gauge
            integer(index, "Pressure gauge index")
            if index >= self.np or not np.isfinite(value):
                raise ValueError("Invalid pressure gauge")
            constrained = np.r_[constrained, 2 * self.nv + index]
            prescribed = np.r_[prescribed, value]
        x = np.r_[result.velocity[:, 0], result.velocity[:, 1], result.pressure]
        if not np.allclose(x[constrained], prescribed, rtol=0, atol=1e-13):
            raise ValueError("Flow violates prescribed boundary data")
        force = self.load(acceleration)
        if (time_step is None) != (previous is None):
            raise ValueError("Transient verification requires previous velocity and a time step")
        if previous is not None:
            self.sampled_velocity(previous)
            force += (self.mass @ previous) / positive_real(time_step, "Physical time step")
        rhs = np.r_[force[:, 0], force[:, 1], np.zeros(self.np)]
        A = self.operator(result.velocity, time_step, convection)
        return self._residual_metrics(A, x, rhs, constrained, prescribed)

    def boundary_flux(self, velocity, scalar=None):
        """Integrate outward volume flux, optionally weighted by a P1 scalar.

        Three-point Gauss quadrature integrates r*v_P2*scalar_P1 exactly.
        """
        edges, t, flux, _ = self._boundary_samples(velocity, 3)
        scalar_samples = np.ones_like(flux)
        if scalar is not None:
            scalar = np.asarray(scalar, dtype=float)
            if scalar.shape != (len(self.mesh.nodes),) or not np.isfinite(scalar).all():
                raise ValueError("Flux weighting requires a finite P1 nodal scalar")
            scalar_samples = scalar[edges[:, :1]] * (1 - t) + scalar[edges[:, 1:]] * t
        return np.sum(flux * scalar_samples, axis=1)

    def kinetic_diagnostics(self, velocity):
        """Measure the kinetic-energy term introduced by pointwise divergence."""
        samples = self.sampled_velocity(velocity)
        gradient = np.einsum("eia,eqib->eqab", velocity[self.p2], self.grad)
        divergence = gradient[:, :, 0, 0] + gradient[:, :, 1, 1] + samples[:, :, 0] / self.radius
        squared = np.sum(samples**2, axis=2)
        defect = 0.5 * float(np.sum(self.measure * divergence * squared))
        convection = float(np.sum(velocity * (self.convection(velocity) @ velocity)))
        _, _, flux, boundary = self._boundary_samples(velocity, 4)
        kinetic_flux = 0.5 * float(np.sum(flux * np.sum(boundary**2, axis=2)))
        viscous = self.viscosity * float(
            np.sum(velocity * (self.diffusion @ velocity))
            + velocity[:, 0] @ (self.hoop @ velocity[:, 0])
        )
        gradient_squared = float(np.sum(self.measure * np.sum(gradient**2, axis=(2, 3))))
        return {
            "convection_energy_rate": convection,
            "kinetic_boundary_flux": kinetic_flux,
            "divergence_energy_term": defect,
            "viscous_dissipation": viscous,
            "grad_div_dissipation": self.grad_div * float(np.sum(self.measure * divergence**2)),
            "integration_identity_defect": convection
            + (defect if self.convection_form == "advective" else 0.0)
            - kinetic_flux,
            "convection_form": self.convection_form,
            "pointwise_divergence_relative_L2": float(
                np.sqrt(np.sum(self.measure * divergence**2) / max(gradient_squared, 1e-30))
            ),
            "divergence_energy_over_viscous": abs(defect) / max(abs(viscous), 1e-30),
            "kinetic_energy": 0.5 * float(np.sum(self.measure * squared)),
            "sampled_max_speed_m_s": float(np.linalg.norm(velocity, axis=1).max()),
        }

    def _boundary_samples(self, velocity, order):
        self.sampled_velocity(velocity)
        edge_ids = self.boundary_edges
        edges = self.edges[edge_ids]
        x = self.mesh.nodes[edges]
        # Recover each boundary edge's adjacent cell to orient its normal.
        cell_edges = self.p2[:, 3:] - len(self.vertices)
        lookup = np.empty(len(self.edges), dtype=int)
        lookup[cell_edges.ravel()] = np.repeat(np.arange(len(self.p2)), 3)
        center = self.mesh.nodes[self.mesh.cells[self.fluid_cells[lookup[edge_ids]]]].mean(axis=1)
        tangent = x[:, 1] - x[:, 0]
        length = np.linalg.norm(tangent, axis=1)
        normal = np.column_stack((tangent[:, 1], -tangent[:, 0])) / length[:, None]
        inward = np.einsum("ed,ed->e", center - x.mean(axis=1), normal) > 0
        normal[inward] *= -1
        ends = np.searchsorted(self.vertices, edges)
        nodal = np.stack(
            (velocity[ends[:, 0]], velocity[len(self.vertices) + edge_ids], velocity[ends[:, 1]]),
            axis=1,
        )
        t, weights = np.polynomial.legendre.leggauss(order)
        t, weights = (t + 1) / 2, weights / 2
        basis = np.column_stack(((1 - t) * (1 - 2 * t), 4 * t * (1 - t), t * (2 * t - 1)))
        samples = np.einsum("qi,eid->eqd", basis, nodal)
        radii = x[:, :1, 0] * (1 - t) + x[:, 1:, 0] * t
        flux = (
            length[:, None]
            * 2
            * np.pi
            * radii
            * np.einsum("eqd,ed,q->eq", samples, normal, weights)
        )
        return edges, t, flux, samples
