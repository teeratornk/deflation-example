"""Fixed-control transformer forward assessment with published oil parameters."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from .assessment_controls import array_digest, source_at_time
from .axisymmetric_flow import AxisymmetricFlow, FlowResult
from .coupled_forward import CoupledForward, CoupledResult
from .mesh_showcases import _assembly, desired_temperature
from .mesh_refinement import refine, refine_quadratic_flow
from .meshes import ThermalMesh
from .oil_properties import momentum_reference
from .reporting import environment, write_report, write_fields
from .validation import integer, positive_real

VISCOSITY_TEMPERATURES = {"low": 357.3, "nominal": 348.15, "high": 341.3}


def transformer_boundaries(flow, inlet_speed):
    r, z = flow.points.T
    lower, upper = flow.mesh.nodes.min(axis=0), flow.mesh.nodes.max(axis=0)
    inlet = np.isclose(z, lower[1], rtol=0, atol=1e-11)
    outlet_interior = np.isclose(z, upper[1], rtol=0, atol=1e-11) & (
        (r > lower[0] + 1e-11) & (r < upper[0] - 1e-11)
    )
    fixed = flow.boundary[~outlet_interior[flow.boundary]]
    values = np.zeros((len(fixed), 2))
    values[inlet[fixed], 1] = inlet_speed
    return fixed, values


def import_velocity(flow, cell_values):
    local = cell_values[flow.fluid_cells]
    counts = np.bincount(flow.p2.ravel(), minlength=flow.nv)
    velocity = np.column_stack(
        [
            np.bincount(flow.p2.ravel(), weights=local[:, :, d].ravel(), minlength=flow.nv) / counts
            for d in range(2)
        ]
    )
    if not np.allclose(velocity[flow.p2], local, rtol=0, atol=1e-10):
        raise ValueError("The prescribed P2 velocity has inconsistent shared values")
    return velocity


def checked_baseline(
    directory, flow, fixed, values, properties, input_hashes, level, tolerance, mass_tolerance
):
    """Reuse a computed baseline after checking the current steady equations."""
    directory = Path(directory)
    record_bytes = (directory / "record.json").read_bytes()
    record = json.loads(record_bytes)
    if record.get("baseline_status") != "converged":
        raise ValueError("The requested baseline lacks a converged momentum solution")
    if (
        record["momentum_properties"] != properties
        or record["source_input_sha256"] != input_hashes
        or record["configuration"]["level"] != level
    ):
        raise ValueError("Baseline geometry or material settings differ")
    field = directory / "baseline.npz"
    if hashlib.sha256(field.read_bytes()).hexdigest() != record["baseline_fields_sha256"]:
        raise ValueError("Baseline field checksum differs")
    with np.load(field, allow_pickle=False) as data:
        result = FlowResult(data["velocity"].copy(), data["pressure"].copy(), "converged", [])
    metrics = flow.verify(result, np.zeros_like(flow.quadrature_points), fixed, values)
    flux = flow.boundary_flux(result.velocity)
    balance = float(abs(flux.sum()) / max(np.abs(flux).sum() / 2, 1e-30))
    if (
        not np.isfinite(list(metrics.values())).all()
        or max(metrics.values()) > tolerance
        or balance > mass_tolerance
    ):
        raise ValueError("Cached flow fails the current steady residual or mass check")
    result.history = [
        {"cached_baseline_verification": True, **metrics, "mass_relative_imbalance": balance}
    ]
    return result, {
        "baseline_origin_record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "baseline_construction_seconds": record.get(
            "baseline_construction_seconds", record["baseline_seconds"]
        ),
    }


def checked_restart(directory, flow, fixed, values, properties, input_hashes, level, tolerance):
    """Reverify a saved viscosity-continuation state before resuming it."""
    directory = Path(directory)
    raw = (directory / "baseline-checkpoint.json").read_bytes()
    metadata = json.loads(raw)
    if (
        metadata["momentum_properties"] != properties
        or metadata["input_sha256"] != input_hashes
        or metadata["level"] != level
    ):
        raise ValueError("Continuation checkpoint settings differ")
    stage = metadata["stage"]
    if stage["status"] != "converged" or "viscosity_factor" not in stage:
        raise ValueError("A verified viscosity-continuation checkpoint is required")
    factor = positive_real(stage["viscosity_factor"], "Checkpoint viscosity factor")
    if factor < 1:
        raise ValueError("Checkpoint viscosity lies below the physical value")
    field = directory / metadata.get("file", "baseline-checkpoint.npz")
    if field.parent.resolve() != directory.resolve():
        raise ValueError("Checkpoint fields must lie inside their record directory")
    digest = hashlib.sha256(field.read_bytes()).hexdigest()
    if "field_sha256" in metadata and digest != metadata["field_sha256"]:
        raise ValueError("Continuation checkpoint checksum differs")
    with np.load(field, allow_pickle=False) as data:
        result = FlowResult(data["velocity"].copy(), data["pressure"].copy(), "initial", [])
    physical_viscosity = flow.viscosity
    try:
        flow.viscosity *= factor
        metrics = flow.verify(result, np.zeros_like(flow.quadrature_points), fixed, values)
    finally:
        flow.viscosity = physical_viscosity
    if not np.isfinite(list(metrics.values())).all() or max(metrics.values()) > tolerance:
        raise ValueError("Continuation checkpoint fails its declared steady equations")
    return (
        result,
        factor,
        {
            "restart_record_sha256": hashlib.sha256(raw).hexdigest(),
            "restart_field_sha256": digest,
            "restart_viscosity_factor": factor,
            "preceding_initialization_seconds": metadata.get("elapsed_seconds"),
            "restart_equation_residuals": metrics,
        },
    )


def load_controls(directory, role, level):
    """Authenticate frozen source bundles before any spatial transfer."""
    directory = Path(directory)
    manifest = json.loads((directory / "selection.json").read_text())
    item = next(s for s in manifest["selected"] if s["role"] == role)
    bundle = directory / item["file"]
    if bundle.parent.resolve() != directory.resolve():
        raise ValueError("Control bundle must lie inside its manifest directory")
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != item["bundle_sha256"]:
        raise ValueError("Frozen control bundle checksum differs")
    data_root = Path(__file__).parent / "data" / "transformer_2d"
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in data_root.iterdir()
        if p.is_file()
    }
    if hashes != manifest["input_sha256"]:
        raise ValueError("Assessment mesh inputs differ from the optimization inputs")
    parameters = json.loads((data_root / "parameters.json").read_text())
    mesh = ThermalMesh.load(data_root / "mesh.npz")
    coarse = mesh
    with np.load(data_root / "inputs.npz", allow_pickle=False) as f:
        velocity, background = f["velocity_P2_m_s"], f["source_W_m3"]
    with np.load(bundle, allow_pickle=False) as f:
        control, state = f["control"].copy(), f["state"].copy()
        ends, initial = f["slab_ends_s"].copy(), f["initial"].copy()
        if array_digest(control) != item["control_array_sha256"]:
            raise ValueError("Saved control array checksum differs")
        factor = (
            parameters["physical"]["oil_conductivity_W_m_K"]
            * parameters["physical"]["temperature_scale_K"]
            / parameters["physical"]["length_scale_m"] ** 2
        )
        if array_digest(f["control_W_m3"]) != item[
            "physical_control_array_sha256"
        ] or not np.array_equal(f["control_W_m3"], factor * control):
            raise ValueError("Physical source conversion differs from the frozen record")
    slabs = len(ends) or 1
    if control.shape != (slabs * len(mesh.free),) or initial.shape != (len(mesh.free),):
        raise ValueError("Frozen controls have incompatible dimensions")
    full_control = np.zeros((slabs, len(mesh.nodes)))
    full_control[:, mesh.free] = control.reshape(slabs, -1)
    full_state = np.zeros_like(full_control)
    full_state[:, mesh.free] = state.reshape(slabs, -1)
    full_initial = np.zeros(len(mesh.nodes))
    full_initial[mesh.free] = initial
    coarse_mass = _assembly(mesh, parameters, velocity, background).mass
    original_integrals = full_control @ coarse_mass * factor
    for _ in range(level):
        fine, P, parent = refine(mesh)
        velocity = refine_quadratic_flow(mesh, fine, parent, velocity)
        background = background[parent]
        full_control = (P @ full_control.T).T
        full_state = (P @ full_state.T).T
        full_initial = P @ full_initial
        mesh = fine
    full_control.flags.writeable = False
    return SimpleNamespace(
        mesh=mesh,
        coarse=coarse,
        parameters=parameters,
        velocity=velocity,
        background=background,
        control=full_control,
        state=full_state,
        initial=full_initial,
        ends=ends,
        selection=item,
        manifest=manifest,
        source_integrals_W=original_integrals,
    )


def fixed_flow_step(model, velocity, source, previous, dt, flow_result):
    start = time.perf_counter()
    a, K, rhs = model._thermal_system(velocity, source, previous, dt)
    state = np.zeros(len(a.mesh.nodes))
    state[model.J] = model.boundary
    state[model.I] = spsolve(
        K[model.I][:, model.I], rhs[model.I] - K[model.I][:, model.J] @ model.boundary
    )
    residual = K @ state - rhs
    relative = float(np.linalg.norm(residual[model.I]) / max(np.linalg.norm(rhs[model.I]), 1e-30))
    status = (
        "converged" if np.isfinite(relative) and relative <= 1e-10 else "thermal_residual_failed"
    )
    row = {
        "thermal_relative_residual": relative,
        "energy": model.energy_balance(a, state, velocity, source, previous, dt),
    }
    return CoupledResult(state, flow_result, status, [row], time.perf_counter() - start)


def assessment_times(ends, time_step):
    """Resolve every endpoint of the source intervals (t[n-1], t[n]]."""
    ends = np.asarray(ends, dtype=float)
    time_step = positive_real(time_step, "Assessment time step")
    if ends.ndim != 1 or not np.isfinite(ends).all():
        raise ValueError("Control endpoints must be a finite vector")
    if not len(ends):
        return np.array([0.0]), np.array([1.0])
    if np.any(np.diff(np.r_[0.0, ends]) <= 0):
        raise ValueError("Control endpoints must be positive and strictly increasing")
    ratio = ends[-1] / time_step
    if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-10):
        raise ValueError("Assessment time step must divide the physical horizon")
    times = np.linspace(time_step, ends[-1], round(ratio))
    if not all(np.min(np.abs(times - t)) < 1e-10 for t in ends):
        raise ValueError("Every original control endpoint must be an assessment time level")
    return times, np.diff(np.r_[0.0, times])


def evaluate_tracking(states, desired, controls, mass, steps, temperature_scale, alpha):
    weights = np.asarray(steps)[:, None] * mass[None, :]
    error2 = float(np.sum(weights * (states - desired) ** 2) * temperature_scale**2)
    penalty = float(0.5 * alpha * temperature_scale**2 * np.sum(weights * controls**2))
    return {
        "tracking_weighted_rms_K": float(np.sqrt(error2 / weights.sum())),
        "tracking_integral_half": 0.5 * error2,
        "fixed_control_penalty": penalty,
        "total_objective": 0.5 * error2 + penalty,
        "weights": "axisymmetric spatial mass and right-endpoint time quadrature in seconds; steady time weight one",
    }


def run(config):
    c = OmegaConf.to_container(config, resolve=True)
    for key in ("level",):
        integer(c[key], key)
    for key in ("threads", "coupling_cap", "flow_cap"):
        integer(c[key], key, 1)
    for key in (
        "time_step_s",
        "coupled_tolerance",
        "mass_tolerance",
        "energy_tolerance",
        "relaxation",
    ):
        positive_real(c[key], key)
    if (
        c["mode"] not in {"prescribed", "frozen", "coupled"}
        or c["viscosity"] not in VISCOSITY_TEMPERATURES
    ):
        raise ValueError("Unknown assessment mode or viscosity setting")
    if c["role"] not in {"nominal", "active", "demanding"} or c["controls_directory"] is None:
        raise ValueError("A frozen control directory and declared role are required")
    if c["relaxation"] > 1:
        raise ValueError("Coupling relaxation must not exceed one")
    if c["flow_method"] not in {"picard", "newton"}:
        raise ValueError("Unknown flow iteration")
    if c["baseline_strategy"] not in {"viscosity", "pseudo_time"}:
        raise ValueError("Unknown baseline initialization")
    if any(type(c[key]) is not bool for key in ("baseline_only", "baseline_continuation")):
        raise ValueError("Baseline switches must be Boolean")
    if c["baseline_restart_directory"] is not None and (
        c["baseline_directory"] is not None
        or not c["baseline_continuation"]
        or c["baseline_strategy"] != "viscosity"
    ):
        raise ValueError("Restart requires viscosity continuation and no completed baseline input")
    output = Path(c["output"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _execute(c, output)
    except Exception as exc:
        path = output / "record.json"
        report = json.loads(path.read_text()) if path.exists() else {}
        report.update(status="execution_failed", exception_type=type(exc).__name__)
        write_report(path, report)
        raise


def _execute(c, output):
    start = time.perf_counter()
    inputs = load_controls(c["controls_directory"], c["role"], c["level"])
    times, steps = assessment_times(inputs.ends, c["time_step_s"])
    p = inputs.parameters["physical"]
    published = momentum_reference(VISCOSITY_TEMPERATURES[c["viscosity"]])
    flow = AxisymmetricFlow(
        inputs.mesh, published["kinematic_viscosity_m2_s"], convection_form=c["convection_form"]
    )
    fixed, values = transformer_boundaries(flow, p["inlet_velocity_m_s"])
    prescribed = import_velocity(flow, inputs.velocity)
    model = CoupledForward(
        flow,
        lambda v: _assembly(inputs.mesh, inputs.parameters, v, inputs.background),
        fixed,
        values,
        expansion=published["expansion_coefficient_K_inverse"],
        temperature_offset=p["inlet_temperature_K"],
        temperature_scale=p["temperature_scale_K"],
        time_scale=p["time_scale_s"],
        buoyancy_reference=p["inlet_temperature_K"],
        transport_factor=p["time_scale_s"] / p["length_scale_m"],
        flow_method=c["flow_method"],
    )
    initial_flow = FlowResult(prescribed, np.zeros(flow.np), "prescribed", [])
    report = {
        "schema": "cht-forward-published-oil-v2-pilot",
        "status": "running",
        "configuration": {
            k: v
            for k, v in c.items()
            if k
            not in {
                "output",
                "controls_directory",
                "baseline_directory",
                "baseline_restart_directory",
            }
        },
        "selection": inputs.selection,
        "optimization_population": inputs.manifest["population"],
        "momentum_properties": published,
        "thermal_properties": p,
        "mesh": {
            "nodes": len(inputs.mesh.nodes),
            "cells": len(inputs.mesh.cells),
            "velocity_dofs_per_component": flow.nv,
            "pressure_dofs": flow.np,
        },
        "environment": environment(),
        "steps": [],
        "preparation_seconds": time.perf_counter() - start,
        "scope": "fixed-control forward assessment; separate from optimization comparisons",
        "temperature_sampling": "free P1 nodes at the recorded backward-Euler endpoints; initial temperature is recorded separately",
        "initial_peak_temperature_K": float(
            p["inlet_temperature_K"]
            + p["temperature_scale_K"] * inputs.initial[inputs.mesh.free].max()
        ),
        "bound_K": inputs.manifest["bound_K"],
        "source_time_representation": "piecewise constant on (t[n-1],t[n]]; finer steps retain the original slab value",
        "source_input_sha256": inputs.manifest["input_sha256"],
        "baseline_seconds": 0.0,
    }
    write_report(output / "record.json", report)
    if c["mode"] != "prescribed":
        baseline_start = time.perf_counter()
        baseline_solver = flow.steady_initialization if c["baseline_continuation"] else flow.solve
        if c["baseline_continuation"] and c["baseline_strategy"] == "pseudo_time":
            baseline_solver = flow.pseudo_time_initialization

        def baseline_progress(stages, candidate):
            write_report(output / "baseline-progress.json", {"stages": stages})
            if candidate.status == "converged":
                checkpoint = output / f"baseline-stage-{len(stages)}.npz"
                write_fields(
                    checkpoint,
                    velocity=candidate.velocity,
                    pressure=candidate.pressure,
                )
                write_report(
                    output / "baseline-checkpoint.json",
                    {
                        "stage": stages[-1],
                        "momentum_properties": published,
                        "level": c["level"],
                        "input_sha256": inputs.manifest["input_sha256"],
                        "file": checkpoint.name,
                        "field_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "elapsed_seconds": time.perf_counter() - baseline_start,
                    },
                )

        solver_options = (
            {
                "callback": baseline_progress,
                "iteration_callback": lambda row: write_report(
                    output / "baseline-iteration.json", row
                ),
            }
            if c["baseline_continuation"]
            else {"method": c["flow_method"]}
        )
        if c["baseline_restart_directory"] is not None:
            initial_flow, initial_factor, restart_metadata = checked_restart(
                c["baseline_restart_directory"],
                flow,
                fixed,
                values,
                published,
                inputs.manifest["input_sha256"],
                c["level"],
                c["coupled_tolerance"] * 0.1,
            )
            solver_options["initial_factor"] = initial_factor
            report.update(restart_metadata)
            write_report(output / "record.json", report)
        if c["baseline_directory"] is None:
            initial_flow = baseline_solver(
                np.zeros_like(flow.quadrature_points),
                fixed,
                values,
                initial=initial_flow,
                tolerance=c["coupled_tolerance"] * 0.1,
                max_iterations=c["flow_cap"],
                **solver_options,
            )
        else:
            initial_flow, origin = checked_baseline(
                c["baseline_directory"],
                flow,
                fixed,
                values,
                published,
                inputs.manifest["input_sha256"],
                c["level"],
                c["coupled_tolerance"] * 0.1,
                c["mass_tolerance"],
            )
            report.update(origin)
        report["baseline_seconds"] = time.perf_counter() - baseline_start
        report["baseline_status"] = initial_flow.status
        report["baseline_reused"] = c["baseline_directory"] is not None
        report["baseline_history"] = initial_flow.history
        flux = flow.boundary_flux(initial_flow.velocity)
        report["baseline_mass_relative_imbalance"] = float(
            abs(flux.sum()) / max(np.abs(flux).sum() / 2, 1e-30)
        )
        difference = initial_flow.velocity - prescribed
        report["baseline_velocity_difference_relative_L2"] = float(
            np.sqrt(
                np.sum(difference * (flow.mass @ difference))
                / np.sum(prescribed * (flow.mass @ prescribed))
            )
        )
        write_fields(
            output / "baseline.npz", velocity=initial_flow.velocity, pressure=initial_flow.pressure
        )
        report["baseline_fields_sha256"] = hashlib.sha256(
            (output / "baseline.npz").read_bytes()
        ).hexdigest()
        if (
            initial_flow.status != "converged"
            or report["baseline_mass_relative_imbalance"] > c["mass_tolerance"]
        ):
            report["status"] = "baseline_failed"
            write_report(output / "record.json", report)
            return report
    if c["baseline_only"]:
        report["status"] = "baseline_complete"
        write_report(output / "record.json", report)
        return report
    transient = bool(len(inputs.ends))
    targets = desired_temperature(
        SimpleNamespace(
            assembly=SimpleNamespace(mesh=inputs.mesh),
            free=inputs.mesh.free,
            steps=steps if transient else np.array([]),
        ),
        inputs.selection["query_zero_based"],
        16,
    ).reshape(len(times), -1)
    a0 = model.thermal_builder(flow.thermal_velocity(prescribed))
    conversion = p["oil_conductivity_W_m_K"] * p["temperature_scale_K"] / p["length_scale_m"] ** 2
    report["source_integrals_W"] = (inputs.control @ a0.mass * conversion).tolist()
    report["source_integral_refinement_difference_W"] = (
        inputs.control @ a0.mass * conversion - inputs.source_integrals_W
    ).tolist()
    state, current_flow = inputs.initial.copy(), initial_flow
    if not transient:
        state = inputs.state[0].copy()
    completed_states, applied_controls = [], []
    run_start = time.perf_counter()
    for n, physical_time in enumerate(times):
        source = (
            source_at_time(inputs.control, inputs.ends, physical_time)
            if transient
            else inputs.control[0]
        )
        previous = state.copy() if transient else None
        dt = steps[n] if transient else None
        if c["mode"] == "coupled":

            def progress(row):
                write_report(
                    output / "progress.json",
                    {"time_index": n, "time_s": physical_time, "iteration": row},
                )

            result = model.solve(
                source,
                state,
                current_flow,
                previous_state=previous,
                previous_velocity=current_flow.velocity if transient else None,
                time_step=dt,
                tolerance=c["coupled_tolerance"],
                mass_tolerance=c["mass_tolerance"],
                energy_tolerance=c["energy_tolerance"],
                max_iterations=c["coupling_cap"],
                flow_cap=c["flow_cap"],
                relaxation=c["relaxation"],
                callback=progress,
            )
        else:
            result = fixed_flow_step(
                model, initial_flow.velocity, source, previous, dt, initial_flow
            )
        state, current_flow = result.state, result.flow
        temperature = p["inlet_temperature_K"] + p["temperature_scale_K"] * state
        constrained = inputs.mesh.free
        hottest = constrained[np.argmax(temperature[constrained])]
        raw_violation = float(temperature[hottest] - inputs.manifest["bound_K"])
        in_range = bool(
            np.isfinite(temperature).all()
            and temperature.min() >= 273.15
            and temperature.max() <= 373.15
        )
        row = {
            "time_index": n,
            "time_s": float(physical_time),
            "status": result.status,
            "seconds": result.seconds,
            "history": result.history,
            "peak_temperature_K": float(temperature[hottest]),
            "minimum_temperature_K": float(temperature.min()),
            "bound_violation_K": max(0.0, raw_violation),
            "signed_bound_excess_K": raw_violation,
            "maximum_location_rz_m": inputs.mesh.nodes[hottest].tolist(),
            "published_temperature_domain_satisfied": in_range,
            "sampled_max_velocity_m_s": float(np.linalg.norm(current_flow.velocity, axis=1).max()),
            "buoyancy_density_variation_parameter": float(
                published["expansion_coefficient_K_inverse"]
                * np.max(np.abs(temperature[flow.vertices] - p["inlet_temperature_K"]))
            ),
        }
        report["steps"].append(row)
        write_fields(
            output / f"step-{n}.npz",
            state=state,
            velocity=current_flow.velocity,
            pressure=current_flow.pressure,
            control=source,
        )
        if result.status != "converged" or not in_range:
            report["status"] = (
                result.status
                if result.status != "converged"
                else "property_temperature_domain_exceeded"
            )
            break
        completed_states.append(state[constrained].copy())
        applied_controls.append(source[constrained].copy())
        write_report(output / "record.json", report)
    else:
        report["status"] = "converged"
    report["forward_seconds"] = time.perf_counter() - run_start
    report["complete_seconds"] = time.perf_counter() - start
    report["control_array_sha256_verified"] = inputs.selection["control_array_sha256"]
    report["resolution_status"] = "pending comparison of predeclared mesh and time resolutions"
    if report["status"] == "converged":
        report["metrics"] = evaluate_tracking(
            np.array(completed_states),
            targets,
            np.array(applied_controls),
            a0.mass[inputs.mesh.free],
            steps,
            p["temperature_scale_K"],
            inputs.manifest["protocol"]["optimization_alpha"],
        )
        report["maximum_bound_violation_K"] = max(
            row["bound_violation_K"] for row in report["steps"]
        )
        if c["mode"] == "prescribed" and c["level"] == 0 and len(times) == len(inputs.state):
            report["saved_state_max_difference_K"] = float(
                np.max(np.abs(np.array(completed_states) - inputs.state[:, inputs.mesh.free]))
                * p["temperature_scale_K"]
            )
    write_report(output / "record.json", report)
    return report


@hydra.main(version_base=None, config_path="conf", config_name="assessment")
def main(config):
    with threadpool_limits(limits=config.threads):
        report = run(config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "forward_seconds": report.get("forward_seconds"),
                "metrics": report.get("metrics"),
            },
            indent=2,
        )
    )
    if report["status"] not in {"converged", "baseline_complete"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
