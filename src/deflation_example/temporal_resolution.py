"""Time-step sensitivity of prescribed-flow optimization and fixed-source replays.

The two targets are selected from desired fields before any optimization.
The mesh, source convention, physical horizon and all final accuracy checks
stay fixed. This study supplies no solver timing comparison or spatial claim.
"""

import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from .benchmark_mesh import build_model, controls
from .mesh_reference import build_mesh_reference
from .mesh_showcases import desired_temperature
from .reporting import environment, write_fields, write_report
from .solvers import independent_residual, relative_norm
from .temperature_bounds import solve_case
from .validation import integer, positive_real


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def settings(config):
    c = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    for key in ("case", "level", "nominal_query"):
        integer(c[key], key)
    for key in (
        "query_population",
        "rank",
        "threads",
        "inner_cap",
        "outer_cap",
        "residual_refresh",
    ):
        integer(c[key], key, 1)
    if c["query_population"] < 2 or c["nominal_query"] >= c["query_population"]:
        raise ValueError("Two distinct targets require a valid population and nominal index")
    for key in (
        "alpha",
        "bound",
        "horizon",
        "rtol",
        "kkt_tolerance",
        "cg_factor",
        "forward_tolerance",
        "state_comparison_tolerance",
        "temperature_change_K",
    ):
        positive_real(c[key], key)
    if c["cg_factor"] > 1 or not isinstance(c["direct_check"], bool):
        raise ValueError("Invalid solver controls")
    if not np.isfinite(c["initial_temperature"]) or c["initial_temperature"] > c["bound"]:
        raise ValueError("The initial temperature must be finite and feasible")
    steps = c["slabs"]
    if len(steps) < 3 or any(integer(n, "Slabs", 1) != n for n in steps):
        raise ValueError("At least three nested time grids are required")
    if any(b != 2 * a for a, b in zip(steps[:-1], steps[1:], strict=True)):
        raise ValueError("Each time grid must double the preceding slab count")
    if c["replay_factors"] != [2, 4]:
        raise ValueError("Fixed-source replays use twice and four times the finest slab count")
    if c["geometry"] not in {"transformer_2d", "engine_3d"}:
        raise ValueError("Unknown thermal geometry")
    if c["transport_form"] not in {"advective", "skew"} or (
        c["transport_form"] == "skew" and c["geometry"] != "transformer_2d"
    ):
        raise ValueError("The skew pilot requires the quadratic transformer velocity")
    if c["action"] not in {"prepare", "run", "report", "stability"}:
        raise ValueError("Choose prepare, run, report or stability")
    return c


def protocol(c):
    return {k: v for k, v in c.items() if k not in {"action", "root", "output", "case"}}


def model(c, slabs):
    return build_model(
        controls(
            OmegaConf.create(
                {
                    **{
                        k: c[k]
                        for k in ("geometry", "level", "alpha", "horizon", "initial_temperature")
                    },
                    "transient": True,
                    "slabs": slabs,
                }
            )
        ),
        transport_form=c["transport_form"],
    )


def prepare(c):
    root = Path(c["root"])
    root.mkdir(parents=True, exist_ok=False)
    showcase, p = model(c, max(c["slabs"]))
    scores = []
    for query in range(c["query_population"]):
        d = desired_temperature(p, query, c["query_population"])
        scores.append(float(p.objective_scale * (p.weights @ np.maximum(d - c["bound"], 0) ** 2)))
    eligible = [q for q in range(c["query_population"]) if q != c["nominal_query"]]
    demanding = max(eligible, key=lambda q: scores[q])
    queries = [c["nominal_query"], demanding]
    result = {
        "schema": "prescribed-flow-temporal-resolution-v1",
        "protocol": protocol(c),
        "protocol_sha256": digest(protocol(c)),
        "environment": environment(),
        "input_sha256": showcase.preparation["input_sha256"],
        "selection": "Nominal index fixed in advance; second target maximizes weighted squared desired-temperature exceedance on the finest declared grid, excluding the nominal target. No optimized outcome enters selection.",
        "desired_exceedance_scores": scores,
        "queries": queries,
        "cases": [{"query": q, "slabs": n} for q in queries for n in c["slabs"]],
        "physical": showcase.parameters["physical"],
        "scope": "Fixed-mesh prescribed-flow temporal sensitivity; no solver speedup or spatial-resolution claim.",
        "transport_form": c["transport_form"],
    }
    write_report(root / "design.json", result)
    return result


def prolong_state(values, initial, fine_slabs):
    """Piecewise-linear state reconstruction through the initial value and time nodes."""
    values, initial = np.asarray(values), np.asarray(initial)
    if values.ndim != 2 or initial.shape != values.shape[1:]:
        raise ValueError("State and initial value dimensions differ")
    n = len(values)
    integer(fine_slabs, "Fine slabs", n)
    if fine_slabs % n:
        raise ValueError("The comparison grid must nest the optimized time nodes")
    ratio = fine_slabs // n
    j = np.arange(1, fine_slabs + 1)
    left = (j - 1) // ratio
    weight = ((j - 1) % ratio + 1) / ratio
    nodes = np.vstack((initial, values))
    return (1 - weight[:, None]) * nodes[left] + weight[:, None] * nodes[left + 1]


def prolong_control(values, fine_slabs):
    """Keep u_n constant on (t_{n-1}, t_n]; fine steps align with every jump."""
    values = np.asarray(values)
    if values.ndim != 2 or not len(values):
        raise ValueError("Controls must have time and space dimensions")
    integer(fine_slabs, "Fine slabs", len(values))
    if fine_slabs % len(values):
        raise ValueError("Control jumps must align with the replay grid")
    return np.repeat(values, fine_slabs // len(values), axis=0)


def weighted_rms(values, mass):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2 @ mass) / np.sum(mass)))


def forward_replay(p, control, slabs):
    """Solve each thermal step independently with one reused sparse factorization."""
    u = prolong_control(control, slabs)
    dt = float(p.steps.sum() / slabs)
    capacity = p.nodal_capacity / dt
    A = p.spatial_A + sparse.diags(capacity)
    solve = splu(A.tocsc()).solve
    previous = p.initial.copy()
    states, residual = [], 0.0
    for un in u:
        b = un + p.spatial_forcing + capacity * previous
        previous = solve(b)
        residual = max(residual, independent_residual(A, previous, b))
        states.append(previous.copy())
    return np.asarray(states), residual


def run_case(c):
    root = Path(c["root"])
    design = json.loads((root / "design.json").read_text())
    if digest(protocol(c)) != design["protocol_sha256"]:
        raise ValueError("Configuration differs from the predeclared temporal design")
    current = environment()
    if current["source_sha256"] != design["environment"]["source_sha256"]:
        raise ValueError("Use the numerical source that selected the targets")
    if c["case"] >= len(design["cases"]):
        raise ValueError("Case index is outside the declared population")
    case = design["cases"][c["case"]]
    output = root / f"case-{c['case']:02d}"
    output.mkdir(exist_ok=False)
    report = {
        **case,
        "status": "running",
        "verified": False,
        "protocol_sha256": design["protocol_sha256"],
        "environment": current,
    }
    write_report(output / "record.json", report)
    try:
        showcase, p = model(c, case["slabs"])
        if showcase.preparation["input_sha256"] != design["input_sha256"]:
            raise ValueError("Thermal inputs changed after target selection")
        d = desired_temperature(p, case["query"], c["query_population"])
        reference = build_mesh_reference(
            p,
            showcase.coarse_assembly,
            showcase.prolongation,
            c["rank"],
            "mode_dependent",
            "scaled_schur",
            "jacobi",
        )
        metrics, fields = solve_case(p, d, c["bound"], c, reference)
        report.update(
            status=(
                "optimization_verification_failed"
                if metrics["status"] == "converged" and not metrics["success"]
                else metrics["status"]
            ),
            optimization=metrics,
            state_dofs=p.size,
            reference=reference.description,
        )
        write_fields(output / "fields.npz", desired=d, **fields)
        report["fields_sha256"] = hashlib.sha256((output / "fields.npz").read_bytes()).hexdigest()
        if metrics["success"]:
            if c["direct_check"] and case["slabs"] == min(c["slabs"]):
                direct, direct_fields = solve_case(p, d, c["bound"], c)
                report["direct_check"] = direct
                report["direct_state_relative_difference"] = relative_norm(
                    fields["state"] - direct_fields["state"], direct_fields["state"]
                )
            control = fields["control"].reshape(case["slabs"], p.spatial_size)
            replay_states, residuals = [], []
            for factor in c["replay_factors"]:
                state, residual = forward_replay(p, control, factor * max(c["slabs"]))
                replay_states.append(state)
                residuals.append(residual)
            physical = design["physical"]
            scale = physical.get("temperature_scale_K", 1.0)
            mass = p.assembly.mass[p.free]
            delta = replay_states[1][1::2] - replay_states[0]
            fine = replay_states[-1]
            violation = np.maximum(fine - c["bound"], 0)
            loc = np.unravel_index(int(violation.argmax()), fine.shape)
            report["replay"] = {
                "slabs": [f * max(c["slabs"]) for f in c["replay_factors"]],
                "maximum_original_residual": max(residuals),
                "temperature_change_max_K": float(np.abs(delta).max() * scale),
                "temperature_change_rms_K": weighted_rms(delta, mass) * scale,
                "maximum_bound_violation_K": float(violation.max() * scale),
                "maximum_bound_violation_time_fraction": (loc[0] + 1) / len(fine),
                "maximum_bound_violation_coordinates": p.assembly.mesh.nodes[
                    p.free[loc[1]]
                ].tolist(),
                "peak_temperature_change_K": float((fine.max() - replay_states[0].max()) * scale),
                "control_representation": "Piecewise constant on original intervals; aligned subdivision; unchanged source values.",
                "resolution_scope": "Differences between two time steps estimate temporal sensitivity; they are not rigorous error bounds.",
            }
            change = report["replay"]["temperature_change_max_K"]
            report["replay"]["temperature_change_within_scale"] = bool(
                np.isfinite(change) and change <= c["temperature_change_K"]
            )
            # These sampled peaks include every replay time level, without clipping.
            write_fields(output / "replays.npz", coarse=replay_states[0], fine=fine)
            direct_ok = "direct_check" not in report or (
                report["direct_check"]["success"]
                and report["direct_state_relative_difference"] <= c["state_comparison_tolerance"]
            )
            report["verified"] = bool(direct_ok and max(residuals) <= c["rtol"])
            report["status"] = "verified" if report["verified"] else "verification_failed"
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__)
        write_report(output / "record.json", report)
        raise
    write_report(output / "record.json", report)
    return report


def summarize(c):
    root = Path(c["root"])
    design = json.loads((root / "design.json").read_text())
    if design["protocol_sha256"] != digest(protocol(c)):
        raise ValueError("Reporting configuration differs from the declared design")
    _, p = model(c, max(c["slabs"]))
    mass = p.assembly.mass[p.free]
    scale = design["physical"].get("temperature_scale_K", 1.0)
    records, arrays = {}, {}
    for index, case in enumerate(design["cases"]):
        path = root / f"case-{index:02d}" / "record.json"
        records[index] = {**case, "status": "missing", "verified": False}
        if path.exists():
            r = json.loads(path.read_text())
            if (
                any(r[k] != v for k, v in case.items())
                or r["protocol_sha256"] != design["protocol_sha256"]
            ):
                raise ValueError("Record differs from the selected case")
            if r["environment"]["source_sha256"] != design["environment"]["source_sha256"]:
                raise ValueError("Mixed numerical sources")
            records[index] = r
            if r["verified"]:
                file = path.with_name("fields.npz")
                if hashlib.sha256(file.read_bytes()).hexdigest() != r["fields_sha256"]:
                    raise ValueError("Saved optimized fields changed")
                if (
                    not r["optimization"]["success"]
                    or max(r["optimization"]["kkt"].values()) > c["kkt_tolerance"]
                ):
                    raise ValueError("A verified row must meet the optimization criteria")
                with np.load(file, allow_pickle=False) as fields:
                    arrays[index] = {k: fields[k] for k in ("state", "control")}
    rows = []
    for index, r in records.items():
        row = {
            "query": r["query"],
            "slabs": r["slabs"],
            "status": r["status"],
            "verified": r["verified"],
        }
        if "optimization" in r:
            row.update(
                {
                    k: r["optimization"][k]
                    for k in (
                        "objective",
                        "active_fraction",
                        "state_maximum",
                        "maximum_bound_violation",
                        "maximum_original_residual",
                        "kkt",
                    )
                }
            )
        if "replay" in r:
            row["replay"] = r["replay"]
        finest = next(
            j
            for j, s in enumerate(design["cases"])
            if s["query"] == r["query"] and s["slabs"] == max(c["slabs"])
        )
        if index in arrays and finest in arrays:
            state = arrays[index]["state"].reshape(r["slabs"], p.spatial_size)
            control = arrays[index]["control"].reshape(r["slabs"], p.spatial_size)
            fine_state = arrays[finest]["state"].reshape(-1, p.spatial_size)
            fine_control = arrays[finest]["control"].reshape(-1, p.spatial_size)
            difference = prolong_state(state, p.initial, len(fine_state)) - fine_state
            row.update(
                temperature_rms_difference_K=weighted_rms(difference, mass) * scale,
                temperature_max_difference_K=float(np.abs(difference).max() * scale),
                control_relative_rms_difference=weighted_rms(
                    prolong_control(control, len(fine_state)) - fine_control, mass
                )
                / max(weighted_rms(fine_control, mass), np.finfo(float).tiny),
                peak_temperature_difference_K=float((state.max() - fine_state.max()) * scale),
                objective_relative_difference=abs(
                    row["objective"] - records[finest]["optimization"]["objective"]
                )
                / max(abs(records[finest]["optimization"]["objective"]), np.finfo(float).tiny),
            )
        rows.append(row)
    result = {
        "design": design,
        "rows": rows,
        "comparison": "Optimized states use piecewise-linear reconstruction through the initial value; controls are piecewise constant. Differences use the finest optimized grid as a numerical reference and physical lumped-mass weights.",
        "all_verified": all(r["verified"] for r in rows),
        "verification_scope": "Discrete residual and optimality checks; temporal resolution is assessed separately.",
        "all_replays_within_temperature_change_scale": all(
            r.get("replay", {}).get("temperature_change_within_scale", False) for r in rows
        ),
    }
    output = Path(c["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "summary.json", result)
    return result


def stability_report(c):
    from .temporal_stability import amplification_mode

    if c["geometry"] != "transformer_2d":
        raise ValueError("The paired transport diagnostic uses the transformer geometry")
    output = Path(c["output"])
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    report = {
        "environment": environment(),
        "protocol": protocol(c),
        "rows": rows,
        "scope": "Thermal forward amplification diagnostics; no optimization timing comparison.",
    }
    for form in ("advective", "skew"):
        showcase, _ = model({**c, "transport_form": form}, min(c["slabs"]))
        seconds = showcase.parameters["physical"]["time_scale_s"]
        report["input_sha256"] = showcase.preparation["input_sha256"]
        for n in [*c["slabs"], *[f * max(c["slabs"]) for f in c["replay_factors"]]]:
            row = {
                "transport_form": form,
                "slabs": n,
                "time_step_seconds": c["horizon"] * seconds / n,
            }
            try:
                row.update(amplification_mode(showcase.assembly, c["horizon"] / n, seconds))
                row["status"] = (
                    "eigenpair_verified"
                    if row["relative_eigenpair_residual"] <= 1e-8
                    else "eigenpair_residual_failed"
                )
            except Exception as error:
                row.update(status="error", error_type=type(error).__name__)
            rows.append(row)
            write_report(output / "stability.json", report)
    return report


@hydra.main(version_base=None, config_path="conf", config_name="temporal_resolution")
def main(config):
    c = settings(config)
    with threadpool_limits(c["threads"]):
        {"prepare": prepare, "run": run_case, "report": summarize, "stability": stability_report}[
            c["action"]
        ](c)


if __name__ == "__main__":
    main()
