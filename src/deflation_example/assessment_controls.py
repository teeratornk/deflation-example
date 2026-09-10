"""Freeze existing optimized source fields for a separate forward assessment.

Selection uses only optimization outputs. The original state equation verifies
the imported fields. This module never optimizes or changes a control.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .mesh_control import build_mesh_control
from .mesh_showcases import build_showcase, desired_temperature
from .reporting import write_report, write_fields
from .benchmark_cht import unpack_mask
from .validation import real_array


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def select_queries(active_fraction, source_mean_square):
    """Three distinct cases, with deterministic ties and no forward outcomes."""
    active = real_array(active_fraction, "Active fractions")
    demand = real_array(source_mean_square, "Source scores")
    if (
        active.shape != (16,)
        or demand.shape != (16,)
        or not np.isfinite(active).all()
        or not np.isfinite(demand).all()
        or np.any((active < 0) | (active > 1))
        or np.any(demand < 0)
    ):
        raise ValueError("Selection requires 16 finite active fractions and source scores")
    remaining = list(range(16))
    result = {}
    for role, score in (
        ("nominal", lambda q: abs((q + 0.25) / 16 - 0.5)),
        ("active", lambda q: -active[q]),
        ("demanding", lambda q: -demand[q]),
    ):
        query = min(remaining, key=lambda q: (score(q), q))
        result[role] = query
        remaining.remove(query)
    return result


def source_at_time(control, slab_ends, time):
    """Evaluate a source on intervals (t[n-1], t[n]], extending the first to t=0."""
    ends = real_array(slab_ends, "Slab endpoints")
    u = real_array(control, "Source history")
    if (
        ends.ndim != 1
        or not len(ends)
        or not np.isfinite(ends).all()
        or ends[0] <= 0
        or np.any(np.diff(ends) <= 0)
        or u.ndim != 2
        or len(u) != len(ends)
        or not np.isfinite(u).all()
        or not np.isfinite(time)
        or time < 0
        or time > ends[-1]
    ):
        raise ValueError("A finite source history and a time inside its horizon are required")
    return u[np.searchsorted(ends, time, side="left")].copy()


def checked_fields(path, problem, query, count, case, bound):
    with np.load(path, allow_pickle=False) as data:
        fields = {k: data[k].copy() for k in ("control", "state", "desired", "active")}
    for key, values in fields.items():
        if (
            values.shape != (problem.size,)
            or values.dtype.kind not in "bif"
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Invalid {key} field for query {query}")
    if fields["active"].dtype.kind != "b":
        raise ValueError("The saved active set must be a Boolean array")
    if int(fields["active"].sum()) != case["active_count"] or not np.array_equal(
        fields["active"], unpack_mask(case["active_mask_bits"], problem.size)
    ):
        raise ValueError("The saved active set differs from the optimization record")
    kkt = case["kkt"]
    required = {"stationarity", "primal", "dual", "complementarity", "projected_gradient"}
    if (
        case["status"] != "converged"
        or set(kkt) != required
        or not np.isfinite(list(kkt.values())).all()
        or min(kkt.values()) < 0
        or max(kkt.values()) > 1e-8
    ):
        raise ValueError("Control selection requires complete optimization with KKT verification")
    for field, name in (("state", "solution_sha256"), ("desired", "target_sha256")):
        if array_digest(fields[field]) != case[name]:
            raise ValueError(f"Saved {field} does not match the optimization record")
    expected = desired_temperature(problem, query, count)
    if not np.allclose(expected, fields["desired"], rtol=1e-13, atol=1e-14):
        raise ValueError("The target does not match the declared analytic population")
    if np.max(fields["state"] - bound) > 1e-8:
        raise ValueError("The saved state violates the original temperature bound")
    # This checks the saved source; it does not construct a replacement source.
    rhs = fields["control"] + problem.forcing
    residual = problem.A @ fields["state"] - rhs
    relative = np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    if not np.isfinite(relative) or relative > 1e-10:
        raise ValueError("Saved source fails the original forward equation")
    return fields, float(relative)


def freeze_population(source_directory, output, protocol, population):
    """Verify all candidate records before writing a new, selected-fields bundle."""
    source_directory, output = Path(source_directory), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory to preserve the frozen selection")
    record_path = source_directory / "record.json"
    record = json.loads(record_path.read_text())
    c = record["controls"]
    transient = population == "transient_four_slabs"
    if population not in protocol["populations"]:
        raise ValueError("Unknown assessment population")
    if (
        record["method"] != protocol["optimization_method"]
        or record["repetition"] != protocol["optimization_repetition"]
        or c["geometry"] != protocol["geometry"]
        or c["level"] != protocol["optimization_level"]
        or c["alpha"] != protocol["optimization_alpha"]
        or c["bound"] != protocol["optimization_bound"]
        or c["transient"] != transient
        or c["targets"] != 16
        or (transient and c["slabs"] != 4)
        or not record["success"]
        or record["environment"]["git_head"] != protocol["optimization_source"]
        or not record["environment"]["source_tree_clean"]
    ):
        raise ValueError("The optimization record differs from the frozen selection protocol")
    showcase = build_showcase(c["geometry"], c["level"])
    if showcase.preparation["input_sha256"] != record["mesh"]["input_sha256"]:
        raise ValueError("The mesh or physical inputs differ from the optimization record")
    if transient and not np.isclose(
        c["horizon"] * showcase.parameters["physical"]["time_scale_s"],
        protocol["optimization_horizon_s"],
        rtol=1e-13,
        atol=0,
    ):
        raise ValueError("The physical horizon differs from the declared assessment")
    steps = np.full(c["slabs"], c["horizon"] / c["slabs"]) if transient else None
    problem = build_mesh_control(
        showcase.assembly,
        alpha=c["alpha"],
        time_steps=steps,
        initial=np.full(len(showcase.assembly.mesh.free), c["initial_temperature"]),
    )
    cases = record["cases"]
    if len(cases) != 16 or [case["query"] for case in cases] != list(range(16)):
        raise ValueError("All 16 original optimization cases must be present in order")
    physical = showcase.parameters["physical"]
    factor = (
        physical["oil_conductivity_W_m_K"]
        * physical["temperature_scale_K"]
        / physical["length_scale_m"] ** 2
    )
    fields, scores = [], []
    for query, case in enumerate(cases):
        field, residual = checked_fields(
            source_directory / f"fields-{query}.npz", problem, query, 16, case, c["bound"]
        )
        fields.append(field)
        scores.append(
            {
                "query_zero_based": query,
                "active_fraction": float(field["active"].mean()),
                "source_mean_square_W2_m6": float(
                    np.average(field["control"] ** 2, weights=problem.weights) * factor**2
                ),
                "original_forward_relative_residual": residual,
            }
        )
    selection = select_queries(
        [s["active_fraction"] for s in scores], [s["source_mean_square_W2_m6"] for s in scores]
    )
    selected = []
    for role, query in selection.items():
        field = fields[query]
        forward = problem.forward(field["control"])
        difference = float(
            np.max(np.abs(forward - field["state"])) * physical["temperature_scale_K"]
        )
        if (
            not np.isfinite(difference)
            or difference > protocol["verification"]["original_forward_max_difference_K"]
        ):
            raise ValueError(
                "Independent forward substitution does not recover the optimized state"
            )
        temperature = (
            physical["inlet_temperature_K"] + physical["temperature_scale_K"] * field["state"]
        )
        selected.append(
            {
                **scores[query],
                "role": role,
                "file": f"{role}.npz",
                "saved_fields_sha256": hashlib.sha256(
                    (source_directory / f"fields-{query}.npz").read_bytes()
                ).hexdigest(),
                "control_array_sha256": array_digest(field["control"]),
                "physical_control_array_sha256": array_digest(field["control"] * factor),
                "original_forward_max_difference_K": difference,
                "original_temperature_range_K": [
                    float(temperature.min()),
                    float(temperature.max()),
                ],
            }
        )
    output.mkdir(parents=True)
    for item in selected:
        field = fields[item["query_zero_based"]]
        write_fields(
            output / item["file"],
            **field,
            control_W_m3=field["control"] * factor,
            initial=problem.initial,
            slab_ends_s=np.cumsum(problem.steps) * physical["time_scale_s"],
        )
        item["bundle_sha256"] = hashlib.sha256((output / item["file"]).read_bytes()).hexdigest()
    report = {
        "schema": protocol["schema"],
        "population": population,
        "status": "controls frozen; coupled forward assessment pending",
        "protocol": protocol,
        "optimization_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "input_sha256": showcase.preparation["input_sha256"],
        "physical": physical,
        "bound_K": physical["inlet_temperature_K"] + physical["temperature_scale_K"] * c["bound"],
        "candidate_scores": scores,
        "selected": selected,
    }
    write_report(output / "selection.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Saved reference-0 directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--population", required=True, choices=["steady", "transient_four_slabs"])
    args = parser.parse_args()
    report = freeze_population(
        args.source, args.output, json.loads(args.protocol.read_text()), args.population
    )
    print(json.dumps({"status": report["status"], "selected": report["selected"]}, indent=2))


if __name__ == "__main__":
    main()
