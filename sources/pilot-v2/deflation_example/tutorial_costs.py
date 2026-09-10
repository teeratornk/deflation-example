"""Cost examples preserving methods, timing boundaries, and termination statuses."""

import math
from statistics import median

import numpy as np

from .benchmark_controls import retained_directions
from .benchmark_extended_report import check_partition, checked_sequences, summarize
from .benchmark_report import checked_series


def break_even(additional_setup, saving):
    if not all(math.isfinite(v) for v in (additional_setup, saving)) or additional_setup < 0:
        raise ValueError("Use a finite nonnegative additional setup cost and finite saving")
    return math.ceil(additional_setup / saving) if saving > 0 else None


def resource_costs(data):
    series = checked_series(data)
    rows = []
    for case in data["cases"]:
        for method, selected in case["solvers"].items():
            runs = case["repetitions"].get(method, [selected])
            if method in case["repetitions"] and sorted(r["repetition"] for r in runs) != list(
                range(data["controls"]["repeats"])
            ):
                raise ValueError("Missing resource-control repetition")
            for run in runs:
                if "components_seconds" in run:
                    check_partition(run["components_seconds"], run["total_seconds"])
                if run["status"] == "converged" and not 0 <= run["residual"] <= 1e-10:
                    raise ValueError("Resource control accepted an invalid original residual")
            accepted = [r["total_seconds"] for r in runs if r["status"] == "converged"]
            rows.append(
                {
                    "n": case["n"],
                    "target": case["index"] + 1,
                    "method": method,
                    "attempts": len(runs),
                    "statuses": [r["status"] for r in runs],
                    "accepted_median_seconds": median(accepted) if accepted else None,
                    "repetitions": [
                        {k: r[k] for k in ("status", "total_seconds", "iterations", "residual")}
                        for r in runs
                    ],
                }
            )
    return {
        "configuration": data["controls"],
        "amgx_configuration": data["amgx_configuration"],
        "measurement_environment": data["environment"],
        "timing_scope": data["timing_scope"],
        "aggregation": data["aggregation"],
        "rows": rows,
        "cumulative_sums_of_instance_medians": series,
        "resource_policy": "persistent Config/Resources; matrix, vectors, solver and hierarchy recreated per inactive system",
    }


def rank_costs(data):
    spec, rows = data["specification"], data["rows"]
    if not data["complete"] or [(r["n"], r["rank"]) for r in rows] != [
        (n, k) for n in spec["grids"] for k in spec["ranks"]
    ]:
        raise ValueError("Missing manufactured rank attempt")
    systems, result = {}, []
    components = (
        "construction_seconds",
        "setup_seconds",
        "iteration_seconds",
        "verification_seconds",
        "cleanup_seconds",
    )
    for r in rows:
        identity = tuple(r[k] for k in ("truth_sha256", "rhs_sha256", "inactive_sha256"))
        if systems.setdefault(r["n"], identity) != identity:
            raise ValueError("Rank comparison changes the manufactured system")
        if r["status"] == "converged" and not all(
            0 <= r[k] <= spec["rtol"] for k in ("cpu_original_residual", "gpu_original_residual")
        ):
            raise ValueError("Accepted rank attempt fails its original residual")
        if not 0 <= r["deployed_rank"] <= r["rank"] or (
            r["deployed_rank"] and (r["coarse_condition"] > 1e10 or r["fallback_reason"])
        ):
            raise ValueError("Invalid deployed rank or coarse safeguard")
        remainder = r["total_seconds"] - sum(r[k] for k in components)
        check_partition(
            {**{k: r[k] for k in components}, "remaining_host_work": remainder}, r["total_seconds"]
        )
        result.append(
            {
                **r,
                "remaining_host_work_seconds": remainder,
                "peak_torch_gib": r["peak_torch_bytes"] / 2**30,
            }
        )
    return {
        "configuration": spec,
        "measurement_environment": data["environment"],
        "rows": result,
        "timing_interpretation": "single runs in ascending grid/rank order; first-use costs and problem-size effects are combined",
    }


def history_costs(data):
    rows = []
    for group in data["recycling"]:
        if len(group["sequences"]) != group["repeats"]:
            raise ValueError("Missing history sequence")
        for sequence in group["sequences"]:
            expected = {(i, m) for i in range(group["targets"]) for m in ("reference", "recycling")}
            actual = [(r["index"], r["method"]) for r in sequence["rows"]]
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError("Missing or duplicate history target")
            for index in range(group["targets"]):
                pair = [r for r in sequence["rows"] if r["index"] == index]
                if len({(r["rhs_sha256"], r["inactive_sha256"]) for r in pair}) != 1:
                    raise ValueError("History pair changes the inactive system")
            for method in ("reference", "recycling"):
                runs = [r for r in sequence["rows"] if r["method"] == method]
                for r in runs:
                    if r["status"] == "converged" and not 0 <= r["original_residual"] <= 1e-10:
                        raise ValueError("History solve fails original-residual acceptance")
                reference = (
                    sequence["reference_construction_seconds"] if method == "reference" else 0
                )
                check_partition(
                    {
                        "reference": reference,
                        "kernels_and_transfer": sum(r["total_seconds"] for r in runs),
                    },
                    sequence["cost_sums"][method],
                )
                rows.append(
                    {
                        "family": group["family"],
                        "n": group["n"],
                        "method": method,
                        "repetition": sequence["repetition"],
                        "seconds": sequence["cost_sums"][method],
                        "iterations": sum(r["iterations"] for r in runs),
                        "statuses": [r["status"] for r in runs],
                        "targets": runs,
                    }
                )
    B, coarse = np.diag([1.0, 2.0, 3.0]), np.eye(3)[:, :1]
    directions = [np.eye(3)[:, 1], np.eye(3)[:, 2]]
    selected = retained_directions(B, directions, 1)
    return {
        "configuration": data["specification"]["recycling"],
        "measurement_environment": data["environment"],
        "policy": "only last 100 new search directions; existing coarse vectors excluded; unscaled Ritz",
        "rows": rows,
        "span_example": {
            "matrix": B,
            "current_coarse": coarse,
            "new_directions": directions,
            "selected": selected,
            "coarse_B_inner_products": coarse.T @ B @ np.column_stack(directions),
            "current_coarse_projection_norm": float(np.linalg.norm(selected.T @ coarse)),
        },
    }


def cost_examples(records):
    raw = records.load("extended/cht_v3/results.json")
    if records.load("extended/cht_v3/protocol.json") != raw["specification"]:
        raise ValueError("Stored CHT protocol differs from the experiment specification")
    for record in raw["sequences"]:
        records.load("extended/cht_v3/" + record["path"])
    index, sequences = checked_sequences(
        records.path("extended/cht_v3/results.json"), require_clean_source=True
    )
    if not all(s["success"] for s in sequences):
        raise ValueError("Preparation comparison requires the accepted final protocol")
    return {
        "resource_persistence": resource_costs(records.load("gpu.json")),
        "preparation_inclusive": summarize(index, sequences, records.load("extended/startup.json")),
        "rank_memory": rank_costs(records.load("extended/scale.json")),
        "last_window_recycling": history_costs(records.load("extended/controls.json")),
        "constant_cost_examples": [
            {
                "scope": "illustrative constant-cost model",
                "additional_setup_seconds": 1.2,
                "per_instance_saving_seconds": saving,
                "break_even_count": break_even(1.2, saving),
            }
            for saving in (0.1, 0.0, -0.1)
        ],
    }
