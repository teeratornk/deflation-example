"""The thermal stabilisation correction, read from its own records.

Three measurements, each from a run rather than from a test name:

* what the declared symmetric term costs in accuracy, and what weighting the whole
  residual buys back, on a manufactured field where the unstabilised operator is
  exact to roundoff;
* what the weighting does to the structure of the operators it touches, with and
  without the bound that keeps a cell from taking more out of a node's row than that
  node's share of the cell mass;
* what the forward solve costs on each model, in Newton iterations per time slab.

Nothing here reads a run past what it recorded. A replay that stopped early reports
where it stopped and contributes no iteration count, because a stalled solve's last
step is not a cost.
"""

import json
from pathlib import Path

import numpy as np

CONVERGENCE_SCHEMA = "coupled-stabilisation-convergence-v1"
CONDITIONING_SCHEMA = "coupled-consistent-storage-conditioning-v1"
REPLAY_SCHEMA = "coupled-fixed-source-newton-replay-v1"
# Below this the manufactured error is roundoff on this field and has no order.
ROUNDOFF_K = 1e-10


def _read(directory, schema):
    record = json.loads((Path(directory) / "record.json").read_text())
    if record.get("schema") != schema:
        raise ValueError(f"{directory} is not a {schema} record")
    return record


def convergence(directory):
    """Observed order of each stabilisation on the manufactured field."""
    record = _read(directory, CONVERGENCE_SCHEMA)
    cases = {}
    for row in record["cases"]:
        orders = row["observed_order"]
        # An operator that is exact on this field has no order to report: its ratios
        # are ratios of roundoff. Say that instead of printing a number for it.
        at_roundoff = max(row["maximum_errors_K"]) < ROUNDOFF_K
        cases[row["case"]] = {
            "divisions": row["divisions"],
            "maximum_errors_K": row["maximum_errors_K"],
            "error_ratios": row["error_ratios"],
            "observed_order": None if at_roundoff else orders,
            "minimum_observed_order": None if at_roundoff else float(min(orders)),
            "at_roundoff": at_roundoff,
            "finest_error_K": float(row["maximum_errors_K"][-1]),
            "row_bound": row.get("row_bound"),
        }
    missing = {"unstabilised", "declared_symmetric_term", "residual_weighted"} - set(cases)
    if missing:
        raise ValueError(f"The convergence record is missing {sorted(missing)}")
    shipped = cases["declared_symmetric_term"]
    weighted = cases["residual_weighted"]
    if not cases["unstabilised"]["at_roundoff"]:
        raise ValueError(
            "The unstabilised operator is not exact on this field, so the comparison does "
            "not isolate the artificial term"
        )
    if shipped["at_roundoff"] or weighted["at_roundoff"]:
        raise ValueError("A stabilised case at roundoff would make this comparison vacuous")
    return {
        "directory": str(directory),
        "speed_m_s": record["speed_m_s"],
        "conductivity_W_m_K": record["conductivity_W_m_K"],
        "cases": cases,
        # The unstabilised operator is exact here, so these are ratios of artificial
        # error rather than of discretisation error.
        "accuracy_gain_at_finest": shipped["finest_error_K"] / weighted["finest_error_K"],
        "order_gain": weighted["minimum_observed_order"] - shipped["minimum_observed_order"],
    }


def _structure(record):
    """Row sums of the two weighted operators and the step operator they sit in."""
    rows = {}
    for entry in record["lengths"]:
        checks = {check["matrix"]: check for check in entry["checks"]}
        rows[entry["streamline_length"]] = {
            "source_action_nonpositive_rows": checks["source_action"]["rows_with_nonpositive_sum"],
            "source_action_row_sum_minimum": checks["source_action"]["row_sum_minimum"],
            "storage_nonpositive_rows": checks["storage_over_step"]["rows_with_nonpositive_sum"],
            "storage_diagonal_minimum": checks["storage_over_step"]["diagonal_minimum"],
            "step_operator_nonpositive_rows": checks["step_operator"]["rows_with_nonpositive_sum"],
            "pivot_magnitude_ratio": entry["step_operator_factorisation"][
                "pivot_magnitude_ratio"
            ],
            "rows": checks["source_action"]["rows"],
            "row_bound": entry.get("row_limit"),
        }
    return rows


def conditioning(unbounded, bounded, length="edge"):
    """What the bound changes about the operators, at one streamline length."""
    before = _structure(_read(unbounded, CONDITIONING_SCHEMA))
    after = _structure(_read(bounded, CONDITIONING_SCHEMA))
    if length not in before or length not in after:
        raise ValueError(f"Neither record reports the {length} streamline length")
    return {
        "unbounded_directory": str(unbounded),
        "bounded_directory": str(bounded),
        "streamline_length": length,
        "unbounded": before[length],
        "bounded": after[length],
        "lengths_agree_once_bounded": {
            name: after[name]["step_operator_nonpositive_rows"] for name in sorted(after)
        },
        "pivot_improvement": after[length]["pivot_magnitude_ratio"]
        / max(before[length]["pivot_magnitude_ratio"], np.finfo(float).tiny),
    }


def replay_cost(directory):
    """Newton iterations per converged slab, and where the run stopped."""
    record = _read(directory, REPLAY_SCHEMA)
    steps = record.get("steps") or []
    converged = [row for row in steps if row.get("status") == "converged"]
    counts = [max(len(row.get("history", [])) - 1, 0) for row in converged]
    terminating = record.get("status")
    return {
        "directory": str(directory),
        "consistent_stabilization": bool(
            (record.get("configuration") or {}).get("consistent_stabilization", False)
        ),
        "slabs": record.get("forward_slabs"),
        "converged_slabs": len(converged),
        "reached_the_horizon": terminating == "converged" and len(converged) == len(steps),
        "terminating_status": terminating,
        "median_newton_iterations": float(np.median(counts)) if counts else None,
        "maximum_newton_iterations": int(max(counts)) if counts else None,
        "first_slab_newton_iterations": counts[0] if counts else None,
        "median_seconds_per_slab": float(np.median([row["seconds"] for row in converged]))
        if converged
        else None,
    }


def build(convergence_directory, unbounded, bounded, replays=()):
    """The whole correction, as one report."""
    return {
        "schema": "coupled-stabilisation-report-v1",
        "convergence": convergence(convergence_directory),
        "conditioning": conditioning(unbounded, bounded),
        "forward_cost": [replay_cost(directory) for directory in replays],
        "scope": (
            "Properties of the discrete thermal operator and of the forward solve on it. "
            "Nothing here is a statement about the optimizer's cost or about the deflation "
            "comparison, which are measured separately on each operator."
        ),
    }


def _fmt(value, spec=".2f", missing="---"):
    return missing if value is None else format(value, spec)


def macros(report):
    """Prose numbers of the correction; a measurement that is missing stays missing."""
    values = {}
    cases = report["convergence"]["cases"]
    for key, label in (
        ("unstabilised", "Unstabilised"),
        ("declared_symmetric_term", "Shipped"),
        ("residual_weighted", "Weighted"),
    ):
        case = cases[key]
        if case["minimum_observed_order"] is not None:
            values[f"coupledStab{label}Order"] = _fmt(case["minimum_observed_order"])
        values[f"coupledStab{label}FinestError"] = _fmt(case["finest_error_K"], ".2e")
    values["coupledStabAccuracyGain"] = _fmt(
        report["convergence"]["accuracy_gain_at_finest"], ".0f"
    )
    unbounded = report["conditioning"]["unbounded"]
    bounded = report["conditioning"]["bounded"]
    values["coupledStabUnboundedBadRows"] = str(unbounded["source_action_nonpositive_rows"])
    values["coupledStabTotalRows"] = str(unbounded["rows"])
    values["coupledStabUnboundedStorageDiagonal"] = _fmt(
        unbounded["storage_diagonal_minimum"], ".0f"
    )
    values["coupledStabBoundedBadRows"] = str(bounded["source_action_nonpositive_rows"])
    values["coupledStabPivotImprovement"] = _fmt(report["conditioning"]["pivot_improvement"], ".1f")
    bound = bounded.get("row_bound") or {}
    if bound:
        values["coupledStabBoundedFraction"] = _fmt(
            100 * bound["fraction_limited"], ".1f"
        )
        values["coupledStabBoundMedianFactor"] = _fmt(bound["median_factor"], ".3f")
    for row in report["forward_cost"]:
        label = "Weighted" if row["consistent_stabilization"] else "Shipped"
        if row["median_newton_iterations"] is not None:
            values[f"coupledStab{label}NewtonMedian"] = _fmt(
                row["median_newton_iterations"], ".0f"
            )
    return values
