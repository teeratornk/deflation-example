"""Validate separately recomputed equations against the exact saved trajectory."""

import json
from pathlib import Path

import numpy as np

from .coupled_saved import file_digest
from .validation import integer, positive_real


def read_equation_audit(directory, record, audit_path):
    """Require checksum, time, status and numerical agreement; never relabel data."""
    directory, audit_path = Path(directory), Path(audit_path)
    audit = json.loads(audit_path.read_text())
    if audit.get("schema") != "coupled-saved-forward-equation-check-v1":
        raise ValueError("Unrecognized independent-equation audit")
    required = {
        "forward_record_sha256": file_digest(directory / "record.json"),
        "forward_fields_sha256": file_digest(directory / "states.npz"),
        "optimization_fields_sha256": record["optimization_field_sha256"],
        "baseline_sha256": record["baseline_sha256"],
        "forward_formulation": record.get("forward_formulation"),
        "forward_source": record["environment"]["git_head"],
    }
    if any(audit.get(key) != value for key, value in required.items()):
        raise ValueError("Independent audit must identify the exact fields, inputs and source")
    if (
        audit["equation_tolerance"] != 1e-12
        or audit["conservation_tolerance"] != 1e-6
        or positive_real(record["forward_solver"]["tolerance"], "Forward tolerance") > 1e-12
    ):
        raise ValueError("Independent audit must use the declared strict criteria")
    slabs = integer(record["configuration"]["slabs"], "Original slabs", 1) * integer(
        record["subdivision"], "Subdivision", 1
    )
    count = len(record["steps"])
    if (
        not 0 < count <= slabs
        or record["forward_slabs"] != slabs
        or audit["declared_steps"] != slabs
        or audit["saved_steps"] != count
        or len(audit["steps"]) != count
    ):
        raise ValueError("Independent audit and trajectory step counts differ")
    horizon = positive_real(record["configuration"]["horizon_s"], "Physical horizon")
    limits = {
        "momentum_relative_residual": 1e-12,
        "continuity_relative_residual": 1e-12,
        "thermal_relative_residual": 1e-12,
        "mass_relative_imbalance": 1e-6,
        "energy_relative_defect": 1e-6,
    }
    flags = []
    for n, (saved, row) in enumerate(zip(record["steps"], audit["steps"], strict=True)):
        time = (n + 1) * horizon / slabs
        if (
            row["slab_zero_based"] != n
            or row["recorded_status"] != saved["status"]
            or not np.isclose(row["time_s"], time, rtol=0, atol=1e-10)
            or not np.isclose(saved["time_s"], time, rtol=0, atol=1e-10)
        ):
            raise ValueError("Independent audit must follow the declared physical time grid")
        values = {key: row["equations"][key] for key in limits}
        values.update(
            {
                key: row[key]
                for key in (
                    "velocity_boundary_maximum_absolute_error",
                    "pressure_gauge_absolute_error",
                )
            }
        )
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0
            for v in values.values()
        ):
            raise ValueError("Equation norms and boundary errors must be finite and nonnegative")
        good = (
            saved["status"] == "converged"
            and all(values[key] <= limit for key, limit in limits.items())
            and values["velocity_boundary_maximum_absolute_error"] <= 1e-12
            and values["pressure_gauge_absolute_error"] <= 1e-12
        )
        if row["verified"] is not bool(good):
            raise ValueError("Independent verification flag disagrees with its original equations")
        flags.append(good)
    prefix = next((n for n, good in enumerate(flags) if not good), count)
    complete = count == slabs and all(flags) and record["status"] == "converged"
    if audit["all_saved_steps_verified"] is not bool(all(flags)) or audit[
        "complete_trajectory_verified"
    ] is not bool(complete):
        raise ValueError("Independent verification totals disagree with their steps")
    if record["status"] == "converged" and not complete:
        raise ValueError("Converged labels cannot override failed original equations")
    if any(flags[prefix:]):
        raise ValueError("Verified prefix cannot skip an unsuccessful physical step")
    return {
        "audit_file": audit_path.name,
        "audit_sha256": file_digest(audit_path),
        "verified_steps": prefix,
        "all_steps_verified": complete,
        "verified_prefix_maxima": {
            key: max((r["equations"][key] for r in audit["steps"][:prefix]), default=None)
            for key in limits
        },
    }
