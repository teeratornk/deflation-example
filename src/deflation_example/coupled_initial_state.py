"""Assessed checkpoint temperatures used as fresh optimization initial guesses."""

import json
from pathlib import Path

import numpy as np

from .coupled_bounds import temperature_bounds
from .reporting import file_sha256
from .validation import positive_real


def snapshot_initial_guess(cfg, problem, baseline, position):
    """Load only temperature and flow guesses; discard every optimization history.

    The derivative assessment binds the checksum-verified snapshot to the supplied
    baseline and evaluated streamline rule. It need not certify a stationary
    state. Every flow, recovered control and gradient is evaluated anew by the
    optimizer, including time levels preceding the derivative-test suffix.
    """
    from .coupled_review import read_snapshot
    from .coupled_sequence import RestoredEvaluation

    record, meta, arrays, manifest = read_snapshot(Path(cfg["initial_state_snapshot"]))
    assessment_path = Path(cfg["initial_state_assessment"])
    assessment = json.loads(assessment_path.read_text())
    if (
        assessment.get("schema") != "coupled-frozen-prefix-review-v1"
        or assessment.get("status") != "review_complete"
        or assessment.get("snapshot") != manifest
        or assessment.get("baseline_sha256") != baseline["baseline_sha256"]
        or assessment.get("evaluated_streamline_rule") != cfg.get("streamline_rule", "hard_min")
    ):
        raise ValueError("The initial-state assessment must match snapshot, baseline and rule")
    rows = assessment.get("centered_checks", {}).get("rows", [])
    if len(rows) != 4 or any(row.get("status") != "evaluated" for row in rows):
        raise ValueError("The initial-state assessment must retain four evaluated centered checks")
    original = record["configuration"]
    alpha_policy = cfg.get("initial_state_alpha_policy", "identical")
    if alpha_policy not in {"identical", "shared_temperature"}:
        raise ValueError("Choose identical or shared_temperature initial-state alpha policy")
    source_alpha = positive_real(original["alpha"], "Snapshot alpha")
    current_alpha = positive_real(cfg["alpha"], "Optimization alpha")
    if alpha_policy == "identical" and source_alpha != current_alpha:
        raise ValueError("Initial-state snapshot differs in alpha")
    if problem.alpha != current_alpha:
        raise ValueError("Loaded problem differs from the declared alpha")
    for key in ("transient", "slabs", "horizon_s", "target_count", "lower_K"):
        if key not in original or original[key] != cfg[key]:
            raise ValueError(f"Initial-state snapshot differs in {key}")
    for key, default in (
        ("target_startup_s", 0.0),
        ("temperature_margin_K", 0.0),
        ("transport_form", "advective"),
        ("consistent_stabilization", False),
    ):
        if original.get(key, default) != cfg.get(key, default):
            raise ValueError(f"Initial-state snapshot differs in {key}")
    if original["queries"][meta["position"]] != cfg["queries"][position]:
        raise ValueError("Initial-state snapshot must share the selected target and bound")
    expected = {
        "state": (problem.size,),
        "velocity": (problem.slabs, problem.flow.nv, 2),
        "pressure": (problem.slabs, problem.flow.np),
    }
    for name, shape in expected.items():
        if np.shape(arrays[name]) != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Initial-state {name} must match the full finite trajectory")
    bounds = temperature_bounds(cfg, upper_K=cfg["queries"][position]["upper_K"])
    lower = (
        bounds["optimization_lower_K"] - problem.temperature_offset
    ) / problem.temperature_scale
    upper = (
        bounds["optimization_upper_K"] - problem.temperature_offset
    ) / problem.temperature_scale
    if np.any(arrays["state"] < lower) or np.any(arrays["state"] > upper):
        raise ValueError("Initial temperature violates the declared bounds; no clipping is applied")
    return RestoredEvaluation(arrays["state"], arrays["velocity"], arrays["pressure"]), {
        "policy": "fresh_optimization_from_assessed_checkpoint_temperature",
        "snapshot": manifest,
        "assessment_sha256": file_sha256(assessment_path),
        "baseline_sha256": baseline["baseline_sha256"],
        "source_streamline_rule": original.get("streamline_rule", "hard_min"),
        "evaluated_streamline_rule": cfg.get("streamline_rule", "hard_min"),
        "alpha_policy": alpha_policy,
        "source_alpha": source_alpha,
        "optimization_alpha": current_alpha,
        "retained_secant_pairs": 0,
        "retained_recycling_directions": 0,
        "scope": "Complete trajectory reevaluation; saved flows are initial guesses. No old controls, gradients, damping, iteration counts or solver histories are imported. Prior optimization cost is excluded.",
    }
