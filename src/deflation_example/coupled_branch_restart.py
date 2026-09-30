"""Checked trajectory inputs for a matched flow-branch optimizer restart."""

import json
from pathlib import Path

import numpy as np

from .reporting import file_sha256


def load_branches(assessment_path, record_path, field_path, saved, source_hashes):
    """Bind both complete flow trajectories to one frozen optimization problem."""
    assessment_path = Path(assessment_path)
    report = json.loads(assessment_path.read_text())
    if (
        report.get("schema") != "coupled-local-flow-branch-v1"
        or report.get("status") != "complete"
        or not report.get("trajectory_check_requested")
        or report.get("record_sha256") != file_sha256(record_path)
        or report.get("field_sha256") != file_sha256(field_path)
    ):
        raise ValueError("The complete branch assessment must match the source trajectory")
    # The diagnostic driver may change; the physical discretization stays frozen.
    for name in (
        "axisymmetric_flow.py",
        "coupled_control.py",
        "coupled_flow_solve.py",
        "coupled_optimize.py",
        "coupled_pilot.py",
        "meshes.py",
        "oil_properties.py",
        "assess_transformer.py",
    ):
        previous = report["environment"]["source_sha256"].get(name)
        if previous is None or previous != source_hashes.get(name):
            raise ValueError(f"The assessed physical implementation differs: {name}")
    rows = report.get("trajectory_checks", [])
    if len(rows) != 2 or {r.get("initial_flow") for r in rows} != {
        "retained",
        "alternate_seed",
    }:
        raise ValueError("Exactly two assessed flow trajectories are required")
    branches, hashes = {}, {}
    for row in rows:
        if (
            row.get("status") != "evaluated"
            or row.get("equations_verified") is not True
            or row.get("adjoint_verified") is not True
        ):
            raise ValueError("Both assessed trajectories must pass equation and adjoint checks")
        name = row["initial_flow"]
        path = assessment_path.parent / f"trajectory-{name}.npz"
        digest = file_sha256(path)
        if digest != row.get("field_sha256"):
            raise ValueError("The assessed trajectory checksum differs")
        with np.load(path, allow_pickle=False) as data:
            fields = {k: data[k].copy() for k in ("state", "velocity", "pressure")}
        for key, values in fields.items():
            if values.shape != saved[key].shape or not np.isfinite(values).all():
                raise ValueError("Assessed fields must be finite and match the source dimensions")
        if not np.array_equal(fields["state"], saved["state"]):
            raise ValueError("Both branch restarts require the identical source temperature")
        branches[name], hashes[name] = fields, digest
    separation = float(
        np.linalg.norm(branches["retained"]["velocity"] - branches["alternate_seed"]["velocity"])
    )
    if separation <= 1e-10:
        raise ValueError("The two assessed velocity trajectories must be distinguishable")
    return branches, {
        "assessment_sha256": file_sha256(assessment_path),
        "trajectory_sha256": hashes,
        "velocity_separation_nodal_norm_m_s": separation,
        "branch_preservation_rule": "distance to selected seed <= 0.1 * branch separation + 1e-12 m/s",
    }


def branch_identity(velocity, branches, selected):
    """Measure initial-guess retention using velocities alone, in nodal norm."""
    if selected not in branches:
        raise ValueError("Unknown initial flow branch")
    other = "alternate_seed" if selected == "retained" else "retained"
    velocity = np.asarray(velocity)
    if velocity.shape != branches[selected]["velocity"].shape or not np.isfinite(velocity).all():
        raise ValueError("The reevaluated velocity has invalid dimensions or entries")
    separation = float(np.linalg.norm(branches[selected]["velocity"] - branches[other]["velocity"]))
    distance = float(np.linalg.norm(velocity - branches[selected]["velocity"]))
    other_distance = float(np.linalg.norm(velocity - branches[other]["velocity"]))
    return {
        "selected": selected,
        "distance_to_selected_nodal_norm_m_s": distance,
        "distance_to_other_nodal_norm_m_s": other_distance,
        "preserved": bool(separation > 1e-10 and distance <= 0.1 * separation + 1e-12),
    }
