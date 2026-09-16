"""A spatial comparison must be pure: one source, one time grid, two meshes."""

import json

import numpy as np
import pytest

from deflation_example.coupled_resolution_v2 import statistics_from_arrays
from deflation_example.coupled_spatial_pair import (
    checked_pair,
    load_trajectory,
    procedure_of,
)
from deflation_example.mesh_refinement import refine
from test_axisymmetric_flow import annular_rectangle


def trajectory(
    directory,
    *,
    dofs,
    slabs=4,
    subdivision=1,
    status="converged",
    source="src",
    scheme="backward_euler",
    horizon=1.0,
    extra=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    times = np.cumsum(np.full(slabs, horizon / slabs))
    record = {
        "schema": "coupled-fixed-point-study-v1",
        "status": status,
        "slabs": slabs,
        "subdivision": subdivision,
        "spatial_state_dofs": dofs,
        "time_scheme": scheme,
        "optimization_field_sha256": source,
        "steps": [{"time_s": float(t), "status": "converged"} for t in times],
        **(extra or {}),
    }
    (directory / "record.json").write_text(json.dumps(record))
    np.savez(
        directory / "states.npz",
        state=np.linspace(0.0, 1.0, slabs * dofs).reshape(slabs, dofs),
        times_s=times,
    )
    return directory


def test_a_fixed_point_policy_is_read_as_the_forward_procedure():
    assert procedure_of({"policy": "anderson5", "settings": {"depth": 5}, "family": "forward"}) == {
        "source": "fixed_point_policy",
        "procedure": "anderson5",
        "settings": {"depth": 5},
        "family": "forward",
    }
    monolithic = procedure_of(
        {"forward_solver": {"procedure": "monolithic_newton", "newton_cap": 30}}
    )
    assert monolithic["procedure"] == "monolithic_newton" and monolithic["newton_cap"] == 30
    assert procedure_of({})["procedure"] is None


def stall(directory, at):
    record = json.loads((directory / "record.json").read_text())
    record["status"] = "numerical_failure"
    record["steps"][at]["status"] = "newton_line_search_stagnation"
    del record["steps"][at + 1 :]
    (directory / "record.json").write_text(json.dumps(record))
    return directory


def test_an_unconverged_trajectory_is_refused_unless_a_partial_one_is_asked_for(tmp_path):
    directory = trajectory(tmp_path / "a", dofs=6, status="numerical_failure")
    with pytest.raises(ValueError, match="converged trajectory"):
        load_trajectory(directory)
    stalled = stall(trajectory(tmp_path / "b", dofs=6), 2)
    with pytest.raises(ValueError, match="converged trajectory"):
        load_trajectory(stalled)
    # Asked for explicitly, the leading converged interval is read and labelled.
    partial = load_trajectory(stalled, allow_partial=True)
    assert partial["complete"] is False
    assert partial["levels_reached"] == 2 and partial["declared_levels"] == 4
    assert partial["terminating_status"] == "newton_line_search_stagnation"
    assert partial["state"].shape[0] == 2 and len(partial["times_s"]) == 2


def test_a_converged_step_after_a_failed_one_is_refused(tmp_path):
    directory = trajectory(tmp_path / "c", dofs=6)
    record = json.loads((directory / "record.json").read_text())
    record["status"] = "numerical_failure"
    record["steps"][1]["status"] = "coupling_iteration_cap"
    (directory / "record.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="converged step after a failed one"):
        load_trajectory(directory, allow_partial=True)


def test_the_comparison_covers_the_interval_both_trajectories_reached(tmp_path):
    coarse = load_trajectory(trajectory(tmp_path / "coarse", dofs=6), allow_partial=True)
    fine = load_trajectory(
        stall(trajectory(tmp_path / "fine", dofs=21), 2), allow_partial=True
    )
    shared = checked_pair(coarse, fine, "src")
    assert shared == 2, "the comparison stops where the fine trajectory stopped"
    assert coarse["complete"] and not fine["complete"]


def test_the_pair_must_differ_in_the_mesh_and_nothing_else(tmp_path):
    coarse = load_trajectory(trajectory(tmp_path / "coarse", dofs=6))
    fine = load_trajectory(trajectory(tmp_path / "fine", dofs=21))
    checked_pair(coarse, fine, "src")

    with pytest.raises(ValueError, match="different saved source"):
        checked_pair(coarse, fine, "other")

    # Refining time as well as space is the confound this module exists to stop.
    both = load_trajectory(trajectory(tmp_path / "both", dofs=21, slabs=8, subdivision=2))
    with pytest.raises(ValueError, match="refine time as well"):
        checked_pair(coarse, both, "src")

    same = load_trajectory(trajectory(tmp_path / "same", dofs=6))
    with pytest.raises(ValueError, match="same mesh"):
        checked_pair(coarse, same, "src")

    with pytest.raises(ValueError, match="more spatial degrees"):
        checked_pair(fine, coarse, "src")

    bdf2 = load_trajectory(trajectory(tmp_path / "bdf2", dofs=21, scheme="bdf2"))
    with pytest.raises(ValueError, match="one time integration scheme"):
        checked_pair(coarse, bdf2, "src")


def test_a_lifted_coarse_trajectory_gives_a_zero_difference_against_itself():
    """Nested interpolation of the coarse states is exact, so identical fields agree."""
    mesh = annular_rectangle(3)
    fine_mesh, P, _ = refine(mesh)
    prolongation = P[fine_mesh.free][:, mesh.free]
    coarse_state = np.linspace(-0.4, 0.9, 4 * len(mesh.free)).reshape(4, -1)
    lifted = np.asarray(prolongation @ coarse_state.T).T
    times = np.cumsum(np.full(4, 0.25))
    mass = np.full(len(fine_mesh.free), 0.5)
    statistics = statistics_from_arrays(lifted, times, lifted, times, fine_mesh, mass, 20.0, 0.0)
    # Interpolation at the shared time levels reproduces the values to roundoff.
    assert statistics["pointwise_maximum_K"] < 1e-12
    assert statistics["mass_weighted_space_time_rms_K"] < 1e-12

    shifted = lifted + 0.001
    moved = statistics_from_arrays(lifted, times, shifted, times, fine_mesh, mass, 20.0, 0.0)
    assert moved["pointwise_maximum_K"] == pytest.approx(0.02)
    assert moved["mass_weighted_space_time_rms_K"] == pytest.approx(0.02)


def test_pair_statistics_still_reads_two_directories(tmp_path):
    """The directory entry point keeps working, so the temporal assessment is unchanged."""
    from deflation_example.coupled_resolution_v2 import pair_statistics

    mesh = annular_rectangle(3)
    dofs = len(mesh.free)
    coarse = trajectory(tmp_path / "c", dofs=dofs, slabs=2)
    fine = trajectory(tmp_path / "f", dofs=dofs, slabs=4)
    mass = np.full(dofs, 0.25)
    statistics = pair_statistics(coarse, fine, mesh, mass, 20.0, 0.0)
    assert statistics["coarse_levels"] == 2 and statistics["fine_levels"] == 4
    assert statistics["pointwise_maximum_K"] >= 0.0
