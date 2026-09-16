"""A spatial comparison must be pure: one source, one time grid, two meshes."""

import json
from pathlib import Path

import numpy as np
import pytest

from deflation_example.coupled_resolution_v2 import statistics_from_arrays
from deflation_example.coupled_spatial_pair import (
    checked_pair,
    load_protocol,
    load_trajectory,
    procedure_of,
    split_assessment,
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
    fine = load_trajectory(stall(trajectory(tmp_path / "fine", dofs=21), 2), allow_partial=True)
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


def declared_protocol(path, boundary, *, schema="coupled-resolution-protocol-v3"):
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "split": {"pointwise_interval_s": [0.0, boundary]},
                "criteria": {
                    "pointwise_interval": {
                        "maximum_temperature_difference_K": 0.05,
                        "mass_weighted_rms_difference_K": 0.05,
                    },
                    "full_horizon": {"mass_weighted_rms_difference_K": 0.05},
                },
            }
        )
    )
    return path


def split_inputs(levels=8, nodes=5, late=10.0):
    """A difference that is nothing early and large late, which is the case at issue."""
    times = np.linspace(1.0, float(levels), levels)
    mesh = annular_rectangle(2)
    mass = np.full(nodes, 0.25)
    coarse = np.zeros((levels, nodes))
    fine = np.zeros((levels, nodes))
    fine[times > levels / 2] = late
    return times, mesh, mass, coarse, fine


def test_the_declared_split_judges_the_interval_and_withholds_what_it_cannot_see(tmp_path):
    times, mesh, mass, coarse, fine = split_inputs()
    protocol, digest = load_protocol(declared_protocol(tmp_path / "p.json", 4.0))
    report = split_assessment(
        protocol, digest, coarse, times, fine, times, mesh, mass, 1.0, whole=True
    )
    assert report["protocol_sha256"] == digest
    assert report["levels_in_interval"] == 4 and report["interval_reached"]
    # The early interval agrees exactly, so it passes while the horizon does not.
    assert report["interval_statistics"]["pointwise_maximum_K"] == 0.0
    assert report["pointwise_met"] and report["interval_rms_met"]
    assert report["horizon_rms_met"] is False and report["horizon_rms_K"] > 0.05
    assert report["not_judged_here"] == [
        "tracking_relative_change",
        "maximum_upper_violation_change_K",
    ]


def test_a_trajectory_that_stops_inside_the_interval_gets_no_verdict(tmp_path):
    times, mesh, mass, coarse, fine = split_inputs()
    protocol, digest = load_protocol(declared_protocol(tmp_path / "p.json", 6.0))
    short = slice(0, 3)
    report = split_assessment(
        protocol,
        digest,
        coarse[short],
        times[short],
        fine[short],
        times[short],
        mesh,
        mass,
        1.0,
        whole=False,
    )
    assert report["levels_in_interval"] == 3 and not report["interval_reached"]
    assert report["pointwise_met"] is None and report["interval_rms_met"] is None
    assert report["horizon_rms_met"] is None
    assert report["interval_statistics"]["pointwise_maximum_K"] == 0.0


def test_only_the_declared_protocol_is_accepted(tmp_path):
    with pytest.raises(ValueError, match="coupled-resolution-protocol-v3"):
        load_protocol(declared_protocol(tmp_path / "q.json", 4.0, schema="something-else"))
    bad = tmp_path / "r.json"
    bad.write_text(
        json.dumps(
            {
                "schema": "coupled-resolution-protocol-v3",
                "split": {"pointwise_interval_s": [0.0, 0.0]},
            }
        )
    )
    with pytest.raises(ValueError, match="positive span"):
        load_protocol(bad)


def test_the_declared_protocol_on_disk_is_the_one_the_reporter_reads():
    """The file the study declared must satisfy the reader, not only a fixture."""
    declared = Path("runs/coupled-resolution-protocol-v3.json")
    if not declared.exists():
        pytest.skip("the declared protocol is not present on this machine")
    protocol, digest = load_protocol(declared)
    assert protocol["interpretation"]["declared_before_reading"] is True
    assert protocol["split"]["pointwise_interval_s"][1] == 93.75
    assert len(digest) == 64


def replay_trajectory(directory, *, slabs, reached, subdivision=1, horizon=1.0):
    """A forward-replay record, which names its declared length forward_slabs."""
    directory.mkdir(parents=True, exist_ok=True)
    times = np.cumsum(np.full(reached, horizon / slabs))
    (directory / "record.json").write_text(
        json.dumps(
            {
                "schema": "coupled-fixed-source-newton-replay-v1",
                "status": "newton_line_search_stagnation",
                "forward_slabs": slabs,
                "subdivision": subdivision,
                "optimization_field_sha256": "src",
                "forward_solver": {"procedure": "monolithic_newton"},
                "steps": [
                    {"time_s": float(t), "status": "converged"} for t in times
                ]
                + [{"time_s": 0.0, "status": "newton_line_search_stagnation"}],
            }
        )
    )
    np.savez(
        directory / "states.npz",
        state=np.linspace(0.0, 1.0, reached * 6).reshape(reached, 6),
        times_s=times,
    )
    return directory


def test_a_stalled_replay_keeps_the_length_it_was_meant_to_have(tmp_path):
    """Falling back to what was reached would make a stalled run look complete."""
    stalled = load_trajectory(
        replay_trajectory(tmp_path / "stalled", slabs=64, reached=3),
        allow_partial=True,
    )
    assert stalled["levels_reached"] == 3
    assert stalled["declared_levels"] == 64 and stalled["slabs"] == 64
    assert stalled["complete"] is False
    assert stalled["procedure"]["procedure"] == "monolithic_newton"
