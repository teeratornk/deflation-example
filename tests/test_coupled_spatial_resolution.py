"""Spatial replay preserves the signed source function and temporal intervals."""

from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example.coupled_spatial_resolution import transfer_source
from deflation_example.mesh_refinement import refine
from test_axisymmetric_flow import annular_rectangle


def test_nested_source_transfer_keeps_signed_values_without_clipping():
    mesh = annular_rectangle(3)
    fine, P, _ = refine(mesh)
    common = dict(
        physical_steps=np.array([0.2, 0.3]),
        slabs=2,
        temperature_scale=20.0,
        temperature_offset=341.3,
        time_scale=10.0,
        velocity_scale=10.0,
        expansion=0.002,
        buoyancy_reference=341.3,
    )
    coarse_problem = SimpleNamespace(mesh=mesh, spatial_size=len(mesh.free), **common)
    fine_problem = SimpleNamespace(mesh=fine, spatial_size=len(fine.free), **common)
    source = np.linspace(-4, 3, 2 * len(mesh.free)).reshape(2, -1)
    transferred, actual_P = transfer_source(coarse_problem, fine_problem, source)
    expected_P = P[fine.free][:, mesh.free]
    np.testing.assert_array_equal(actual_P.toarray(), expected_P.toarray())
    np.testing.assert_array_equal(transferred, np.asarray(expected_P @ source.T).T)
    assert transferred.min() < 0 and transferred.max() > 0
    fine_problem.physical_steps = np.array([0.25, 0.25])
    with pytest.raises(ValueError, match="time intervals"):
        transfer_source(coarse_problem, fine_problem, source)


def test_spatial_source_transfer_rejects_changed_physical_scaling():
    mesh = annular_rectangle(3)
    coarse = SimpleNamespace(physical_steps=np.ones(2), temperature_scale=20)
    fine = SimpleNamespace(physical_steps=np.ones(2), temperature_scale=10)
    with pytest.raises(ValueError, match="physical scales"):
        transfer_source(coarse, fine, np.zeros(2 * len(mesh.free)))


@pytest.mark.parametrize(
    "corruption", ["source", "configuration", "status", "step", "times", "shape", "nan", "scheme"]
)
def test_coarse_replay_rejects_unmatched_or_incomplete_comparisons(tmp_path, corruption):
    import json
    from deflation_example.coupled_spatial_resolution import coarse_replay_states

    problem = SimpleNamespace(slabs=2, spatial_size=3, physical_steps=np.array([0.2, 0.3]))
    record = {
        "optimization_field_sha256": "source",
        "baseline_sha256": "baseline",
        "configuration": {"slabs": 1},
        "status": "converged",
        "steps": [{"time_s": 0.2, "status": "converged"}, {"time_s": 0.5, "status": "converged"}],
        "forward_solver": {"tolerance": 1e-12},
    }
    times, states = np.array([0.2, 0.5]), np.zeros((2, 3))
    if corruption == "source":
        record["optimization_field_sha256"] = "changed"
    elif corruption == "configuration":
        record["configuration"]["slabs"] = 2
    elif corruption == "status":
        record["status"] = "newton_iteration_cap"
    elif corruption == "step":
        record["steps"][-1]["status"] = "newton_line_search_stagnation"
    elif corruption == "times":
        times[-1] = 0.6
    elif corruption == "shape":
        states = states[:1]
    elif corruption == "nan":
        states[0, 0] = np.nan
    elif corruption == "scheme":
        record["forward_solver"]["time_scheme"] = "bdf2"
    (tmp_path / "record.json").write_text(json.dumps(record))
    np.savez(tmp_path / "states.npz", state=states, times_s=times)
    with pytest.raises(ValueError):
        coarse_replay_states(tmp_path, problem, "baseline", "source", {"slabs": 1})


@pytest.mark.parametrize("scheme", ["backward_euler", "bdf2"])
def test_matched_spatial_time_scheme_retains_all_times_and_source(tmp_path, scheme):
    import json
    from deflation_example.coupled_spatial_resolution import coarse_replay_states

    problem = SimpleNamespace(slabs=4, spatial_size=3, physical_steps=np.full(4, 0.1))
    times, states = np.cumsum(problem.physical_steps), np.arange(12).reshape(4, 3)
    protocol = {
        "time_scheme": scheme,
        "time_integrator_restart": "Backward Euler on the first substep of every original piecewise-constant source interval",
    }
    record = {
        "optimization_field_sha256": "source",
        "baseline_sha256": "baseline",
        "configuration": {"slabs": 2},
        "status": "converged",
        "steps": [{"time_s": float(t), "status": "converged"} for t in times],
        "forward_solver": protocol,
    }
    (tmp_path / "record.json").write_text(json.dumps(record))
    np.savez(tmp_path / "states.npz", state=states, times_s=times)
    actual, metadata = coarse_replay_states(
        tmp_path, problem, "baseline", "source", {"slabs": 2}, time_scheme=scheme
    )
    np.testing.assert_array_equal(actual, states)
    assert metadata["forward_solver"] == protocol
    if scheme == "bdf2":
        del record["forward_solver"]["time_integrator_restart"]
        (tmp_path / "record.json").write_text(json.dumps(record))
        with pytest.raises(ValueError, match="restarts"):
            coarse_replay_states(
                tmp_path, problem, "baseline", "source", {"slabs": 2}, time_scheme=scheme
            )


@pytest.mark.parametrize(
    "options",
    [
        ["--time-scheme", "bdf2"],
        ["--time-scheme", "bdf2", "--procedure", "monolithic_newton"],
        ["--subdivision", "2", "--procedure", "monolithic_newton"],
    ],
)
def test_spatial_cli_rejects_missing_matching_time_protocol_before_data_access(
    monkeypatch, options
):
    import sys
    from deflation_example.coupled_spatial_resolution import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coupled_spatial_resolution",
            "--baseline",
            "unused",
            "--fine-baseline",
            "unused",
            "--optimization",
            "unused",
            "--output",
            "unused",
            *options,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
