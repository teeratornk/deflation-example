"""Continuation identity, fresh preceding equations and local branch diagnostics."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example.coupled_saved import file_digest
from deflation_example.reporting import write_arrays, write_report
from test_coupled_newton_replay import data
from test_coupled_switch import switch_problem

directory = Path(__file__).parents[1] / "examples/coupled_optimization/fixed_point"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location(
    "switch_diagnostic", directory / "switch_diagnostic.py"
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
sys.path.pop(0)


def fixture(tmp_path, monkeypatch):
    problem, expected, evaluation, _ = data()
    args = SimpleNamespace(
        trajectory=tmp_path / "original",
        continuation=tmp_path / "continued",
        repair=tmp_path / "repair",
        slab=None,
        threads=1,
    )
    for folder in (args.trajectory, args.continuation, args.repair):
        folder.mkdir()
    metadata = {
        "optimization_field_sha256": "control-id",
        "baseline_sha256": "base-id",
        "fine_baseline_sha256": "fine-id",
        "slabs": 2,
        "spatial_state_dofs": problem.spatial_size,
        "configuration": {"slabs": 2},
    }
    original = {
        **metadata,
        "time_scheme": "backward_euler",
        "policy": "anderson5",
        "repetition": 0,
        "subdivision": 1,
        "steps": [{"field_sha256": "failed-id"}],
    }
    write_report(args.trajectory / "record.json", original)
    write_report(
        args.repair / "record.json",
        {
            "trajectory_record_sha256": file_digest(args.trajectory / "record.json"),
            "status": "complete",
            "verified": True,
            "slab_zero_based": 0,
            "initial_field_sha256": "failed-id",
        },
    )
    rows = []
    for n in range(2):
        filename = f"step-{n}.npz"
        write_arrays(
            args.continuation / filename,
            state=expected.reshape(2, -1)[n],
            velocity=evaluation.flows[n].velocity,
            pressure=evaluation.flows[n].pressure,
        )
        rows.append(
            {
                "fields": filename,
                "field_sha256": file_digest(args.continuation / filename),
                "verified": n == 0,
                "time_s": float(problem.physical_steps[: n + 1].sum()),
                "slab_zero_based": n,
            }
        )
    continued = {
        "status": "numerical_failure",
        "corrected_slab": 0,
        "steps": rows,
        "configuration": metadata["configuration"],
        "slabs": 2,
        "trajectory_record_sha256": file_digest(args.trajectory / "record.json"),
        "repair_record_sha256": file_digest(args.repair / "record.json"),
    }
    write_report(args.continuation / "record.json", continued)
    monkeypatch.setattr(
        diagnostic,
        "prepare",
        lambda args, subdivision: (problem, evaluation.control.reshape(2, -1).copy(), metadata),
    )
    return args, problem, expected


def test_continuation_loader_checks_hashes_and_recovers_the_same_fields(tmp_path, monkeypatch):
    args, problem, expected = fixture(tmp_path, monkeypatch)
    result = diagnostic.load_continuation(args)
    assert result[0] is problem and result[-2] == 1
    np.testing.assert_array_equal(result[4][problem.free], expected.reshape(2, -1)[1])


@pytest.mark.parametrize("change", ["hash", "time", "status", "previous", "configuration", "slab"])
def test_continuation_loader_rejects_mismatched_inputs(tmp_path, monkeypatch, change):
    args, _, _ = fixture(tmp_path, monkeypatch)
    path = args.continuation / "record.json"
    record = json.loads(path.read_text())
    if change == "hash":
        record["repair_record_sha256"] = "different"
    elif change == "time":
        record["steps"][1]["time_s"] += 1
    elif change == "status":
        record["status"] = "running"
    elif change == "previous":
        record["steps"][0]["verified"] = False
    elif change == "configuration":
        record["configuration"]["slabs"] += 1
    else:
        args.slab = 2
    write_report(path, record)
    with pytest.raises(ValueError):
        diagnostic.load_continuation(args)


def test_fresh_previous_equations_override_a_verified_label(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(diagnostic, "criteria_met", lambda *args: False)
    with pytest.raises(ValueError, match="Previous fields"):
        diagnostic.load_continuation(args)


def test_switch_distance_has_the_correct_sign_and_zero_velocity_limit():
    problem, velocity = switch_problem()
    tie = diagnostic.branch_values(problem, velocity)
    assert np.max(np.abs(tie["distance"])) < 1e-14
    assert tie["strict_switch"].all()
    assert (diagnostic.branch_values(problem, 2 * velocity)["distance"] > 0).all()
    zero = diagnostic.branch_values(problem, 0 * velocity)
    np.testing.assert_array_equal(zero["distance"], -1)
    assert not zero["advective"].any()


def test_step_inspection_reports_original_equations_and_keeps_inputs():
    problem, _, _, source = data()
    state = problem.full_temperature(problem.initial)
    flow = problem.initial_flow
    before = [x.copy() for x in (state, flow.velocity, source)]
    report = diagnostic.inspect(problem, source, state, flow, state, flow, 0)
    assert report["linear_relative_residual"] < 1e-10
    assert len(report["trials"]) == 10
    assert report["trials"][0]["fixed_scaled_residual_ratio"] < 1
    for a, b in zip(before, (state, flow.velocity, source), strict=True):
        np.testing.assert_array_equal(a, b)
