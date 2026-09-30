"""Input identity, branch preservation and returned-field verification."""

import copy
import json
import sys

import numpy as np
import pytest

from deflation_example.coupled_branch_restart import branch_identity, load_branches
from deflation_example.coupled_recovery import RecoveryStore, identity
from deflation_example.reporting import environment, file_sha256, write_arrays, write_report


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source.json"
    write_report(source, {"status": "trust_radius_exhausted", "configuration": {"alpha": 1e-14}})
    saved = dict(state=np.arange(4.0), velocity=np.zeros((2, 3, 2)), pressure=np.zeros((2, 2)))
    field = tmp_path / "target-01.npz"
    write_arrays(field, **saved)
    hashes = environment()["source_sha256"]
    report = dict(
        schema="coupled-local-flow-branch-v1",
        status="complete",
        trajectory_check_requested=True,
        record_sha256=file_sha256(source),
        field_sha256=file_sha256(field),
        environment={"source_sha256": hashes},
        trajectory_checks=[],
    )
    for name in ("retained", "alternate_seed"):
        data = {k: v.copy() for k, v in saved.items()}
        if name == "alternate_seed":
            data["velocity"][1, 1, 0] = 1e-4
        path = tmp_path / f"trajectory-{name}.npz"
        write_arrays(path, **data)
        report["trajectory_checks"].append(
            dict(
                initial_flow=name,
                status="evaluated",
                equations_verified=True,
                adjoint_verified=True,
                field_sha256=file_sha256(path),
            )
        )
    assessment = tmp_path / "assessment.json"
    write_report(assessment, report)
    return assessment, source, field, saved, hashes


def test_complete_branch_inputs_are_copied_and_preserved(inputs):
    before = copy.deepcopy(inputs[3])
    branches, metadata = load_branches(*inputs)
    assert metadata["velocity_separation_nodal_norm_m_s"] == pytest.approx(1e-4)
    for name in branches:
        assert branch_identity(branches[name]["velocity"], branches, name)["preserved"]
    assert not branch_identity(branches["retained"]["velocity"], branches, "alternate_seed")[
        "preserved"
    ]
    branches["retained"]["state"][:] = 100
    for key in before:
        np.testing.assert_array_equal(inputs[3][key], before[key])


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "unknown"),
        ("status", "running"),
        ("trajectory_check_requested", False),
        ("record_sha256", "incorrect"),
        ("field_sha256", "incorrect"),
    ],
)
def test_assessment_must_identify_complete_matching_inputs(inputs, key, value):
    report = json.loads(inputs[0].read_text())
    report[key] = value
    write_report(inputs[0], report)
    with pytest.raises(ValueError, match="assessment"):
        load_branches(*inputs)


@pytest.mark.parametrize("change", ["duplicate", "missing", "unverified", "bad_hash", "source"])
def test_invalid_assessed_population_and_source_fail(inputs, change):
    report = json.loads(inputs[0].read_text())
    if change == "duplicate":
        report["trajectory_checks"][1] = report["trajectory_checks"][0]
    elif change == "missing":
        report["trajectory_checks"].pop()
    elif change == "unverified":
        report["trajectory_checks"][1]["adjoint_verified"] = False
    elif change == "bad_hash":
        report["trajectory_checks"][1]["field_sha256"] = "changed"
    else:
        report["environment"]["source_sha256"]["coupled_control.py"] = "changed"
    write_report(inputs[0], report)
    with pytest.raises(ValueError):
        load_branches(*inputs)


@pytest.mark.parametrize("change", ["temperature", "shape", "nonfinite", "same_branch"])
def test_fields_are_checked_beyond_their_checksum(inputs, change):
    path = inputs[0].parent / "trajectory-alternate_seed.npz"
    with np.load(path, allow_pickle=False) as archive:
        fields = {k: archive[k].copy() for k in archive.files}
    if change == "temperature":
        fields["state"][0] += 1e-5
    elif change == "shape":
        fields["pressure"] = np.zeros((1, 2))
    elif change == "nonfinite":
        fields["velocity"][0, 0, 0] = np.nan
    else:
        fields["velocity"][:] = 0
    write_arrays(path, **fields)
    report = json.loads(inputs[0].read_text())
    report["trajectory_checks"][1]["field_sha256"] = file_sha256(path)
    write_report(inputs[0], report)
    with pytest.raises(ValueError):
        load_branches(*inputs)


def test_checkpoint_identity_changes_with_branch_and_source(inputs, tmp_path):
    _, metadata = load_branches(*inputs)
    settings = {"branch": "retained", "inputs": metadata, "outer_cap": 100}
    fingerprint = identity(settings, inputs[4])
    store = RecoveryStore(tmp_path / "recovery", fingerprint)
    store.save({"state": inputs[3]["state"], "residual": 0.25})
    assert store.load()["residual"] == 0.25
    for changed in ({**settings, "branch": "alternate_seed"}, {**settings, "outer_cap": 40}):
        with pytest.raises(ValueError, match="differs"):
            RecoveryStore(store.directory, identity(changed, inputs[4])).load()
    with pytest.raises(ValueError, match="differs"):
        RecoveryStore(store.directory, identity(settings, {"changed": "source"})).load()


def test_restart_returns_fresh_metrics_for_its_saved_state(tmp_path, monkeypatch):
    from deflation_example import coupled_radius_restart as driver
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    state = np.full(problem.size, 0.1)
    desired = np.full(problem.size, 0.2)
    evaluation = problem.evaluate(state)
    cfg = dict(
        threads=1,
        target_count=16,
        target_startup_s=60,
        lower_K=problem.temperature_offset - 0.3 * problem.temperature_scale,
        queries=[{"upper_K": problem.temperature_offset + 0.3 * problem.temperature_scale}],
        inner_tolerance=1e-10,
        inner_cap=50000,
        inner_refresh=50,
        nonlinear_tolerance=1e-8,
        qp_tolerance=1e-10,
        qp_cap=100,
        secant_memory=10,
        inner_preconditioner="frozen",
        frozen_sweeps=3,
        trust_accuracy="adaptive_projected",
        trial_policy="backtrack",
        qp_solver="projected",
        qp_correction_policy="kkt_decrease",
        equation_acceptance_tolerance=1e-9,
        conservation_tolerance=1e-6,
    )
    source = tmp_path / "source.json"
    write_report(
        source,
        dict(
            configuration=cfg,
            status="trust_radius_exhausted",
            cases=[{"status": "trust_radius_exhausted", "target": 8}],
            baseline_sha256="fixture",
        ),
    )
    write_arrays(
        tmp_path / "target-00.npz",
        state=state,
        desired=desired,
        velocity=np.stack([f.velocity for f in evaluation.flows]),
        pressure=np.stack([f.pressure for f in evaluation.flows]),
    )
    monkeypatch.setattr(
        driver, "load_problem", lambda cfg: (problem, {"baseline_sha256": "fixture"})
    )
    monkeypatch.setattr(driver, "desired_temperature", lambda *args: desired)
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restart",
            "--record",
            str(source),
            "--position",
            "0",
            "--minimum-radius-K",
            "1e-10",
            "--outer-cap",
            "1",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        driver.main()
    assert stopped.value.code == 2
    report = json.loads((output / "record.json").read_text())
    assert report["status"] == "nonlinear_iteration_cap"
    assert not report["gpu_gate_passed"]
    assert report["retained_kkt_difference"] < 1e-14
    with np.load(output / "result.npz", allow_pickle=False) as fields:
        returned = problem.reassemble(
            fields["state"],
            driver.RestoredEvaluation(
                fields["state"], fields["velocity"], fields["pressure"]
            ).flows,
        )
        value, gradient = problem.objective_gradient(returned, desired)
        kkt, _ = driver.optimality(problem, returned, desired, gradient, -0.3, 0.3)
        np.testing.assert_allclose(fields["gradient"], gradient)
    assert report["objective"] == pytest.approx(value)
    assert report["kkt"] == pytest.approx(kkt)
    checkpoint = RecoveryStore(output / "recovery", report["checkpoint_identity"]).load()
    assert checkpoint["minimum_radius_K"] == 1e-10
    assert checkpoint["history"]
