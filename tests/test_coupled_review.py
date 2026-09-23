"""Review snapshots and diagnostics must retain states, failures and source identity."""

from argparse import Namespace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example import coupled_review as review
from deflation_example.coupled_sequence import CHECKPOINT_SCHEMA, configuration_digest
from deflation_example.reporting import file_sha256, write_arrays, write_report
from test_coupled_derivatives import small_coupled_problem


def source_files(path, configuration=None, arrays=None):
    path.mkdir()
    configuration = configuration or {"method": "jacobi"}
    write_report(path / "record.json", {"configuration": configuration})
    write_arrays(path / "checkpoint-latest.npz", **(arrays or {"state": np.array([1.0])}))
    write_report(
        path / "checkpoint-latest.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "method": configuration["method"],
            "position": 0,
            "configuration_sha256": configuration_digest(configuration),
            "iteration": 0,
            "arrays_sha256": file_sha256(path / "checkpoint-latest.npz"),
            "objective": 0.0,
            "damping": 0.0,
        },
    )


def test_snapshot_is_immutable_and_checks_every_input(tmp_path):
    source, output = tmp_path / "source", tmp_path / "snapshot"
    source_files(source)
    original = file_sha256(source / "checkpoint-latest.npz")
    manifest = review.snapshot(source, output)
    assert manifest["arrays_sha256"] == original
    assert file_sha256(source / "checkpoint-latest.npz") == original
    np.testing.assert_array_equal(review.read_snapshot(output)[2]["state"], [1.0])
    with pytest.raises(FileExistsError):
        review.snapshot(source, output)
    write_report(output / "record.json", {})
    with pytest.raises(ValueError, match="checksum"):
        review.read_snapshot(output)


def test_snapshot_retry_does_not_publish_a_mixed_pair(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "snapshot"
    source_files(source)
    original = review.shutil.copyfile
    count = 0

    def moving_file(src, dst):
        nonlocal count
        count += 1
        if count == 1:
            write_arrays(dst, state=np.array([2.0]))
        else:
            original(src, dst)

    monkeypatch.setattr(review.shutil, "copyfile", moving_file)
    review.snapshot(source, output)
    assert count == 2
    np.testing.assert_array_equal(review.read_snapshot(output)[2]["state"], [1.0])


def test_exhausted_snapshot_has_no_valid_manifest(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "snapshot"
    source_files(source)
    monkeypatch.setattr(
        review.shutil, "copyfile", lambda src, dst: write_arrays(dst, state=np.zeros(1))
    )
    with pytest.raises(RuntimeError, match="checksum-consistent"):
        review.snapshot(source, output, attempts=2)
    assert not (output / "snapshot.json").exists()


@pytest.mark.parametrize(
    "cfg", [{"method": "recycling"}, {"method": "reference", "reference_transfer": "sequential"}]
)
def test_step_replay_rejects_unrestored_transfer_history(tmp_path, cfg):
    source_files(tmp_path / "source", cfg)
    review.snapshot(tmp_path / "source", tmp_path / "snapshot")
    args = Namespace(
        output=tmp_path / "review", snapshot=tmp_path / "snapshot", baseline=tmp_path, mode="step"
    )
    with pytest.raises(ValueError, match="histories are not restored"):
        review.run(args)


def test_stationarity_locations_keep_weight_scaling_and_bound_signs():
    problem = SimpleNamespace(
        weights=np.array([1e-6, 1.0, 2.0]),
        spatial_size=3,
        free=np.arange(3),
        mesh=SimpleNamespace(nodes=np.zeros((3, 2))),
    )
    state, gradient = np.array([0.5, 0.0, 1.0]), np.array([2e-6, 3.0, -4.0])
    rows = review.stationarity_locations(problem, state, gradient, 0, 1, 2)
    assert rows[0]["index"] == 0
    assert rows[0]["normalized_stationarity"] == 1
    assert [r["normalized_stationarity"] for r in rows[1:]] == [0, 0]
    problem.weights[0] = 0
    with pytest.raises(ValueError, match="weights"):
        review.stationarity_locations(problem, state, gradient, 0, 1, 2)


@pytest.mark.parametrize("mode", ["inspect", "step"])
@pytest.mark.parametrize("initial", [False, True])
def test_review_runs_small_complete_problem_and_serializes_checks(
    tmp_path, monkeypatch, mode, initial
):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True, consistent=True)
    state = np.full(problem.size, 0.06)
    evaluation = problem.evaluate(state)
    cfg = dict(
        method="jacobi",
        queries=[dict(target=0, upper_K=problem.temperature_offset + 0.3)],
        lower_K=problem.temperature_offset - 0.1,
        target_count=2,
        target_startup_s=0,
        device="cpu",
        memory_interval=0.01,
        rank=0,
        recycle_window=1,
        inner_tolerance=1e-10,
        inner_cap=1000,
        qp_tolerance=1e-10,
        qp_cap=100,
    )
    arrays = dict(
        state=state,
        velocity=np.stack([f.velocity for f in evaluation.flows]),
        pressure=np.stack([f.pressure for f in evaluation.flows]),
        secant_steps=np.empty((0, problem.size)),
        secant_gradients=np.empty((0, problem.size)),
    )
    source_files(tmp_path / "source", cfg, arrays)
    review.snapshot(tmp_path / "source", tmp_path / "snapshot")
    monkeypatch.setattr(review, "load_problem", lambda cfg: (problem, {"baseline_sha256": "test"}))
    monkeypatch.setattr(review, "desired_temperature", lambda p, *args: np.full(p.size, 0.1))
    args = Namespace(
        output=tmp_path / "review",
        snapshot=tmp_path / "snapshot",
        baseline=tmp_path,
        mode=mode,
        secants="retained",
        initial=initial,
        continuation=False,
    )
    review.run(args)
    report = json.loads((tmp_path / "review/review.json").read_text())
    assert report["status"] == "review_complete"
    assert report["adjoint"]["gradient_relative_difference"] < 1e-10
    if mode == "step":
        assert report["qp"]["status"] == "converged"
        assert len(report["trials"]) == 4
        assert all(row["status"] == "evaluated" for row in report["trials"])
    with np.load(tmp_path / "review/evaluation.npz") as stored:
        expected = np.zeros_like(state) if initial else state
        np.testing.assert_array_equal(stored["state"], expected)
        _, gradient = problem.objective_gradient(
            problem.evaluate(expected), np.full(problem.size, 0.1)
        )
        np.testing.assert_allclose(stored["gradient"], gradient, atol=1e-10)
