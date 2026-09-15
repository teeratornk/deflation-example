"""Read-only audit of fac3085; desired contracts, including known failing cases.

Run on a compute node with the frozen source and its tests on PYTHONPATH.
All simulated records and checkpoints are written only below pytest's tmp_path.
These tests do not modify a running job, source checkout, or measured record.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example import coupled_campaign as campaign
from deflation_example import coupled_presentation as presentation
from deflation_example import coupled_sequence as sequence
from deflation_example.coupled_report import summarize
from deflation_example.reporting import write_report
from test_coupled_campaign import prepared, run_stage
from test_coupled_derivatives import small_coupled_problem
from test_coupled_report import record


def test_hybrid_block_total_does_not_count_enclosed_upload_twice(monkeypatch):
    from deflation_example import coupled_hybrid_solver as hybrid
    from deflation_example.study_solvers import ArrayReference
    from test_coupled_hybrid_solver import system

    problem, operator, diagonal = system()

    class FakeDeviceJacobian:
        def __init__(self, jacobian):
            self.cp = SimpleNamespace(
                asnumpy=np.asarray,
                cuda=SimpleNamespace(
                    get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)
                ),
            )

        def close(self):
            pass

    monkeypatch.setattr(hybrid, "CudaControlJacobian", FakeDeviceJacobian)
    monkeypatch.setattr(
        hybrid,
        "CudaGaussNewton",
        lambda *args, **kwargs: SimpleNamespace(
            restrict=lambda indices: operator.restrict(indices).__matmul__
        ),
    )
    indices = np.arange(problem.size)
    matrix = operator.restrict(indices)
    matrix.diagonal = lambda: diagonal
    rng = np.random.default_rng(12)
    solver = hybrid.HybridCoupledSolver(
        "reference",
        reference=ArrayReference(rng.normal(size=(problem.size, 3)), {}),
        rank=3,
        block_min_columns=2,
        coarse_device="cpu",
        rtol=1e-10,
    )
    try:
        _, timing = solver.solve(matrix, matrix @ rng.normal(size=problem.size), indices)
    finally:
        solver.close()
    log = timing["hybrid_block_processing"]
    assert sum(call["kind"] == "factor_upload" for call in log["calls"]) == 1
    # In this CPU-coarse path each upload occurs wholly inside an operator_block.
    enclosing = sum(call["seconds"] for call in log["calls"] if call["kind"] == "operator_block")
    assert log["seconds"] == pytest.approx(enclosing, rel=0, abs=1e-12)


def test_submission_retry_preserves_resume_checkpoint():
    from deflation_example.study_submission import submit_pending

    entry = {"resume_from": "checkpoint-a", "live": None}

    def fail(resume):
        assert resume == "checkpoint-a"
        raise RuntimeError("submission limit")

    with pytest.raises(RuntimeError, match="submission limit"):
        submit_pending(entry, fail)
    assert entry == {"resume_from": "checkpoint-a", "live": None}
    submit_pending(entry, lambda resume: {"job": "123", "resume": resume})
    assert "resume_from" not in entry
    assert entry["live"]["resume"] == "checkpoint-a"


def test_two_interrupted_attempts_count_local_discarded_work():
    # Each process runs 100 s and checkpoints at its local 90 s. The second
    # checkpoint's stage counter includes the first attempt's retained 90 s.
    attempts = [
        {"directory": "a", "last_checkpoint": {"stage_elapsed_seconds": 90.0}},
        {"directory": "b", "last_checkpoint": {"stage_elapsed_seconds": 180.0}},
    ]
    scheduler = {
        "attempts": [
            {"directory": "a", "elapsed_seconds": 100.0},
            {"directory": "b", "elapsed_seconds": 100.0},
        ]
    }
    assert campaign.discarded_seconds(attempts, scheduler) == pytest.approx(20.0)


def test_interrupted_checkpoint_write_preserves_previous_restart(monkeypatch, tmp_path):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    prepared(monkeypatch, problem)
    writer = sequence.write_report
    count = 0

    def interrupt_second_manifest(path, payload):
        nonlocal count
        if path.name == "checkpoint-latest.json":
            count += 1
            if count == 2:
                # Arrays have already been replaced. Simulate termination before
                # the new metadata is committed, after a valid first checkpoint.
                raise RuntimeError("injected interruption between checkpoint files")
        return writer(path, payload)

    monkeypatch.setattr(sequence, "write_report", interrupt_second_manifest)
    root = tmp_path / "stage-0"
    outcome = run_stage(problem, "jacobi", root, positions=[0])
    assert count == 2
    assert not outcome["all_problems_verified"]
    meta = json.loads((root / "checkpoint-latest.json").read_text())
    assert meta["iteration"] == 0
    restored = sequence.load_resume(root, 0, meta["configuration_sha256"], "jacobi")
    assert restored["iteration"] == 0


@pytest.mark.parametrize("changed_identity", ["baseline", "source"])
def test_resume_rejects_different_numerical_identity(monkeypatch, tmp_path, changed_identity):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    prepared(monkeypatch, problem)
    source = tmp_path / "stage-0"
    first = run_stage(problem, "jacobi", source, positions=[0])
    assert first["all_problems_verified"]
    if changed_identity == "baseline":
        loader = sequence.load_problem

        def different_baseline(cfg):
            model, baseline = loader(cfg)
            return model, {**baseline, "baseline_sha256": "different-baseline"}

        monkeypatch.setattr(sequence, "load_problem", different_baseline)
    else:
        environment = sequence.environment

        def different_source():
            return {**environment(), "source_sha256": {"solver.py": "different-source"}}

        monkeypatch.setattr(sequence, "environment", different_source)
    try:
        resumed = run_stage(
            problem, "jacobi", tmp_path / "stage-0-resume-0", positions=[0], resume=source
        )
    except ValueError:
        return
    assert not resumed["all_problems_verified"], "Mismatched checkpoint identity was accepted"


def test_report_wall_ratio_uses_elapsed_time(monkeypatch, tmp_path):
    # Deliberately different staged and elapsed rankings. This is synthetic
    # bookkeeping evidence, never a measured optimization comparison.
    data = [record("jacobi", 0, 20.0), record("reference", 0, 10.0)]
    for row, elapsed, overhead in zip(data, [20, 100], [0.0, 50.0], strict=True):
        row["assembly"] = {
            "policy": "staged-v1",
            "started_utc": "2026-09-15T00:00:00+00:00",
            "finished_utc": f"2026-09-15T00:{elapsed // 60:02d}:{elapsed % 60:02d}+00:00",
            "restart_count": 1,
            "restart_overhead_seconds": overhead,
            "discarded_seconds": 0.0,
        }
        row["cases"][0]["seconds"] = row["sequence_seconds"]
        row["cases"][0]["history"] = [
            {
                "evaluation_seconds": 0.1,
                "attempts": [
                    {
                        "trials": [],
                        "qp_history": [{"linear_iterations": 1, "timing": {"total_seconds": 0.1}}],
                    }
                ],
            }
        ]
    summary = summarize(data, repetitions=1, population=["jacobi", "reference"])
    root = tmp_path / "summary"
    for row in data:
        destination = root / "assembled" / f"{row['configuration']['method']}-rep-0"
        destination.mkdir(parents=True)
        write_report(destination / "record.json", row)
    write_report(root / "summary.json", {"summary": summary, "chains": []})
    built = presentation.build(root, tmp_path / "tables")
    assert built["methods"]["reference"]["elapsed_wall_seconds"] == [100.0]
    assert built["ratios"]["wall_ratio"] == pytest.approx(20.0 / 100.0)
