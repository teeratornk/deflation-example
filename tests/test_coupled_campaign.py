"""Staged execution reproduces the single-process sequence and assembles verifiably."""

import json

import numpy as np
from omegaconf import OmegaConf
import pytest

from deflation_example import coupled_campaign as campaign
from deflation_example import coupled_sequence as sequence
from deflation_example.coupled_optimizer import minimize_coupled
from deflation_example.coupled_report import summarize, validate_record
from deflation_example.study_solvers import ArrayReference, StudySolver
from test_coupled_derivatives import small_coupled_problem
from test_coupled_optimizer import solver as cpu_solver
from test_coupled_sequence import configuration


class Sampler:
    def start(self):
        return self

    def finish(self):
        return {"complete": True, "peak_host_rss_bytes": 1, "peak_gpu_process_bytes": 0}


def prepared(monkeypatch, problem):
    monkeypatch.setattr(
        sequence,
        "load_problem",
        lambda cfg: (
            problem,
            {"baseline_sha256": "test", "configuration": {}, "input_sha256": {}, "seconds": 0.25},
        ),
    )
    monkeypatch.setattr(
        sequence,
        "configured_reference",
        lambda *args: ArrayReference(np.eye(problem.size)[:, :3], {"construction": "test"}),
    )
    monkeypatch.setattr(
        sequence,
        "prepare_device",
        lambda *args: (Sampler(), lambda: None, sequence.StudySolver, {}),
    )


def base_config(problem, method, output):
    config = configuration(problem)
    config.update(
        method=method,
        device="cpu",
        memory_interval=0.01,
        repetition=0,
        rank=3,
        recycle_window=6,
        inner_tolerance=1e-11,
        inner_cap=1000,
        threads=1,
        output=str(output),
    )
    return config


def run_stage(problem, method, output, positions=None, restore=None, resume=None):
    config = base_config(problem, method, output)
    config["stage"] = {
        "positions": positions,
        "restore": None if restore is None else str(restore),
        "resume": None if resume is None else str(resume),
    }
    return sequence.run(OmegaConf.create(config))


@pytest.mark.parametrize("method", ["jacobi", "recycling"])
def test_staged_chain_reproduces_the_single_process_sequence(monkeypatch, tmp_path, method):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    prepared(monkeypatch, problem)
    whole = sequence.run(OmegaConf.create(base_config(problem, method, tmp_path / "whole")))
    assert whole["all_problems_verified"] and whole["schema"] == sequence.COMPLETE_SCHEMA
    first = run_stage(problem, method, tmp_path / "stage-0", positions=[0])
    assert first["schema"] == sequence.STAGE_SCHEMA and first["all_problems_verified"]
    assert first["stage"]["final"]["warm_start_valid"]
    assert first["stage"]["checkpoints"]["count"] >= 1
    assert (tmp_path / "stage-0/checkpoint-latest.npz").is_file()
    second = run_stage(
        problem, method, tmp_path / "stage-1", positions=[1], restore=tmp_path / "stage-0"
    )
    assert second["all_problems_verified"]
    assert second["cases"][0]["position"] == 1 and second["cases"][0]["warm_start_used"]
    assert second["stage"]["restore"]["arrays_sha256"] == first["stage"]["final"]["sha256"]
    assert (tmp_path / "stage-1/target-01.npz").is_file()

    assembled = campaign.assemble([tmp_path / "stage-0", tmp_path / "stage-1"])
    assert validate_record(assembled)
    assert assembled["status"] == "complete" and assembled["all_problems_verified"]
    assert [row["position"] for row in assembled["cases"]] == [0, 1]
    for expected, actual in zip(whole["cases"], assembled["cases"], strict=True):
        assert actual["objective"] == expected["objective"]
        assert actual["nonlinear_iterations"] == expected["nonlinear_iterations"]
        assert actual["inner_iterations"] == expected["inner_iterations"]
        assert actual["kkt"] == expected["kkt"]
    with (
        np.load(tmp_path / "whole/target-01.npz") as a,
        np.load(tmp_path / "stage-1/target-01.npz") as b,
    ):
        assert np.array_equal(a["state"], b["state"])
        assert np.array_equal(a["control"], b["control"])
    assert assembled["sequence_seconds"] == pytest.approx(
        first["sequence_seconds"] + second["sequence_seconds"]
    )
    assert sum(assembled["components_seconds"].values()) == pytest.approx(
        assembled["sequence_seconds"]
    )
    assert assembled["assembly"]["policy"] == "staged-v1"
    assert assembled["assembly"]["restart_count"] == 1
    assert (
        assembled["assembly"]["restart_overhead_seconds"] >= second["process_preparation_seconds"]
    )
    summary = summarize([assembled], repetitions=1, population=["jacobi", "reference"])
    assert summary["headline_population"] == ["jacobi", "reference"]
    assert not summary["all_declared_sequences_verified"]
    with pytest.raises(ValueError):
        summarize([assembled, whole], repetitions=2)


def test_restore_rejects_tampered_or_mismatched_predecessors(monkeypatch, tmp_path):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    prepared(monkeypatch, problem)
    run_stage(problem, "jacobi", tmp_path / "stage-0", positions=[0])
    with np.load(tmp_path / "stage-0/stage-final.npz") as data:
        arrays = {key: data[key] for key in data.files}
    arrays["state"] = arrays["state"] + 1e-3
    np.savez(tmp_path / "stage-0/stage-final.npz", **arrays)
    with pytest.raises(ValueError, match="checksum"):
        run_stage(
            problem, "jacobi", tmp_path / "stage-1", positions=[1], restore=tmp_path / "stage-0"
        )
    with pytest.raises(ValueError):
        run_stage(problem, "jacobi", tmp_path / "bad", positions=[0], restore=tmp_path / "stage-0")
    with pytest.raises(ValueError):
        run_stage(problem, "jacobi", tmp_path / "bad2", positions=[1, 0])


def test_checkpoint_hook_is_neutral_and_resume_finishes_the_same_problem():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    desired = np.linspace(-0.5, 0.8, problem.size)
    lower, upper = -0.04, 0.15
    settings = dict(tolerance=1e-8, max_iterations=60, qp_tolerance=1e-11, secant_memory=3)
    plain = minimize_coupled(problem, desired, lower, upper, cpu_solver(), **settings)
    checkpoints = []
    hooked = minimize_coupled(
        problem, desired, lower, upper, cpu_solver(), checkpoint=checkpoints.append, **settings
    )
    assert plain.status == hooked.status == "converged"
    assert plain.objective == hooked.objective
    assert len(plain.history) == len(hooked.history) == len(checkpoints) >= 2
    assert [row["iteration"] for row in checkpoints] == list(range(len(checkpoints)))
    assert len(checkpoints[0]["history"]) == 1
    payload = checkpoints[0]
    resume = {
        "state": payload["state"],
        "initial_evaluation": sequence.RestoredEvaluation(
            payload["state"],
            np.stack([f.velocity for f in payload["flows"]]),
            np.stack([f.pressure for f in payload["flows"]]),
        ),
        "history": payload["history"],
        "damping": payload["damping"],
        "secants": payload["secants"],
        "iteration": payload["iteration"],
    }
    resumed = minimize_coupled(
        problem, desired, lower, upper, cpu_solver(), resume=resume, **settings
    )
    assert resumed.status == "converged"
    assert resumed.objective == pytest.approx(plain.objective, abs=1e-10)
    assert len(resumed.history) == len(plain.history)
    assert resumed.history[0]["iteration"] == 0 and resumed.history[1]["iteration"] == 1
    with pytest.raises(ValueError):
        minimize_coupled(problem, desired, lower, upper, cpu_solver(), checkpoint=3, **settings)


def test_interrupted_stage_resumes_from_its_checkpoint_and_assembles(monkeypatch, tmp_path):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    prepared(monkeypatch, problem)
    chain = tmp_path / "runs" / "recycling" / "rep-0"
    first = run_stage(problem, "recycling", chain / "stage-0", positions=[0])
    assert first["all_problems_verified"]
    original = sequence.minimize_coupled

    def interrupted(*args, **kwargs):
        hook = kwargs["checkpoint"]

        def once(payload):
            hook(payload)
            raise RuntimeError("simulated wall-time termination")

        return original(*args, **{**kwargs, "checkpoint": once})

    monkeypatch.setattr(sequence, "minimize_coupled", interrupted)
    attempt = run_stage(
        problem, "recycling", chain / "stage-1", positions=[1], restore=chain / "stage-0"
    )
    assert not attempt["all_problems_verified"]
    assert attempt["cases"][0]["status"] == "numerical_error"
    meta = json.loads((chain / "stage-1/checkpoint-latest.json").read_text())
    assert meta["iteration"] == 0 and meta["position"] == 1
    monkeypatch.setattr(sequence, "minimize_coupled", original)
    resumed = run_stage(
        problem,
        "recycling",
        chain / "stage-1-resume-0",
        positions=[1],
        restore=chain / "stage-0",
        resume=chain / "stage-1",
    )
    assert resumed["all_problems_verified"], resumed["cases"][0]["status"]
    case = resumed["cases"][0]
    assert case["resumed_from_iteration"] == 0
    assert case["history"][0]["iteration"] == 0 and case["history"][1]["iteration"] == 1
    assert resumed["components_seconds"]["prior_attempt_work"] == pytest.approx(
        meta["stage_elapsed_seconds"]
    )
    assert resumed["stage"]["resume"]["iteration"] == 0
    assembled = campaign.assemble(
        [chain / "stage-0", chain / "stage-1-resume-0"], [chain / "stage-1"]
    )
    assert validate_record(assembled) and assembled["all_problems_verified"]
    assert assembled["assembly"]["restart_count"] == 2
    assert assembled["assembly"]["interrupted_attempts"][0]["last_checkpoint"]["iteration"] == 0
    status = {"attempts": [{"directory": str(chain / "stage-1"), "elapsed_seconds": 1e4}]}
    with_status = campaign.assemble(
        [chain / "stage-0", chain / "stage-1-resume-0"], [chain / "stage-1"], status
    )
    assert with_status["assembly"]["discarded_seconds"] == pytest.approx(
        1e4 - meta["stage_elapsed_seconds"]
    )
    chains = campaign.discover_chains(tmp_path / "runs")
    assert set(chains) == {("recycling", 0)}
    assert chains[("recycling", 0)]["stages"] == [chain / "stage-0", chain / "stage-1-resume-0"]
    assert chains[("recycling", 0)]["attempts"] == [chain / "stage-1"]


def test_assembler_rejects_broken_chains(monkeypatch, tmp_path):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    prepared(monkeypatch, problem)
    run_stage(problem, "jacobi", tmp_path / "stage-0", positions=[0])
    run_stage(problem, "jacobi", tmp_path / "stage-1", positions=[1], restore=tmp_path / "stage-0")
    with pytest.raises(ValueError):
        campaign.assemble([tmp_path / "stage-1"])
    with pytest.raises(ValueError):
        campaign.assemble([tmp_path / "stage-1", tmp_path / "stage-0"])
    record = json.loads((tmp_path / "stage-1/record.json").read_text())
    record["stage"]["restore"]["arrays_sha256"] = "0" * 64
    (tmp_path / "stage-1/record.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="final arrays"):
        campaign.assemble([tmp_path / "stage-0", tmp_path / "stage-1"])
    partial = campaign.assemble([tmp_path / "stage-0"])
    assert partial["status"] == "incomplete_chain" and not partial["all_problems_verified"]


def test_recycling_history_round_trip_and_validation():
    from deflation_example.coupled_derivatives import GaussNewtonOperator

    problem = small_coupled_problem([0.2, 0.35])
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
    diagonal = problem.preconditioning_diagonal(evaluation)
    indices = np.arange(problem.size)
    B = H.restrict(indices)
    B.diagonal = lambda: diagonal
    rng = np.random.default_rng(5)
    first = cpu_solver("recycling")
    result, _ = first.solve(B, B @ rng.normal(size=problem.size), indices)
    assert result.status == "converged"
    exported = first.export_history()
    assert exported["basis"].shape[1] == 3 and exported["previous"] is not None
    second = cpu_solver("recycling")
    second.import_history(exported)
    again = second.export_history()
    assert np.array_equal(again["basis"], exported["basis"])
    assert np.array_equal(again["indices"], exported["indices"])
    assert StudySolver("jacobi", rank=0).export_history() is None
    with pytest.raises(ValueError):
        second.import_history({"indices": indices[:2], "basis": exported["basis"]})
    with pytest.raises(ValueError):
        second.import_history({"indices": indices, "basis": np.ones((problem.size, 9))})
