"""Complete-sequence warm starts, retained failures and timing partitions."""

import json
import numpy as np
from omegaconf import OmegaConf
import pytest

from deflation_example import coupled_sequence as sequence
from test_coupled_derivatives import small_coupled_problem
from test_coupled_optimizer import solver


def configuration(problem):
    return {
        "lower_K": problem.temperature_offset - 0.1,
        "queries": [
            {"target": 0, "upper_K": problem.temperature_offset + 0.3},
            {"target": 1, "upper_K": problem.temperature_offset + 0.25},
        ],
        "target_count": 2,
        "warm_start": True,
        "nonlinear_tolerance": 1e-8,
        "nonlinear_cap": 60,
        "qp_tolerance": 1e-11,
        "qp_cap": 100,
        "backtracking": "quadratic",
        "secant_memory": 3,
    }


@pytest.mark.parametrize("fail_first", [False, True])
@pytest.mark.parametrize("startup", [0.0, 60.0])
def test_sequence_uses_only_verified_previous_states(monkeypatch, fail_first, startup):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    config = configuration(problem)
    config["target_startup_s"] = startup
    original = sequence.minimize_coupled
    initial_guesses = []

    def optimize(*args, **kwargs):
        initial_guesses.append((kwargs["initial"], kwargs["initial_evaluation"]))
        result = original(*args, **kwargs)
        if fail_first and len(initial_guesses) == 1:
            result.status = "nonlinear_iteration_cap"
        return result

    monkeypatch.setattr(sequence, "minimize_coupled", optimize)
    cases, fields = sequence.optimize_targets(problem, solver(), config)
    assert len(cases) == len(fields) == 2
    assert cases[1]["verified"], cases[1]
    assert initial_guesses[0] == (None, None)
    if fail_first:
        assert initial_guesses[1] == (None, None)
        assert not cases[0]["verified"]
        assert cases[0]["status"] == "nonlinear_iteration_cap"
    else:
        assert all(case["verified"] for case in cases)
        np.testing.assert_array_equal(initial_guesses[1][0], fields[0]["state"])
        np.testing.assert_array_equal(initial_guesses[1][1].state, fields[0]["state"])
    assert cases[1]["warm_start_used"] != fail_first
    from deflation_example.coupled_targets import desired_temperature

    for query, field in zip(config["queries"], fields, strict=True):
        np.testing.assert_array_equal(
            field["desired"], desired_temperature(problem, query["target"], 2, startup)
        )


def test_runtime_failure_keeps_the_remaining_declared_targets(monkeypatch):
    problem = small_coupled_problem(uniform_capacity=True)

    def failure(*args, **kwargs):
        raise MemoryError("test allocation failure")

    monkeypatch.setattr(sequence, "minimize_coupled", failure)
    cases, fields = sequence.optimize_targets(problem, solver(), configuration(problem))
    assert [case["status"] for case in cases] == ["memory_limited", "not_run_after_numerical_error"]
    assert not any(case["verified"] for case in cases)
    assert len(fields) == 2


def test_complete_sequence_outputs_partition_and_original_state_checks(monkeypatch, tmp_path):
    problem = small_coupled_problem(uniform_capacity=True)
    config = configuration(problem)
    config.update(
        method="jacobi",
        device="cpu",
        memory_interval=0.01,
        repetition=0,
        rank=0,
        recycle_window=3,
        inner_tolerance=1e-11,
        inner_cap=1000,
        threads=1,
        evaluation_progress=True,
        linear_progress=True,
        output=str(tmp_path / "sequence"),
    )
    monkeypatch.setattr(
        sequence,
        "load_problem",
        lambda cfg: (
            problem,
            {
                "baseline_sha256": "test",
                "configuration": {},
                "input_sha256": {},
                "seconds": 0.25,
            },
        ),
    )

    class Sampler:
        def start(self):
            return self

        def finish(self):
            return {"test_sampler": True}

    monkeypatch.setattr(
        sequence,
        "prepare_device",
        lambda *args: (Sampler(), lambda: None, sequence.StudySolver, {}),
    )
    report = sequence.run(OmegaConf.create(config))
    assert report["status"] == "complete"
    assert report["verified_problems"] == 2
    from deflation_example.coupled_report import validate_record

    assert validate_record(report)
    assert sum(report["components_seconds"].values()) == pytest.approx(report["sequence_seconds"])
    assert report["preparation_inclusive_seconds"] == pytest.approx(
        report["sequence_seconds"] + report["process_preparation_seconds"] + 0.25
    )
    assert "output" not in report["configuration"]
    stored = json.loads((tmp_path / "sequence/record.json").read_text())
    assert stored["cases"][1]["warm_start_used"]
    assert (tmp_path / "sequence/target-00.npz").is_file()
    progress = json.loads((tmp_path / "sequence/optimization-progress.json").read_text())
    assert progress["position"] == 1
    assert progress["iteration"] >= 0
    linear = json.loads((tmp_path / "sequence/linear-progress.json").read_text())
    assert linear["status"] == "converged"
    assert linear["residual"] <= config["inner_tolerance"]
    with pytest.raises(FileExistsError):
        sequence.run(OmegaConf.create(config))
