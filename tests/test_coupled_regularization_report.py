"""The regularization gate keeps cost, accuracy and population requirements separate."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from deflation_example.coupled_regularization import ALPHAS, RANKS, TARGETS
from deflation_example.coupled_regularization_report import (
    KKT_COMPONENTS,
    summarize,
    summarize_alpha,
)
from deflation_example.reporting import file_sha256, write_arrays, write_report


def make_record(directory, alpha, *, construction=20.0, times=(100, 90, 85, 80)):
    directory.mkdir()
    protocol = json.loads(
        (Path(__file__).parents[1] / "examples/coupled_regularization/protocol.json").read_text()
    )
    cfg = {**protocol, "alpha": alpha, "queries": [{"target": 7, "upper_K": 357.3}]}
    record = {
        "schema": "coupled-regularization-screen-v1",
        "alpha": alpha,
        "configuration": cfg,
        "trace_sha256": "same-trace",
        "initial_trajectory_sha256": "same-initial-state",
        "environment": {
            "git_head": "same-source",
            "source_tree_clean": True,
            "cpu_model": "same-cpu",
        },
        "status": "complete",
        "ranks": list(RANKS),
        "targets": list(TARGETS),
        "repetitions": 3,
        "state_dofs": 4,
        "reference": {"construction_seconds": construction},
        "sequences": [],
    }
    for rank, seconds in zip(RANKS, times, strict=True):
        for rep in range(3):
            cases = []
            for target in TARGETS:
                history = [
                    {
                        "linear_status": "converged",
                        "linear_residual": 1e-12,
                        "linear_iterations": 12,
                        "deployed_rank": rank,
                        "fallback": None,
                    }
                ]
                cases.append(
                    {
                        "target": target,
                        "verified": True,
                        "status": "converged",
                        "kkt": dict.fromkeys(KKT_COMPONENTS, 0.0),
                        "history": history,
                        "outer_pdas_steps": 1,
                        "inner_iterations": 12,
                        "deployed_ranks": [rank],
                        "fallbacks": [None],
                        "maximum_original_relative_residual": 1e-12,
                        "quadratic_objective": -2.0,
                    }
                )
            path = directory / f"r{rank}-rep{rep}.npz"
            write_arrays(path, increments=np.zeros((3, 4)))
            setup = construction if rank else 0.0
            record["sequences"].append(
                {
                    "rank": rank,
                    "repetition": rep,
                    "cases": cases,
                    "verified": True,
                    "online_seconds": seconds,
                    "construction_seconds_once": setup,
                    "setup_inclusive_model_seconds": seconds + setup,
                    "arrays": {"file": path.name, "sha256": file_sha256(path)},
                }
            )
    write_report(directory / "record.json", record)
    return record


def update(directory, record):
    write_report(directory / "record.json", record)


def test_selection_waits_for_all_alphas_and_uses_frozen_rule(tmp_path):
    runs = [tmp_path / str(i) for i in range(4)]
    for path, alpha in zip(runs, ALPHAS, strict=True):
        make_record(path, alpha)
    partial = summarize(runs[:3])
    assert partial["selected"] is None
    assert partial["decision"] == "await_declared_population"
    full = summarize(runs)
    assert full["selected"]["alpha"] == ALPHAS[0]
    assert full["selected"]["rank"] == 16
    assert full["selected"]["constant_cost_break_even_three_quadratic_sequences"] == 1
    assert full["decision"] == "evaluate_one_complete_optimization_comparison"


@pytest.mark.parametrize("kind", ["construction", "online"])
def test_iteration_improvement_alone_does_not_pass(tmp_path, kind):
    record = make_record(
        tmp_path / "run",
        ALPHAS[0],
        construction=1000 if kind == "construction" else 20,
        times=(100, 104, 105, 110) if kind == "online" else (100, 90, 85, 80),
    )
    for seq in record["sequences"]:
        if seq["rank"]:
            for case in seq["cases"]:
                case["inner_iterations"] = case["history"][0]["linear_iterations"] = 2
    update(tmp_path / "run", record)
    summary, _ = summarize_alpha(tmp_path / "run")
    assert all(row["eligible_accuracy"] for row in summary["rows"])
    assert not any(row["gate_passed"] for row in summary["rows"])


@pytest.mark.parametrize(
    "defect", ["residual", "kkt", "status", "null_objective", "history", "missing", "rank_zero"]
)
def test_accuracy_coverage_and_nonzero_deployment_are_required(tmp_path, defect):
    path = tmp_path / "run"
    record = make_record(path, ALPHAS[0])
    seq = next(s for s in record["sequences"] if s["rank"] == 4)
    case = seq["cases"][0]
    if defect == "residual":
        case["history"][0]["linear_residual"] = 1e-7
    elif defect == "kkt":
        case["kkt"]["primal_absolute"] = 1e-7
    elif defect == "status":
        case["status"] = "iteration_limit"
    elif defect == "null_objective":
        case["quadratic_objective"] = None
    elif defect == "history":
        case["history"] = []
    elif defect == "missing":
        record["sequences"].remove(seq)
    elif defect == "rank_zero":
        for s in record["sequences"]:
            if s["rank"] == 4:
                for c in s["cases"]:
                    c["deployed_ranks"] = [0]
                    c["history"][0]["deployed_rank"] = 0
    update(path, record)
    group, _ = summarize_alpha(path)
    row = group["rows"][1]
    assert not row["eligible_accuracy"]
    assert not row["gate_passed"]


def test_verified_states_must_agree_with_control(tmp_path):
    path = tmp_path / "run"
    record = make_record(path, ALPHAS[0])
    seq = next(s for s in record["sequences"] if s["rank"] == 4)
    archive = path / seq["arrays"]["file"]
    write_arrays(archive, increments=np.full((3, 4), 0.01))
    seq["arrays"]["sha256"] = file_sha256(archive)
    update(path, record)
    summary, _ = summarize_alpha(path)
    assert "increment_disagreement" in summary["rows"][1]["reasons"]


@pytest.mark.parametrize("defect", ["checksum", "duplicate", "target", "timing", "configuration"])
def test_inconsistent_evidence_is_rejected(tmp_path, defect):
    path = tmp_path / "run"
    record = make_record(path, ALPHAS[0])
    seq = record["sequences"][0]
    if defect == "checksum":
        seq["arrays"]["sha256"] = "bad"
    elif defect == "duplicate":
        record["sequences"].append(copy.deepcopy(seq))
    elif defect == "target":
        seq["cases"][0]["target"] = 3
    elif defect == "timing":
        seq["setup_inclusive_model_seconds"] += 100
    elif defect == "configuration":
        record["configuration"]["slabs"] = 32
    update(path, record)
    with pytest.raises(ValueError):
        summarize_alpha(path)


def test_failed_and_running_alpha_attempts_remain_visible(tmp_path):
    runs = [tmp_path / str(i) for i in range(4)]
    for path, alpha in zip(runs, ALPHAS, strict=True):
        record = make_record(path, alpha)
    record["status"] = "running"
    update(runs[-1], record)
    assert summarize(runs)["selected"] is None
    record["status"] = "screen_error"
    record["error_type"] = "ValueError"
    record["error_message"] = "Retained failed attempt"
    record["sequences"] = []
    record.pop("reference")
    update(runs[-1], record)
    result = summarize(runs)
    assert len(result["groups"]) == 4
    assert result["groups"][-1]["error_type"] == "ValueError"
    assert result["selected"] is not None


def test_source_or_hardware_changes_are_not_pooled(tmp_path):
    paths = [tmp_path / "a", tmp_path / "b"]
    make_record(paths[0], ALPHAS[0])
    record = make_record(paths[1], ALPHAS[1])
    record["environment"]["cpu_model"] = "different"
    update(paths[1], record)
    with pytest.raises(ValueError, match="hardware"):
        summarize(paths)


def test_initially_optimal_quadratic_needs_no_inner_solve(tmp_path):
    path = tmp_path / "run"
    record = make_record(path, ALPHAS[0])
    for seq in record["sequences"]:
        for case in seq["cases"]:
            case.update(
                history=[],
                outer_pdas_steps=0,
                inner_iterations=0,
                deployed_ranks=[],
                fallbacks=[],
                maximum_original_relative_residual=0.0,
            )
    update(path, record)
    result, _ = summarize_alpha(path)
    assert result["rows"][0]["eligible_accuracy"]
    assert all("no_nonzero_reference_deployment" in r["reasons"] for r in result["rows"][1:])


def test_plot_retains_all_ranks_and_incomplete_populations(tmp_path):
    pytest.importorskip("matplotlib")
    from deflation_example.coupled_regularization_report import plot

    path = tmp_path / "run"
    make_record(path, ALPHAS[0])
    result = summarize([path])
    plot(result, tmp_path)
    assert (tmp_path / "regularization_cost.png").stat().st_size > 1000
    assert (tmp_path / "regularization_cost.pdf").stat().st_size > 1000
