"""Continuation preserves source identity, timing scope and every original outcome."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example.coupled_regularization import ALPHAS, TARGETS, solve_quadratic
from deflation_example.coupled_regularization_report import summarize, summarize_alpha
from deflation_example.coupled_regularization_resume import (
    SCHEMA,
    RUNTIME_FIELDS,
    declared_sequences,
    missing_sequences,
    read_population,
    run,
    solve_sequence,
    verify_environment,
)
from deflation_example.reporting import file_sha256, write_arrays, write_report
from deflation_example.study_solvers import StudySolver
from test_coupled_derivatives import small_coupled_problem
from test_coupled_regularization_report import make_record


def source_environment():
    return {
        **dict.fromkeys(RUNTIME_FIELDS, "fixed-runtime"),
        "git_head": "original-source",
        "source_tree_clean": True,
        "source_sha256": dict.fromkeys(
            ["coupled_regularization.py", "coupled_optimizer.py", "study_solvers.py", "solvers.py"],
            "unchanged-hash",
        ),
    }


def original_fixture(path, missing=((4, 2),), alpha=ALPHAS[0]):
    record = make_record(path, alpha)
    record["environment"] = source_environment()
    record["baseline_sha256"] = "baseline"
    record["reference"]["sha256"] = "same-reference"
    removed = [r for r in record["sequences"] if (r["rank"], r["repetition"]) in missing]
    record["sequences"] = [r for r in record["sequences"] if r not in removed]
    record["status"] = "running" if missing else "complete"
    write_report(path / "record.json", record)
    return record, removed


def continuation_fixture(path, original_path, original, sequence):
    path.mkdir()
    row = copy.deepcopy(sequence)
    archive = path / row["arrays"]["file"]
    write_arrays(archive, increments=np.zeros((3, 4)))
    row["arrays"]["sha256"] = file_sha256(archive)
    env = source_environment()
    env["git_head"] = "continuation-source"
    env["source_sha256"]["new_driver.py"] = "new-hash"
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "environment": env,
        "original_record_sha256": file_sha256(original_path / "record.json"),
        **{
            k: original[k]
            for k in (
                "alpha",
                "configuration",
                "baseline_sha256",
                "trace_sha256",
                "initial_trajectory_sha256",
                "state_dofs",
            )
        },
        "reference_sha256": original["reference"]["sha256"],
        "rank": row["rank"],
        "repetition": row["repetition"],
        "restart_preparation_seconds": 12.0,
        "sequences": [row],
    }
    write_report(path / "record.json", result)
    return result


def test_missing_sequence_order_is_the_declared_cyclic_order():
    expected = declared_sequences()
    assert expected == [
        (0, 0),
        (4, 0),
        (8, 0),
        (16, 0),
        (4, 1),
        (8, 1),
        (16, 1),
        (0, 1),
        (8, 2),
        (16, 2),
        (0, 2),
        (4, 2),
    ]
    assert (
        missing_sequences({"sequences": [{"rank": r, "repetition": p} for r, p in expected[:9]]})
        == expected[9:]
    )


@pytest.mark.parametrize("defect", ["runtime", "source", "dirty", "missing_source"])
def test_incompatible_continuation_is_rejected(defect):
    original, current = source_environment(), source_environment()
    if defect == "runtime":
        current["numpy"] = "different"
    elif defect == "source":
        current["source_sha256"]["solvers.py"] = "changed"
    elif defect == "dirty":
        current["source_tree_clean"] = False
    else:
        original["source_sha256"].pop("solvers.py")
    with pytest.raises(ValueError):
        verify_environment(original, current)


def test_new_driver_is_identified_without_relabeling_original_source():
    original, current = source_environment(), source_environment()
    current["git_head"] = "new-source"
    current["source_sha256"]["new_driver.py"] = "new-hash"
    assert verify_environment(original, current)["all_original_modules_identical"]


def test_completed_population_preserves_raw_original_and_records_actual_sources(tmp_path):
    parent, extra = tmp_path / "original", tmp_path / "extra"
    original, removed = original_fixture(parent)
    before = (parent / "record.json").read_bytes()
    continuation_fixture(extra, parent, original, removed[0])
    result, _ = summarize_alpha(parent, [extra])
    assert result["status"] == "complete"
    assert result["original_record_status"] == "running"
    assert result["source"] == "original-source"
    assert result["continuations"][0]["source"] == "continuation-source"
    assert result["continuations"][0]["restart_preparation_seconds"] == 12.0
    assert result["rows"][1]["eligible_accuracy"]
    assert result["rows"][1]["construction_seconds_once"] == 20.0
    assert (parent / "record.json").read_bytes() == before


@pytest.mark.parametrize(
    "defect",
    [
        "parent",
        "physics",
        "reference",
        "replacement",
        "extra_row",
        "source",
        "preparation",
        "checksum",
        "timing",
    ],
)
def test_bad_merged_evidence_is_rejected(tmp_path, defect):
    parent, extra = tmp_path / "original", tmp_path / "extra"
    original, removed = original_fixture(parent)
    addition = continuation_fixture(extra, parent, original, removed[0])
    if defect == "parent":
        addition["original_record_sha256"] = "wrong"
    elif defect == "physics":
        addition["configuration"] = {**addition["configuration"], "slabs": 32}
    elif defect == "reference":
        addition["reference_sha256"] = "changed"
    elif defect == "replacement":
        addition["rank"] = 0
    elif defect == "extra_row":
        addition["sequences"].append(copy.deepcopy(addition["sequences"][0]))
    elif defect == "source":
        addition["environment"]["source_sha256"]["coupled_optimizer.py"] = "changed"
    elif defect == "preparation":
        addition.pop("restart_preparation_seconds")
    elif defect == "checksum":
        addition["sequences"][0]["arrays"]["sha256"] = "wrong"
    elif defect == "timing":
        addition["sequences"][0]["setup_inclusive_model_seconds"] = 0.0
    write_report(extra / "record.json", addition)
    with pytest.raises(ValueError):
        summarize_alpha(parent, [extra])


def test_duplicate_continuations_cannot_select_a_faster_attempt(tmp_path):
    parent = tmp_path / "original"
    original, removed = original_fixture(parent)
    paths = [tmp_path / "a", tmp_path / "b"]
    for path in paths:
        continuation_fixture(path, parent, original, removed[0])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_alpha(parent, paths)


def test_numerical_failure_is_retained_and_disqualifies_the_rank(tmp_path):
    parent, extra = tmp_path / "original", tmp_path / "extra"
    original, removed = original_fixture(parent)
    addition = continuation_fixture(extra, parent, original, removed[0])
    addition["sequences"][0]["cases"][0]["status"] = "linear_maxiter"
    addition["sequences"][0]["cases"][0]["verified"] = False
    addition["sequences"][0]["verified"] = False
    write_report(extra / "record.json", addition)
    result, _ = summarize_alpha(parent, [extra])
    assert result["status"] == "complete"
    assert not result["rows"][1]["eligible_accuracy"]
    assert result["rows"][1]["failed_cases"] == [[2, 7, "linear_maxiter"]]


def test_interrupted_continuation_does_not_complete_the_population(tmp_path):
    parent, extra = tmp_path / "original", tmp_path / "extra"
    original, removed = original_fixture(parent)
    addition = continuation_fixture(extra, parent, original, removed[0])
    addition.update(status="solving_sequence", sequences=[])
    write_report(extra / "record.json", addition)
    result, origins, provenance = read_population(parent, [extra])
    assert result["status"] == "continuation_incomplete"
    assert len(origins) == 11
    assert provenance[0]["status"] == "solving_sequence"


def test_complete_four_alpha_gate_includes_continued_repetitions(tmp_path):
    parents, extras = [], []
    for i, alpha in enumerate(ALPHAS):
        parent, extra = tmp_path / f"original-{i}", tmp_path / f"extra-{i}"
        original, removed = original_fixture(parent, alpha=alpha)
        continuation_fixture(extra, parent, original, removed[0])
        parents.append(parent)
        extras.append(extra)
    result = summarize(parents, extras)
    assert result["all_alpha_attempts_finished"]
    assert result["selected"]["rank"] == 16
    with pytest.raises(ValueError, match="supplied original"):
        summarize(parents[:-1], extras)


def test_existing_result_is_rejected_before_any_new_output(tmp_path):
    original, _ = original_fixture(tmp_path / "original")
    args = SimpleNamespace(
        original=tmp_path / "original", rank=0, repetition=0, output=tmp_path / "new"
    )
    with pytest.raises(ValueError, match="unrecorded"):
        run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("rank", [0, 4])
def test_three_target_driver_matches_original_numerical_solves(rank):
    problem = small_coupled_problem([0.2, 0.35], consistent=True, streamline_rule="smooth_p8")
    evaluation = problem.evaluate(np.linspace(0.04, 0.1, problem.size))
    # Map the fixed Kelvin bounds to a convenient interval for this small algebraic check.
    problem.temperature_offset = 337.3
    targets = [np.full(problem.size, value) for value in (0.12, 0.16, 0.2)]
    basis = np.eye(problem.size)
    cfg = {
        "inner_tolerance": 1e-10,
        "qp_tolerance": 1e-10,
        "qp_cap": 100,
        "inner_cap": 50000,
        "inner_refresh": 50000,
    }
    row, states = solve_sequence(problem, evaluation, targets, basis, rank, 2, cfg, 2.5)
    assert row["verified"]
    assert [c["target"] for c in row["cases"]] == list(TARGETS)
    assert row["construction_seconds_once"] == (2.5 if rank else 0)
    assert row["setup_inclusive_model_seconds"] == row["online_seconds"] + (2.5 if rank else 0)
    from deflation_example.study_solvers import ArrayReference

    solver = StudySolver(
        "reference" if rank else "jacobi",
        rank=rank,
        reference=ArrayReference(basis[:, :rank], {}) if rank else None,
        rtol=1e-10,
        cg_factor=0.1,
        maxiter=50000,
        refresh=50000,
        residual_policy="refine",
    )
    for desired, state, measured in zip(targets, states, row["cases"], strict=True):
        expected, check = solve_quadratic(problem, evaluation, desired, 0.0, 10.0, solver, cfg)
        np.testing.assert_allclose(state, expected, atol=1e-12, rtol=0)
        assert measured["inner_iterations"] == check["inner_iterations"]
        assert measured["deployed_ranks"] == check["deployed_ranks"]
    solver.close()
