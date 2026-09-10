"""Study protocol, compact traces and complete trajectory acceptance."""

import base64
import json
import zlib

import numpy as np
import pytest
from omegaconf import OmegaConf

from deflation_example import benchmark_cht
from deflation_example.memory import ProcessMemory


def small_config(problem="steady"):
    config = OmegaConf.structured(benchmark_cht.StudyConfig)
    config.device = "cpu"
    config.methods = ["jacobi", "reference", "recycling"]
    config.problem = problem
    config.n = config.calibration_grid = 3
    config.slabs, config.targets, config.rank, config.window = 3, 3, 5, 8
    config.bound = 0.0001
    return config


def test_query_parameter_is_separate_from_physical_time():
    protocol = benchmark_cht.specification(small_config("transient"))
    assert [row["theta"] for row in protocol["targets"]] == [0.0, 0.5, 1.0]
    assert all("t" not in row for row in protocol["targets"])
    assert protocol["controls"]["horizon"] == 0.1


@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_all_active_initialization_releases_constraints(monkeypatch, method):
    monkeypatch.setattr(
        benchmark_cht,
        "ProcessMemory",
        lambda *args: ProcessMemory(interval=60, query=lambda: (100, 0)),
    )
    config = small_config("transient")
    config.initial_active = "all"
    protocol = benchmark_cht.specification(config)
    record = benchmark_cht.complete_sequence(protocol, method, "outer_inner")
    assert record["success"]
    assert record["cases"][0]["initial_active_count"] == 81
    assert record["cases"][0]["final_active_count"] < 81


def test_summary_checks_semantic_acceptance_and_timing(monkeypatch):
    from copy import deepcopy
    from deflation_example.benchmark_cht_report import validate_sequence, summarize

    monkeypatch.setattr(
        benchmark_cht,
        "ProcessMemory",
        lambda *args: ProcessMemory(interval=60, query=lambda: (100, 0)),
    )
    protocol = benchmark_cht.specification(small_config("transient"))
    record = benchmark_cht.complete_sequence(protocol, "reference", "outer_inner")
    validate_sequence(record, protocol)
    assert summarize(protocol, [record])[1]["accepted_sequences"] == 1
    bad = deepcopy(record)
    bad["cases"][0]["inner"][0]["acceptance_rtol"] *= 10
    with pytest.raises(ValueError, match="Final residual target"):
        validate_sequence(bad, protocol)
    bad = deepcopy(record)
    bad["total_seconds"] += 1
    with pytest.raises(ValueError, match="timing sum"):
        validate_sequence(bad, protocol)
    bad = deepcopy(record)
    bad["cases"][0]["kkt"]["primal"] = 1
    with pytest.raises(ValueError, match="KKT"):
        validate_sequence(bad, protocol)


def test_memory_budget_failure_preserves_numerical_completion(monkeypatch):
    monkeypatch.setattr(
        benchmark_cht,
        "ProcessMemory",
        lambda *args: ProcessMemory(interval=60, query=lambda: (100, 0)),
    )
    config = small_config()
    config.host_memory_budget_bytes = 99
    record = benchmark_cht.complete_sequence(
        benchmark_cht.specification(config), "jacobi", "outer_inner"
    )
    assert record["numerical_success"] and not record["success"]
    assert not record["memory_budget_satisfied"]
    assert record["status"] == "memory_budget_exceeded"
    assert all(c["status"] == "converged" for c in record["cases"])


def test_calibration_input_is_hashed_and_does_not_export_its_path(tmp_path):
    config = small_config("transient")
    source = benchmark_cht.specification(config)
    record = {
        "specification": source,
        "protocol_sha256": "a" * 64,
        "environment": {"git_head": "b" * 40},
        "calibration_seconds": 12.5,
    }
    path = tmp_path / "private-input.json"
    benchmark_cht.write_report(path, record)
    config.calibration_report = str(path)
    config.slabs = 6
    protocol = benchmark_cht.specification(config)
    assert str(path) not in json.dumps(protocol)
    assert protocol["calibration_input"]["calibration_seconds"] == 12.5
    assert protocol["calibration_input"]["time_slabs"] == 3
    assert len(protocol["calibration_input"]["source_record_sha256"]) == 64
    config.horizon *= 2
    with pytest.raises(ValueError, match="horizon"):
        benchmark_cht.specification(config)


@pytest.mark.parametrize("size", [0, 1, 31, 10000])
def test_mask_encoding_round_trip_and_dimension_check(size):
    mask = np.random.default_rng(12).random(size) > 0.2
    encoded = benchmark_cht.pack_mask(mask)
    np.testing.assert_array_equal(benchmark_cht.unpack_mask(encoded, size), mask)
    with pytest.raises(ValueError):
        benchmark_cht.unpack_mask(encoded, size + 16)
    oversized = "zlib:" + base64.b64encode(zlib.compress(b"x" * 10000)).decode()
    with pytest.raises(ValueError):
        benchmark_cht.unpack_mask(oversized, 8)


@pytest.mark.parametrize("problem", ["steady", "transient"])
@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_complete_study_records_equations_memory_and_transfers(monkeypatch, problem, method):
    monkeypatch.setattr(
        benchmark_cht,
        "ProcessMemory",
        lambda *args: ProcessMemory(interval=60, query=lambda: (100, 0)),
    )
    protocol = benchmark_cht.specification(small_config(problem))
    record = benchmark_cht.complete_sequence(protocol, method, "outer_inner")
    assert record["success"] and record["memory"]["complete"]
    assert len(record["cases"]) == 3
    assert sum(record["components_seconds"].values()) == pytest.approx(record["total_seconds"])
    assert all(value >= 0 for value in record["components_seconds"].values())
    dimension = 27 * (3 if problem == "transient" else 1)
    for case in record["cases"]:
        assert max(case["kkt"].values()) <= protocol["controls"]["outer_tolerance"]
        assert case["physical_objective"] > 0
        active = benchmark_cht.unpack_mask(case["final_active_bits"], dimension)
        assert active.sum() == case["final_active_count"]
        for inner in case["inner"]:
            mask = benchmark_cht.unpack_mask(inner["inactive_mask_bits"], dimension)
            assert mask.sum() == inner["n_inactive"]
            assert inner["original_residual"] <= protocol["controls"]["rtol"]
        if problem == "transient":
            assert len(case["physical_time_newly_inactive"]) == 2
            assert max(case["recovery"].values()) <= protocol["controls"]["outer_tolerance"]


def test_failed_capped_sequences_are_retained(monkeypatch):
    monkeypatch.setattr(
        benchmark_cht,
        "ProcessMemory",
        lambda *args: ProcessMemory(interval=60, query=lambda: (100, 0)),
    )
    config = small_config()
    config.inner_cap = 1
    protocol = benchmark_cht.specification(config)
    record = benchmark_cht.complete_sequence(protocol, "jacobi", "outer_inner")
    assert not record["success"]
    assert len(record["cases"]) == 3
    assert all(case["status"] == "inner_maxiter" for case in record["cases"])
    assert all(case["initial_active_count"] == 0 for case in record["cases"])


def test_cleanup_failure_preserves_numerical_outcomes_and_closes_sampler(monkeypatch):
    sampler = ProcessMemory(interval=60, query=lambda: (100, 0))
    monkeypatch.setattr(benchmark_cht, "ProcessMemory", lambda *args: sampler)

    def failed_close(self):
        raise RuntimeError("Cleanup failure")

    monkeypatch.setattr(benchmark_cht.StudySolver, "close", failed_close)
    record = benchmark_cht.complete_sequence(
        benchmark_cht.specification(small_config()), "jacobi", "outer_inner"
    )
    assert not record["success"] and record["error_type"] == "RuntimeError"
    assert len(record["cases"]) == 3 and sampler.closed
    assert all(case["status"] == "converged" for case in record["cases"])


def test_run_records_worker_timeout_and_preserves_all_declared_attempts(tmp_path, monkeypatch):
    config = small_config()
    config.output = str(tmp_path / "run")

    def timeout(*args, **kwargs):
        raise benchmark_cht.subprocess.TimeoutExpired("worker", 1)

    monkeypatch.setattr(benchmark_cht.subprocess, "run", timeout)
    report = benchmark_cht.run(config)
    assert report["complete"] and not report["success"]
    assert len(report["sequences"]) == 3
    for row in report["sequences"]:
        record = json.loads((tmp_path / "run" / row["path"]).read_text())
        assert record["status"] == "walltime_cap" and not record["success"]
        assert row["total_seconds"] is None
