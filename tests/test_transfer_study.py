"""Independent checks of transfer replay, energy removal and residual acceptance."""

import hashlib
import json

import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy import sparse

from deflation_example import benchmark_cht, benchmark_transfer
from deflation_example.memory import ProcessMemory
from deflation_example.reporting import write_report


def test_coarse_energy_and_deflated_spectrum_match_direct_projection():
    B = sparse.csr_matrix([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    exact = np.array([1.0, 2.0, -1.0])
    b = B @ exact
    basis = np.array([[1.0], [2.0], [0.0]])
    result = benchmark_transfer.correction_diagnostics(B, b, basis, 1, exact)
    Z = basis[:, 0]
    correction = Z * (Z @ b) / (Z @ (B @ Z))
    expected = correction @ (B @ correction) / (exact @ b)
    assert result["coarse_removed_energy_fraction"] == pytest.approx(expected)
    assert result["remaining_energy_fraction"] == pytest.approx(1 - expected)
    assert result["deflated_zero_eigenvalue_max_abs"] < 1e-12
    assert len(result["remaining_nonzero_eigenvalues"]) == 2
    zero = benchmark_transfer.correction_diagnostics(B, b, basis, 0, exact)
    assert zero["coarse_removed_energy_fraction"] == 0
    assert zero["effective_condition"] == pytest.approx(zero["original_preconditioned_condition"])
    full = benchmark_transfer.correction_diagnostics(B, b, np.eye(3), 3, exact)
    assert full["coarse_removed_energy_fraction"] == pytest.approx(1)
    assert full["effective_condition"] is None


def test_rank_mismatch_is_reported_instead_of_a_different_coarse_diagnostic():
    B = sparse.eye(3, format="csr")
    result = benchmark_transfer.correction_diagnostics(
        B, np.ones(3), np.ones((3, 2)), 2, np.ones(3)
    )
    assert result["status"] == "diagnostic_rank_differs_from_solver"


def source_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark_cht,
        "ProcessMemory",
        lambda *args: ProcessMemory(interval=60, query=lambda: (100, 0)),
    )
    c = OmegaConf.structured(benchmark_cht.StudyConfig)
    c.device, c.methods, c.problem = "cpu", ["reference"], "transient"
    c.n, c.slabs, c.targets, c.rank, c.window, c.bound = 3, 3, 3, 5, 8, 0.0001
    protocol = benchmark_cht.specification(c)
    root = tmp_path / "source"
    root.mkdir()
    write_report(root / "protocol.json", protocol)
    checksum = hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
    record = benchmark_cht.complete_sequence(protocol, "reference", "outer_inner")
    assert record["success"]
    record["protocol_sha256"] = checksum
    record["repetition"] = 0
    write_report(root / "sequence.json", record)
    write_report(
        root / "results.json",
        {
            "protocol_sha256": checksum,
            "sequences": [
                {
                    "method": "reference",
                    "repetition": 0,
                    "warm_start": "outer_inner",
                    "path": "sequence.json",
                    "success": True,
                    "sha256": hashlib.sha256((root / "sequence.json").read_bytes()).hexdigest(),
                }
            ],
        },
    )
    return root


def test_complete_transient_trace_replays_identical_systems(tmp_path, monkeypatch):
    source = source_trace(tmp_path, monkeypatch)
    result = benchmark_transfer.run(source, tmp_path / "replay", repeats=2, threads=1)
    assert result["success"] and result["common_warmup_seconds"] > 0
    assert len(result["rows"]) >= 3
    for row in result["rows"]:
        assert row["one_step_identity_error"] < 1e-13
        assert row["independent_solution_residual"] < 1e-10
        for group in row["methods"].values():
            assert group["accepted_repetitions"] == 2
            assert group["diagnostics"]["status"] == "verified"
            for repeat in group["repetitions"]:
                assert repeat["total_seconds"] == pytest.approx(
                    repeat["transfer_seconds"] + repeat["kernel_seconds"]
                )
    assert any(row["newly_inactive"] > 0 for row in result["rows"])
    assert any(row["accumulated_transfer_difference_frobenius"] > 0 for row in result["rows"])


def test_source_trace_tampering_is_rejected(tmp_path, monkeypatch):
    root = source_trace(tmp_path, monkeypatch)
    record = json.loads((root / "sequence.json").read_text())
    record["success"] = False
    write_report(root / "sequence.json", record)
    with pytest.raises(ValueError, match="hash"):
        benchmark_transfer.load_trace(root, "reference", 0, "outer_inner")


def test_source_trace_acceptance_is_rechecked_after_hash_verification(tmp_path, monkeypatch):
    root = source_trace(tmp_path, monkeypatch)
    record = json.loads((root / "sequence.json").read_text())
    record["cases"][0]["kkt"]["stationarity"] = 1.0
    write_report(root / "sequence.json", record)
    manifest = json.loads((root / "results.json").read_text())
    manifest["sequences"][0]["sha256"] = hashlib.sha256(
        (root / "sequence.json").read_bytes()
    ).hexdigest()
    write_report(root / "results.json", manifest)
    with pytest.raises(ValueError, match="KKT"):
        benchmark_transfer.load_trace(root, "reference", 0, "outer_inner")
