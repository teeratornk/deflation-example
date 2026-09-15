"""Named array sizes and factor inventories do not certify complete memory use."""

import numpy as np
import pytest
from scipy import sparse

from deflation_example.coupled_memory_screen import dense_storage, factor_inventory


def test_rank_zero_screen_has_no_reference_arrays():
    result = dense_storage(10, 20, 0)
    assert result["state_degrees_of_freedom"] == 200
    assert result["one_state_vector_bytes"] == 1600
    assert all(
        value == 0
        for key, value in result.items()
        if key not in {"state_degrees_of_freedom", "one_state_vector_bytes"}
    )


def test_named_reference_arrays_have_explicit_storage_formulas():
    result = dense_storage(1000, 2048, 200)
    assert result["one_full_restricted_basis_bytes"] == 1000 * 2048 * 200 * 8
    assert result["dense_temporal_vector_pool_bytes"] == 2048**2 * 200 * 8
    assert result["tridiagonal_selected_temporal_vectors_bytes"] == 2048 * 200 * 8
    assert result["compact_reference_upper_bytes"] == (1000 + 2048 + 1) * 200 * 8


@pytest.mark.parametrize("parameters", [(1, 10, 2), (10, 0, 2), (10, 2, -1), (10, 2, True)])
def test_storage_rejects_invalid_dimensions(parameters):
    with pytest.raises(ValueError):
        dense_storage(*parameters)


def test_factor_inventory_checks_original_system_and_separates_arrays():
    A = sparse.diags(
        [np.full(4, -1.0), np.full(5, 3.0), np.full(4, -1.0)], [-1, 0, 1], format="csr"
    )
    result = factor_inventory(A)
    assert result["factor_check_passed"]
    assert result["fresh_relative_residual"] < 1e-14
    assert result["exported_factor_array_bytes"] == sum(result["exported_arrays"].values())
    assert result["factor_seconds"] >= 0
    assert result["dimension"] == 5


def test_factor_inventory_reports_singular_system_through_failure():
    with pytest.raises(RuntimeError):
        factor_inventory(sparse.csc_matrix((4, 4)))


@pytest.mark.parametrize("factor_failure", [False, True])
def test_cli_preserves_all_samples_and_separates_projected_memory(
    monkeypatch, tmp_path, factor_failure
):
    import json
    import sys
    from types import SimpleNamespace

    import deflation_example.coupled_memory_screen as module

    saved, output = tmp_path / "saved", tmp_path / "report"
    saved.mkdir()
    np.savez(saved / "reference-fields.npz", velocity=np.ones((2, 3, 2)))
    cfg = {"transient": True, "slabs": 2, "horizon_s": 20.0}
    baseline = {"baseline_sha256": "baseline"}
    monkeypatch.setattr(module, "load_saved_solution", lambda *a: (baseline, cfg, {}, "fields"))
    problem = SimpleNamespace(
        spatial_size=5,
        flow_free=np.arange(3),
        flow=SimpleNamespace(
            nv=3,
            operator=lambda v, time_step: sparse.eye(3, format="csr") / time_step,
            convection_derivative=lambda v: sparse.csr_matrix((3, 3)),
        ),
    )
    monkeypatch.setattr(module, "load_problem", lambda cfg: (problem, baseline))
    calls = []

    class Sampler:
        def __init__(self, *args):
            pass

        def start(self):
            calls.append("start")

        def finish(self):
            calls.append("finish")
            return {"complete": True, "peak_host_rss_bytes": 1024}

    monkeypatch.setattr(module, "ProcessMemory", Sampler)
    if factor_failure:

        def fail(matrix):
            raise RuntimeError("singular")

        monkeypatch.setattr(module, "factor_inventory", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen",
            "--baseline",
            str(saved),
            "--optimization",
            str(saved),
            "--sample-steps",
            "0",
            "1",
            "--slabs",
            "4",
            "8",
            "--ranks",
            "0",
            "2",
            "--output",
            str(output),
        ],
    )
    module.main()
    report = json.loads((output / "record.json").read_text())
    assert calls == ["start", "finish"]
    assert len(report["rows"]) == 4
    assert len(report["named_dense_arrays"]) == 4
    assert report["all_factor_checks_passed"] is not factor_failure
    assert report["optimization_memory_feasibility_established"] is False
    assert report["sampled_process_memory"]["peak_host_rss_bytes"] == 1024
    for row in report["rows"]:
        if factor_failure:
            assert row["status"] == "factorization_failed"
        else:
            assert (
                row["one_trajectory_factor_array_extrapolation_bytes"]
                == row["slabs"] * row["exported_factor_array_bytes"]
            )
