import pytest

from deflation_example.coupled_qp_cuda_report import summarize
from test_coupled_qp_inexact_report import record


def gpu_record(rank):
    value = record()
    value.update(
        device="cuda",
        reference_rank=rank,
        gpu={"model": "test-gpu"},
        quadratic_seconds=5,
        reference_construction_seconds=rank,
        preconditioner_construction_seconds=1,
    )
    return value


def test_missing_and_failed_arms_and_cost_boundary():
    result = summarize([gpu_record(0), gpu_record(16), None], [0, 16, 32])
    assert not result["all_quadratics_verified"]
    assert result["rows"][0]["quadratic_plus_construction_seconds"] == 6
    assert result["rows"][1]["quadratic_plus_construction_seconds"] == 22
    assert result["rows"][2]["status"] == "missing"
    assert "speedup" not in result


@pytest.mark.parametrize(
    "key,value",
    [
        ("linear_tolerance", 1e-4),
        ("quadratic_sha256", "different"),
        ("quadratic_tolerance", 0.1),
        ("gpu", {"model": "other"}),
    ],
)
def test_unmatched_gpu_comparisons_rejected(key, value):
    other = gpu_record(16)
    other[key] = value
    if key == "linear_tolerance":
        other["history"][0]["linear_residual"] = value / 10
    with pytest.raises(ValueError, match="Match"):
        summarize([gpu_record(0), other], [0, 16])


def test_missing_gpu_identity_and_wrong_rank_rejected():
    value = gpu_record(0)
    del value["gpu"]
    with pytest.raises(ValueError, match="identify"):
        summarize([value], [0])
    with pytest.raises(ValueError, match="declared"):
        summarize([gpu_record(0)], [16])


def test_reconstruction_error_remains_visible():
    value = gpu_record(16)
    value.update(status="diagnostic_error", error="device allocation failed")
    del value["gpu"]
    result = summarize([gpu_record(0), value], [0, 16])
    assert result["rows"][1]["error"] == "device allocation failed"
    assert not result["all_quadratics_verified"]
