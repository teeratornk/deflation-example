"""The Cartesian reference restricted on the device matches the host restriction."""

import pytest
from omegaconf import OmegaConf

from deflation_example import benchmark_cht


def config(problem="transient", device="cpu", reference_device="cpu"):
    c = OmegaConf.structured(benchmark_cht.StudyConfig)
    c.device, c.reference_device = device, reference_device
    c.methods = ["jacobi", "reference"]
    c.problem = problem
    c.n = c.calibration_grid = 3
    c.slabs, c.targets, c.rank, c.window = 3, 3, 5, 8
    c.bound = 0.0001
    return c


def test_device_restriction_is_declared_and_requires_cuda():
    assert benchmark_cht.specification(config())["controls"]["reference_device"] == "cpu"
    with pytest.raises(ValueError, match="requires CUDA"):
        benchmark_cht.specification(config(reference_device="cuda"))
    with pytest.raises(ValueError):
        benchmark_cht.specification(config(reference_device="tpu"))


@pytest.mark.gpu
@pytest.mark.parametrize("problem", ["steady", "transient"])
def test_device_and_host_restrictions_take_the_same_path(problem):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    records = {}
    for where in ("cpu", "cuda"):
        protocol = benchmark_cht.specification(
            config(problem, device="cuda", reference_device=where)
        )
        records[where] = benchmark_cht.complete_sequence(
            protocol, "reference", "outer_inner", torch=torch
        )
        assert records[where]["success"]
    for host, device in zip(records["cpu"]["cases"], records["cuda"]["cases"]):
        assert host["final_active_bits"] == device["final_active_bits"]
        assert [r["iterations"] for r in host["inner"]] == [
            r["iterations"] for r in device["inner"]
        ]
    assert records["cuda"]["storage"]["reference_restriction_device"] == "cuda"
