"""Memory boundaries and sampler failure visibility without a CUDA dependency."""

import pytest
import time

from deflation_example.memory import ProcessMemory


def test_memory_uses_same_process_boundary_and_reports_peaks():
    values = iter([(100, 50), (150, 200), (120, 70)])
    sampler = ProcessMemory(interval=60, query=lambda: next(values)).start()
    sampler.sample()
    record = sampler.finish()
    assert record["complete"]
    assert record["samples"] == 3
    assert record["peak_host_rss_bytes"] == 150
    assert record["peak_gpu_process_bytes"] == 200
    assert record["initial_gpu_process_bytes"] == 50
    assert not {"pid", "uuid", "hostname"} & record.keys()


def test_memory_sampling_errors_remain_visible():
    values = iter([(1, 2)])
    sampler = ProcessMemory(interval=60, query=lambda: next(values)).start()
    record = sampler.finish()
    assert not record["complete"] and record["error_type"] == "StopIteration"
    with pytest.raises(RuntimeError):
        ProcessMemory(query=lambda: (-1, 2)).start()


def test_external_sampler_observes_parent_memory_and_terminates():
    pytest.importorskip("psutil")
    sampler = ProcessMemory(interval=0.01).start()
    buffer = bytearray(8 * 1024 * 1024)
    time.sleep(0.04)
    record = sampler.finish()
    assert len(buffer) > 0 and record["complete"]
    assert record["samples"] >= 3
    assert record["peak_host_rss_bytes"] >= record["initial_host_rss_bytes"]
    assert record["peak_gpu_process_bytes"] == 0
    assert not sampler.process.is_alive()
    with pytest.raises(RuntimeError):
        sampler.finish()


@pytest.mark.gpu
def test_external_sampler_measures_cuda_process_memory():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pynvml")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    initialized = torch.ones(1, device="cuda")
    sampler = ProcessMemory("cuda", interval=0.01).start()
    buffer = torch.ones(2**22, device="cuda", dtype=torch.float64)
    torch.cuda.synchronize()
    time.sleep(0.04)
    record = sampler.finish()
    assert record["complete"] and record["samples"] >= 3
    assert record["peak_gpu_process_bytes"] >= record["initial_gpu_process_bytes"] > 0
    assert not sampler.process.is_alive()
    del buffer, initialized
