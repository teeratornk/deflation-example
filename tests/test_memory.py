"""Memory boundaries and sampler failure visibility without a CUDA dependency."""

import pytest

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
