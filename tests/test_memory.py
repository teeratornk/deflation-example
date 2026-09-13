"""Memory boundaries and sampler failure visibility without a CUDA dependency."""

import pytest
import time
from types import SimpleNamespace

from deflation_example.memory import ProcessMemory, _nvml_device_uuid


@pytest.mark.parametrize(
    "identifier, expected",
    [
        ("01234567-89ab-cdef-0123-456789abcdef", "GPU-01234567-89ab-cdef-0123-456789abcdef"),
        ("01234567-89AB-CDEF-0123-456789ABCDEF", "GPU-01234567-89ab-cdef-0123-456789abcdef"),
        ("GPU-01234567-89ab-cdef-0123-456789abcdef", "GPU-01234567-89ab-cdef-0123-456789abcdef"),
        ("MIG-01234567-89ab-cdef-0123-456789abcdef", "MIG-01234567-89ab-cdef-0123-456789abcdef"),
        (
            "MIG-GPU-01234567-89ab-cdef-0123-456789abcdef/1/0",
            "MIG-GPU-01234567-89ab-cdef-0123-456789abcdef/1/0",
        ),
    ],
)
def test_memory_normalizes_cuda_uuid_for_nvml(identifier, expected):
    assert _nvml_device_uuid(identifier) == expected


@pytest.mark.parametrize("identifier", ["", "cuda:0", "unknown"])
def test_memory_rejects_an_unrecognized_uuid(identifier):
    with pytest.raises(ValueError, match="CUDA device UUID"):
        _nvml_device_uuid(identifier)


def test_sampler_resolves_the_current_cuda_device_by_uuid(monkeypatch):
    from deflation_example import gpu

    observed = []

    def properties(index):
        observed.append(index)
        return SimpleNamespace(uuid="01234567-89ab-cdef-0123-456789abcdef")

    cuda = SimpleNamespace(current_device=lambda: 2, get_device_properties=properties)
    monkeypatch.setattr(gpu, "require_cuda", lambda: SimpleNamespace(cuda=cuda))
    sampler = ProcessMemory("cuda")
    assert observed == [2]
    assert sampler.device_uuid == "GPU-01234567-89ab-cdef-0123-456789abcdef"
    assert sampler.process is None


@pytest.mark.parametrize("ready", [True, False])
def test_failed_external_initialization_closes_the_sampler(monkeypatch, ready):
    from deflation_example import memory

    operations = []
    parent = SimpleNamespace(
        poll=lambda timeout: ready,
        recv=lambda: {"error": "NVMLError_NotFound"},
        close=lambda: operations.append("parent_closed"),
    )
    child = SimpleNamespace(close=lambda: operations.append("child_closed"))
    process = SimpleNamespace(
        start=lambda: operations.append("started"),
        join=lambda timeout: operations.append("joined"),
        terminate=lambda: operations.append("terminated"),
    )
    context = SimpleNamespace(Pipe=lambda: (parent, child), Process=lambda **kwargs: process)
    monkeypatch.setattr(memory.multiprocessing, "get_context", lambda method: context)
    sampler = ProcessMemory()
    message = "NVMLError_NotFound" if ready else "did not initialize"
    with pytest.raises(RuntimeError, match=message):
        sampler.start()
    assert sampler.closed
    assert operations == [
        "started",
        "child_closed",
        *([] if ready else ["terminated"]),
        "joined",
        "parent_closed",
    ]
    with pytest.raises(RuntimeError, match="has not been started"):
        sampler.finish()


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
