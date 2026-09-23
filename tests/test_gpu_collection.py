"""Backend-specific skips must preserve the other backend's GPU coverage."""

from types import SimpleNamespace

import conftest
import pytest


def item(*markers):
    added = []
    return SimpleNamespace(keywords=set(markers), add_marker=added.append, added=added)


@pytest.mark.parametrize("available", [False, True])
def test_cupy_device_guard_leaves_torch_and_cpu_tests_alone(monkeypatch, available):
    monkeypatch.setattr(conftest, "_device_available", lambda: (available, "unavailable"))
    torch, cupy, cpu = item("gpu"), item("gpu", "cupy"), item("cupy")
    conftest.pytest_collection_modifyitems(None, [torch, cupy, cpu])
    assert not torch.added and not cpu.added
    assert len(cupy.added) == (0 if available else 1)


def test_no_cupy_query_without_cupy_gpu_tests(monkeypatch):
    def forbidden():
        raise AssertionError("Unneeded backend initialization")

    monkeypatch.setattr(conftest, "_device_available", forbidden)
    conftest.pytest_collection_modifyitems(None, [item("gpu"), item("cupy"), item()])
