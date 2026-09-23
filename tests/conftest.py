"""Skip CuPy device tests for an unavailable driver without skipping PyTorch tests."""

import pytest


def _device_available():
    try:
        import cupy
    except Exception:
        return False, "cupy is not installed"
    try:
        if not cupy.cuda.runtime.getDeviceCount():
            return False, "no CUDA device is present"
    except Exception as failure:  # an unusable driver answers no question at all
        return False, f"no usable CUDA device: {type(failure).__name__}"
    return True, ""


def pytest_collection_modifyitems(config, items):
    selected = [item for item in items if "gpu" in item.keywords and "cupy" in item.keywords]
    if not selected:
        return
    available, reason = _device_available()
    if available:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in selected:
        item.add_marker(skip)
