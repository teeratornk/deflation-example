"""Skip device tests where there is no usable device, rather than failing them.

Each device test already asks for cupy and for a device count. That guard covers a
machine without cupy and a machine whose device count is zero, but not a machine that
has cupy installed against a driver too old to answer the question: there the count
query itself raises, and the test fails instead of skipping.

That combination is the ordinary state of this project's login node now that the
corrected model's worktree carries the coupled GPU dependency for its screen, so the
processor-only suite has to stay green there.
"""

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
    available, reason = _device_available()
    if available:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
