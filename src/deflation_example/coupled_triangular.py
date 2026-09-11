"""Persistent cuSPARSE analysis for fixed real momentum factors.

The matrix values, descriptors and analysis buffer stay unchanged between
analysis and solve. Only the contents of the fixed right-hand-side allocation
change. Plans belong to one CUDA stream and are not reentrant. This optional
adapter uses the low-level bindings supplied by the pinned CuPy dependency.
"""

from functools import cache

import numpy as np


@cache
def _vector_api():
    """Bind the documented SpSV C API absent from CuPy's Python bindings."""
    import ctypes as ct
    import os
    from cupy.cuda import device

    # Use precisely the library already loaded by CuPy. Loading the packaged
    # wheel by absolute path could mix two minor versions when the deployment
    # selects a system library first; their opaque descriptors are incompatible.
    device.get_cusparse_handle()
    if not hasattr(os, "RTLD_NOLOAD"):
        raise RuntimeError("Persistent vector solves currently require Linux")
    try:
        lib = ct.CDLL("libcusparse.so.12", mode=os.RTLD_NOLOAD)
    except OSError as error:
        raise RuntimeError("CuPy must have loaded CUDA-12 cuSPARSE before SpSV binding") from error
    pointer, integer = ct.c_void_p, ct.c_int
    common = [pointer, integer, pointer, pointer, pointer, pointer, integer, integer, pointer]
    signatures = {
        "createDescr": [ct.POINTER(pointer)],
        "destroyDescr": [pointer],
        "bufferSize": common + [ct.POINTER(ct.c_size_t)],
        "analysis": common + [pointer],
        "solve": common,
    }
    for name, signature in signatures.items():
        function = getattr(lib, "cusparseSpSV_" + name)
        function.argtypes, function.restype = signature, integer
    return lib


def _check_status(status):
    if status:
        raise RuntimeError(f"cuSPARSE SpSV returned status {status}")


class VectorTriangularPlan:
    """Persistent vector-specific analysis with fixed descriptors and storage."""

    def __init__(self, matrix, columns, *, lower, transpose):
        import ctypes as ct
        import cupy as cp
        from cupy._core import _dtype
        from cupy.cuda import device
        from cupy_backends.cuda.libs import cusparse as api
        from cupyx.cusparse import DnVecDescriptor, SpMatDescriptor

        if columns != 1:
            raise ValueError("The vector plan requires one right-hand side")
        self.descriptor = None
        self.cp, self.matrix, self.library = cp, matrix, _vector_api()
        self.stream = cp.cuda.get_current_stream()
        self.rhs = cp.empty((matrix.shape[0], 1), dtype=cp.float64)
        self.solution = cp.zeros_like(self.rhs)
        self.mat_a = SpMatDescriptor.create(matrix)
        self.mat_b = DnVecDescriptor.create(self.rhs.ravel())
        self.mat_c = DnVecDescriptor.create(self.solution.ravel())
        self.mat_a.set_attribute(
            api.CUSPARSE_SPMAT_FILL_MODE,
            api.CUSPARSE_FILL_MODE_LOWER if lower else api.CUSPARSE_FILL_MODE_UPPER,
        )
        self.mat_a.set_attribute(api.CUSPARSE_SPMAT_DIAG_TYPE, api.CUSPARSE_DIAG_TYPE_NON_UNIT)
        self.alpha = np.array(1.0, dtype=np.float64)
        descriptor = ct.c_void_p()
        _check_status(self.library.cusparseSpSV_createDescr(ct.byref(descriptor)))
        self.descriptor = descriptor
        self.arguments = (
            device.get_cusparse_handle(),
            api.CUSPARSE_OPERATION_TRANSPOSE if transpose else api.CUSPARSE_OPERATION_NON_TRANSPOSE,
            self.alpha.ctypes.data,
            self.mat_a.desc,
            self.mat_b.desc,
            self.mat_c.desc,
            _dtype.to_cuda_dtype(self.rhs.dtype),
            0,  # CUSPARSE_SPSV_ALG_DEFAULT
            self.descriptor,
        )
        try:
            size = ct.c_size_t()
            _check_status(self.library.cusparseSpSV_bufferSize(*self.arguments, ct.byref(size)))
            self.buffer = cp.empty(size.value, dtype=cp.int8)
            _check_status(self.library.cusparseSpSV_analysis(*self.arguments, self.buffer.data.ptr))
        except BaseException:
            self.close()
            raise

    def solve(self, rhs):
        if self.descriptor is None:
            raise RuntimeError("The triangular plan is closed")
        if self.cp.cuda.get_current_stream().ptr != self.stream.ptr:
            raise RuntimeError("A triangular plan must run on its original CUDA stream")
        if rhs.shape != self.rhs.shape or rhs.dtype != self.rhs.dtype:
            raise ValueError("The right-hand side must match the analyzed shape and precision")
        self.rhs[...] = rhs
        _check_status(self.library.cusparseSpSV_solve(*self.arguments))
        return self.solution

    def close(self):
        if getattr(self, "descriptor", None) is not None:
            self.stream.synchronize()
            descriptor, self.descriptor = self.descriptor, None
            _check_status(self.library.cusparseSpSV_destroyDescr(descriptor))

    def __del__(self):
        self.close()

    def storage_bytes(self):
        return self.rhs.nbytes + self.solution.nbytes + self.buffer.nbytes


class TriangularPlan:
    """Analyze one fixed triangular matrix and one right-hand-side width."""

    def __init__(self, matrix, columns, *, lower, transpose):
        import cupy as cp
        from cupy._core import _dtype
        from cupy.cuda import device
        from cupy_backends.cuda.libs import cusparse as api
        from cupyx.cusparse import DnMatDescriptor, SpMatDescriptor

        self.cp, self.api = cp, api
        self.descriptor = None
        self.matrix = matrix
        self.stream = cp.cuda.get_current_stream()
        self.handle = device.get_cusparse_handle()
        self.rhs = cp.empty((matrix.shape[0], columns), dtype=cp.float64, order="F")
        self.solution = cp.zeros_like(self.rhs, order="F")
        self.mat_a = SpMatDescriptor.create(matrix)
        self.mat_b = DnMatDescriptor.create(self.rhs)
        self.mat_c = DnMatDescriptor.create(self.solution)
        self.mat_a.set_attribute(
            api.CUSPARSE_SPMAT_FILL_MODE,
            api.CUSPARSE_FILL_MODE_LOWER if lower else api.CUSPARSE_FILL_MODE_UPPER,
        )
        self.mat_a.set_attribute(api.CUSPARSE_SPMAT_DIAG_TYPE, api.CUSPARSE_DIAG_TYPE_NON_UNIT)
        self.alpha = np.array(1.0, dtype=np.float64)
        self.descriptor = api.spSM_createDescr()
        self.arguments = (
            self.handle,
            api.CUSPARSE_OPERATION_TRANSPOSE if transpose else api.CUSPARSE_OPERATION_NON_TRANSPOSE,
            api.CUSPARSE_OPERATION_NON_TRANSPOSE,
            self.alpha.ctypes.data,
            self.mat_a.desc,
            self.mat_b.desc,
            self.mat_c.desc,
            _dtype.to_cuda_dtype(self.rhs.dtype),
            api.CUSPARSE_SPSM_ALG_DEFAULT,
            self.descriptor,
        )
        try:
            size = api.spSM_bufferSize(*self.arguments)
            self.buffer = cp.empty(size, dtype=cp.int8)
            api.spSM_analysis(*self.arguments, self.buffer.data.ptr)
        except BaseException:
            self.close()
            raise

    def solve(self, rhs):
        if self.descriptor is None:
            raise RuntimeError("The triangular plan is closed")
        if self.cp.cuda.get_current_stream().ptr != self.stream.ptr:
            raise RuntimeError("A triangular plan must run on its original CUDA stream")
        if rhs.shape != self.rhs.shape or rhs.dtype != self.rhs.dtype:
            raise ValueError("The right-hand side must match the analyzed shape and precision")
        self.rhs[...] = rhs
        # cuSPARSE requires zero-initialized output for this operation.
        self.solution.fill(0)
        self.api.spSM_solve(*self.arguments, self.buffer.data.ptr)
        return self.solution

    def close(self):
        if getattr(self, "descriptor", None) is not None:
            self.stream.synchronize()
            self.api.spSM_destroyDescr(self.descriptor)
            self.descriptor = None

    def __del__(self):
        self.close()

    def storage_bytes(self):
        return self.rhs.nbytes + self.solution.nbytes + self.buffer.nbytes


class PersistentSuperLU:
    """Copy fixed CPU factors, then reuse their GPU triangular analyses.

    Each transpose policy retains its vector plan and at most one block-width
    plan. Changing the block width releases the preceding block plan. Returned
    arrays own their data, so subsequent applications cannot overwrite them.
    """

    def __init__(self, factor):
        from cupyx.scipy.sparse.linalg import SuperLU

        self.factor = SuperLU(factor)
        self.L, self.U = self.factor.L, self.factor.U
        self.perm_r, self.perm_c = self.factor.perm_r, self.factor.perm_c
        self.plans = {}
        self.analysis_count = 0

    def solve(self, rhs, trans="N"):
        if trans not in {"N", "T"}:
            raise ValueError("Real momentum solves support N and T")
        vector = rhs.ndim == 1
        rhs = rhs[:, None] if vector else rhs
        if rhs.ndim != 2 or rhs.shape[0] != self.L.shape[0]:
            raise ValueError("Momentum directions must match the factors")
        columns = rhs.shape[1]
        key = trans, columns
        if key not in self.plans:
            for old in list(self.plans):
                if old[0] == trans and old[1] != 1 and columns != 1:
                    for plan in self.plans.pop(old):
                        plan.close()
            order = ((self.L, True), (self.U, False))
            if trans == "T":
                order = order[::-1]
            plan_class = VectorTriangularPlan if columns == 1 else TriangularPlan
            first = plan_class(order[0][0], columns, lower=order[0][1], transpose=trans == "T")
            try:
                second = plan_class(order[1][0], columns, lower=order[1][1], transpose=trans == "T")
            except BaseException:
                first.close()
                raise
            self.plans[key] = first, second
            self.analysis_count += 2
        first, second = self.plans[key]
        if trans == "N":
            work = rhs[self.factor._perm_r_rev]
            answer = second.solve(first.solve(work))[self.perm_c]
        else:
            work = rhs[self.factor._perm_c_rev]
            answer = second.solve(first.solve(work))[self.perm_r]
        return answer.ravel() if vector else answer

    def close(self):
        for plans in self.plans.values():
            for plan in plans:
                plan.close()
        self.plans.clear()

    def storage_bytes(self):
        return sum(plan.storage_bytes() for plans in self.plans.values() for plan in plans)
