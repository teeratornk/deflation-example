"""Optional AmgX PCG with explicit resource ownership and timing.

No hierarchy is reused. A session retains only configuration and resource
handles; matrix, vectors, solver and hierarchy are recreated for each solve.
The caller initializes/finalizes the AmgX library once per process.
"""

from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.linalg import norm
from .solvers import LinearResult, independent_residual, validate_linear_inputs
from .timing import PhaseTimer
from .validation import matrix, positive_real, integer


def amgx_configuration(rtol=1e-10, maxiter=10000, rhs_relative=False):
    positive_real(rtol, "Relative tolerance")
    integer(maxiter, "Iteration limit", 1)
    return {
        "config_version": 2,
        "determinism_flag": 1,
        "solver": {
            "scope": "main",
            "solver": "PCG",
            "preconditioner": {
                "scope": "amg",
                "solver": "AMG",
                "algorithm": "CLASSICAL",
                "selector": "PMIS",
                "interpolator": "D2",
                "smoother": {
                    "solver": "MULTICOLOR_GS",
                    "relaxation_factor": 1.0,
                    "symmetric_GS": 1,
                },
                "presweeps": 1,
                "postsweeps": 1,
                "cycle": "V",
                "max_iters": 1,
                "max_levels": 25,
                "coarse_solver": "DENSE_LU_SOLVER",
            },
            "tolerance": rtol,
            "max_iters": maxiter,
            "norm": "L2",
            "convergence": "ABSOLUTE" if rhs_relative else "RELATIVE_INI_CORE",
            "monitor_residual": 1,
            "obtain_timings": 1,
            "print_solve_stats": 0,
        },
    }


def destroy_all(objects):
    """Attempt every destruction, including when one cleanup operation fails."""
    errors = []
    while objects:
        obj = objects.pop()
        try:
            obj.destroy()
        except Exception as error:
            errors.append(error)
    if errors:
        raise RuntimeError("AmgX resource cleanup failed") from errors[0]


class AmgxSession:
    def __init__(self, api, rtol=1e-10, maxiter=10000, rhs_relative=False):
        self.api, self.rtol, self.maxiter = api, rtol, maxiter
        self.rhs_relative = rhs_relative
        self.configuration = amgx_configuration(rtol, maxiter, rhs_relative)
        self.objects = []
        self.config = self.resources = None

    def open(self):
        if self.objects:
            raise RuntimeError("AmgX session is already open")
        try:
            self.config = self.api.Config().create_from_dict(self.configuration)
            self.objects.append(self.config)
            self.resources = self.api.Resources().create_simple(self.config)
            self.objects.append(self.resources)
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        try:
            destroy_all(self.objects)
        finally:
            self.config = self.resources = None


def amgx_cg(
    A,
    b,
    *,
    api,
    synchronize,
    session=None,
    rtol=1e-10,
    maxiter=10000,
    x0=None,
    rhs_relative=False,
    acceptance_rtol=None,
):
    timer = PhaseTimer()
    timer.synchronize(synchronize)
    A = sparse.csr_matrix(matrix(A), dtype=np.float64).copy()
    b, initial, _ = validate_linear_inputs(A, b, None, None, x0, rtol, maxiter, 1000, 1e10)
    A.sort_indices()
    A.sum_duplicates()
    if max(A.shape) > np.iinfo(np.int32).max or A.nnz > np.iinfo(np.int32).max:
        raise ValueError("AmgX dDDI requires 32-bit matrix indices")
    A.indptr, A.indices = A.indptr.astype(np.int32), A.indices.astype(np.int32)
    acceptance_rtol = (
        rtol if acceptance_rtol is None else positive_real(acceptance_rtol, "Acceptance tolerance")
    )
    if acceptance_rtol < rtol:
        raise ValueError("Acceptance tolerance must be at least the iteration tolerance")
    absolute_tolerance = rtol * (float(norm(b)) or 1.0)
    if rhs_relative and not 0 < absolute_tolerance < float("inf"):
        raise ValueError("Native absolute tolerance is outside the finite range")
    rhs_values = np.ascontiguousarray(b)
    initial = np.ascontiguousarray(initial)
    initial_residual = independent_residual(A, initial, b)
    timer.mark("conversion")
    own_session = session is None
    objects = []
    try:
        if own_session:
            session = AmgxSession(api, rtol, maxiter, rhs_relative).open()
        if (
            session.resources is None
            or session.rtol != rtol
            or session.maxiter != maxiter
            or session.rhs_relative != rhs_relative
        ):
            raise ValueError("AmgX session is closed or has different solver controls")
        timer.mark("resource_creation")
        timer.synchronize(synchronize)
        solver_config = session.config
        if rhs_relative:
            configuration = deepcopy(session.configuration)
            configuration["solver"]["tolerance"] = absolute_tolerance
            solver_config = api.Config().create_from_dict(configuration)
            objects.append(solver_config)
        gpu_matrix = api.Matrix().create(session.resources, "dDDI")
        objects.append(gpu_matrix)
        x = api.Vector().create(session.resources, "dDDI")
        objects.append(x)
        rhs = api.Vector().create(session.resources, "dDDI")
        objects.append(rhs)
        solver = api.Solver().create(session.resources, solver_config, "dDDI")
        objects.append(solver)
        timer.mark("handle_creation")
        timer.synchronize(synchronize)
        gpu_matrix.upload_CSR(A)
        rhs.upload(rhs_values)
        x.upload(initial)
        timer.mark("upload")
        timer.synchronize(synchronize)
        solver.setup(gpu_matrix)
        timer.mark("coarse_or_hierarchy_setup")
        timer.synchronize(synchronize)
        solver.solve(rhs, x)
        timer.mark("iteration")
        timer.synchronize(synchronize)
        solution = np.empty_like(b)
        x.download(solution)
        timer.mark("download")
        timer.synchronize(synchronize)
        iterations, native_status = int(solver.iterations_number), str(solver.status)
        residual = independent_residual(A, solution, b)
        # Retain native status independently; fresh original equations are
        # always required. A cap is not converted into success.
        native_converged = native_status.lower() in {"success", "amgx_solve_success", "0"}
        accepted = native_converged and np.isfinite(residual) and residual <= acceptance_rtol
        status = "converged" if accepted else "residual_failed"
        timer.mark("verification")
    finally:
        try:
            destroy_all(objects)
        finally:
            if own_session and session is not None:
                session.close()
            timer.mark("cleanup")
            timer.synchronize(synchronize)
    metrics = timer.finish()
    metrics.update(
        native_status=native_status,
        hierarchy_reused=False,
        resources_reused=not own_session,
        rhs_relative=rhs_relative,
        initial_residual=initial_residual,
        iteration_rtol=rtol,
        acceptance_rtol=acceptance_rtol,
        native_absolute_tolerance=absolute_tolerance if rhs_relative else None,
        solver_configuration_per_call=rhs_relative,
    )
    return LinearResult(solution, iterations, residual, status), metrics
