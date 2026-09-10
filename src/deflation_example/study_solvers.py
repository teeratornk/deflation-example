"""Matched inner-solver adapters for complete CHT optimization sequences."""

import time

import numpy as np

from .amgx import AmgxSession, amgx_cg
from .recycling import RecycleSpace, inactive_indices
from .solvers import deflated_cg, orthonormalize
from .timing import PHASES
from .validation import integer, positive_real


METHODS = ("jacobi", "reference", "recycling", "amgx")


class ArrayReference:
    """A reference defined and stored before optimization on the full domain."""

    def __init__(self, basis, description):
        self.basis = basis
        self.rank = basis.shape[1]
        self.description = description

    def restrict(self, indices):
        return self.basis[indices]

    def storage(self):
        return {"full_reference_bytes": self.basis.nbytes, "compact_reference_bytes": 0}


class StudySolver:
    """One method's state and resource ownership across PDAS inner solves.

    The runner supplies identical precision, stopping checks and warm-start
    access. Each call recreates the restricted operator on the device. Only the
    reference, selected recycling vectors or AmgX Config/Resources persist.
    """

    def __init__(
        self,
        method,
        *,
        device="cpu",
        rank=20,
        window=20,
        reference=None,
        rtol=1e-10,
        cg_factor=1.0,
        amgx_factor=0.1,
        maxiter=10000,
        refresh=1000,
        cache_operator_product=True,
        torch=None,
        api=None,
        resident_recycling=False,
    ):
        if method not in METHODS or device not in {"cpu", "cuda"}:
            raise ValueError("Unknown study solver or device")
        if method == "amgx" and (device != "cuda" or api is None or torch is None):
            raise ValueError("AmgX requires its native binding and a CUDA device")
        if method == "reference" and reference is None:
            raise ValueError("Reference deflation requires a declared full-domain reference")
        self.rank = integer(rank, "Rank")
        self.window = integer(window, "Direction window", 1)
        self.rtol = positive_real(rtol, "Final relative tolerance")
        self.cg_factor = positive_real(cg_factor, "CG stopping factor")
        self.maxiter = integer(maxiter, "Iteration cap", 1)
        self.refresh = integer(refresh, "Residual refresh interval", 1)
        if not isinstance(cache_operator_product, bool):
            raise ValueError("Operator-product caching must be Boolean")
        self.cache_operator_product = cache_operator_product
        self.amgx_factor = positive_real(amgx_factor, "AmgX stopping factor")
        if max(self.amgx_factor, self.cg_factor) > 1:
            raise ValueError("Internal stopping factors must not exceed one")
        self.method, self.device = method, device
        self.reference, self.torch, self.api = reference, torch, api
        self.history = RecycleSpace(rank, window, device, resident=resident_recycling) if method == "recycling" else None
        self.session = None
        self.previous = None
        if method == "amgx":
            self.session = AmgxSession(api, rtol * amgx_factor, maxiter, True).open()

    def reset_history(self):
        if self.history is not None:
            self.history.clear()
        self.previous = None

    def close(self):
        if self.session is not None:
            self.session.close()
        self.reset_history()

    def solve(self, B, b, indices, initial=None):
        start = time.perf_counter()
        indices = inactive_indices(indices)
        if self.previous is None:
            newly_inactive, newly_active = None, None
        else:
            newly_inactive = len(np.setdiff1d(indices, self.previous, assume_unique=True))
            newly_active = len(np.setdiff1d(self.previous, indices, assume_unique=True))
        device_reference = (self.method == "reference" and self.device == "cuda"
                            and callable(getattr(self.reference, "restrict_device", None)))
        device_basis = device_reference or (self.history is not None and self.history.resident)
        basis = (
            self.reference.restrict_device(indices)
            if device_reference
            else self.reference.restrict(indices)
            if self.method == "reference"
            else self.history.begin(indices)
            if self.history is not None
            else None
        )
        diagonal = B.diagonal()
        if device_basis:
            self.torch.cuda.synchronize()
        preparation = time.perf_counter() - start
        if self.method == "amgx":
            result, metrics = amgx_cg(
                B,
                b,
                api=self.api,
                synchronize=self.torch.cuda.synchronize,
                session=self.session,
                rtol=self.rtol * self.amgx_factor,
                acceptance_rtol=self.rtol,
                maxiter=self.maxiter,
                x0=initial,
                rhs_relative=True,
            )
        elif self.device == "cuda":
            from .gpu import gpu_deflated_cg

            result, metrics = gpu_deflated_cg(
                B,
                b,
                None if device_basis else basis,
                diagonal,
                x0=initial,
                rtol=self.rtol * self.cg_factor,
                acceptance_rtol=self.rtol,
                maxiter=self.maxiter,
                basis_backend="gpu_qr",
                refresh=self.refresh,
                cache_operator_product=self.cache_operator_product,
                direction_callback=None if self.history is None else self.history.capture,
                completion_callback=None if self.history is None else self.history.finish,
                device_basis=basis if device_basis else None,
            )
        else:
            tick = time.perf_counter()
            result = deflated_cg(
                B,
                b,
                basis,
                diagonal,
                x0=initial,
                rtol=self.rtol * self.cg_factor,
                maxiter=self.maxiter,
                refresh=self.refresh,
                cache_operator_product=self.cache_operator_product,
                direction_callback=None if self.history is None else self.history.capture,
            )
            kernel_seconds = time.perf_counter() - tick
            if result.status in {"maxiter", "residual_failed"} and result.residual <= self.rtol:
                result.status = "converged"
            tick = time.perf_counter()
            selection = None
            if self.history is not None:
                deployed = orthonormalize(basis) if result.rank else None
                selection = self.history.finish(B, diagonal, deployed, result.status)
            selection_seconds = time.perf_counter() - tick
            parts = dict.fromkeys(PHASES, 0.0)
            # CPU controls expose the aggregate CG interval. GPU primary
            # comparisons retain the full phase partition from their kernels.
            parts["iteration"] = kernel_seconds
            parts["basis_processing"] = selection_seconds
            metrics = {
                "total_seconds": kernel_seconds + selection_seconds,
                "components_seconds": parts,
                "completion": selection,
                "iteration_rtol": self.rtol * self.cg_factor,
                "acceptance_rtol": self.rtol,
                "operator_product_cached": bool(self.cache_operator_product and result.rank),
                "cached_operator_product_bytes": B.shape[0] * result.rank * 8
                if self.cache_operator_product
                else 0,
            }
        self.previous = indices
        return result, {
            **metrics,
            "restriction_and_transfer_seconds": preparation,
            "newly_inactive": newly_inactive,
            "newly_active": newly_active,
            "input_basis_columns": 0 if basis is None else basis.shape[1],
            "restricted_basis_bytes": 0 if basis is None else (
                basis.numel()*basis.element_size() if device_basis else basis.nbytes),
            "reference_restriction_device": "cuda" if device_reference else "cpu",
            "basis_transfer_device": "cuda" if device_basis else "cpu",
            "callback_seconds": time.perf_counter() - start,
        }
