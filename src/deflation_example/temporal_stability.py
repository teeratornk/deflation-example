"""Computed thermal amplification modes and their discrete energy contributions."""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, eigs, splu

from .validation import positive_real


def amplification_mode(assembly, dt, time_scale_s=1.0):
    """Measure a dominant mode of (C/dt+K)^(-1) C/dt.

    The reported eigenpair residual verifies a computed mode. A growing mode
    demonstrates instability; absence of a growing computed mode is not an
    exhaustive spectral certificate for a large sparse matrix.
    """
    dt = positive_real(dt, "Time step")
    seconds = positive_real(time_scale_s, "Time scale")
    indices = assembly.mesh.free
    capacity = assembly.capacity[indices]
    K = assembly.stiffness[indices][:, indices].tocsc()
    if not len(indices) or np.any(capacity <= 0) or not np.isfinite(capacity).all():
        raise ValueError("Thermal modes require unknowns with positive finite capacity")
    Cdt = capacity / dt
    factor = splu(K + sparse.diags(Cdt))
    operator = LinearOperator(K.shape, matvec=lambda x: factor.solve(Cdt * x), dtype=float)
    if len(indices) <= 8:
        values, vectors = linalg.eig(factor.solve(np.diag(Cdt)))
    else:
        values, vectors = eigs(
            operator,
            k=3,
            which="LM",
            tol=1e-10,
            maxiter=1000,
            v0=np.ones(len(indices)),
        )
    j = int(np.abs(values).argmax())
    mu = values[j]
    vector = vectors[:, j]
    vector /= np.sqrt(np.vdot(vector, capacity * vector).real)
    spatial_value = (1 / mu - 1) / dt
    action = K @ vector
    residual = float(
        np.linalg.norm(action - spatial_value * capacity * vector)
        / max(np.linalg.norm(action), np.finfo(float).tiny)
    )
    components = {
        name: float(np.vdot(vector, getattr(assembly, name)[indices][:, indices] @ vector).real)
        / seconds
        for name in ("diffusion", "transport", "stabilization")
    }
    return {
        "largest_computed_amplification_modulus": float(abs(mu)),
        "amplification_real": float(mu.real),
        "amplification_imaginary": float(mu.imag),
        "spatial_eigenvalue_real_per_second": float(spatial_value.real / seconds),
        "spatial_eigenvalue_imaginary_per_second": float(spatial_value.imag / seconds),
        "relative_eigenpair_residual": residual,
        "modal_energy_per_second": components,
        "energy_identity_absolute_error": abs(
            sum(components.values()) - spatial_value.real / seconds
        ),
    }
