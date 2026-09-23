"""Smooth bounded streamline coefficients and their velocity derivatives."""

import numpy as np


def smooth_parameter(grad, lump, capacity, kmin, velocity, vertices, limit_rows):
    r"""Return tau=(tau_d^-8+tau_a^-8+tau_r^-8)^(-1/8) and d tau/d v.

    Omit the row term when ``limit_rows`` is false. With N finite limits,
    N^(-1/8) min(tau_i) <= tau <= min(tau_i). The eighth powers of the
    velocity norms are polynomials, so the coefficient is smooth at rest and
    at coincident limits. Geometry and material coefficients are held fixed.
    Rescaling the inverse limits avoids forming their large eighth powers.
    """
    h = np.max(np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=3), axis=(1, 2))
    speed = np.hypot.reduce(velocity, axis=1)
    diffusion = 12 * kmin / h**2
    advection = 2 * capacity * speed / h
    unit = np.divide(
        velocity, speed[:, None], out=np.zeros_like(velocity), where=speed[:, None] > 0
    )
    rates = [diffusion, advection]
    derivatives = [np.zeros_like(velocity), (2 * capacity / h)[:, None] * unit]
    if limit_rows:
        share = (lump / lump.sum(axis=1)[:, None]).min(axis=1)
        directional = np.einsum("eid,ed->ei", grad, velocity)
        reach = np.hypot.reduce(directional, axis=1)
        unit_directional = np.divide(
            directional, reach[:, None], out=np.zeros_like(directional), where=reach[:, None] > 0
        )
        rates.append(capacity * reach / share)
        derivatives.append(
            (capacity / share)[:, None] * np.einsum("ei,eid->ed", unit_directional, grad)
        )
    rates = np.stack(rates, axis=1)
    derivatives = np.stack(derivatives, axis=1)
    scale = rates.max(axis=1)
    if not np.isfinite(rates).all() or np.any(scale <= 0):
        raise ValueError("Streamline limiting rates must be finite and positive")
    normalized = rates / scale[:, None]
    total = np.sum(normalized**8, axis=1)
    tau = (1 / scale) * total ** (-1 / 8)
    dtau = (
        -tau[:, None]
        * np.sum(normalized[:, :, None] ** 7 * (derivatives / scale[:, None, None]), axis=1)
        / total[:, None]
    )
    if not np.isfinite(dtau).all() or np.any(tau <= 0):
        raise ValueError("Streamline coefficient or derivative is outside floating-point range")
    return tau, dtau
