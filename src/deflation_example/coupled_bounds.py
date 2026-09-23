"""Physical and optimization temperature bounds for a declared resolution margin."""

import math


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def resolution_margin(temporal_error_K, spatial_error_K):
    """Propose the predeclared margin from verified resolution differences.

    This is an empirical design margin, not a rigorous error bound. Calling this
    helper does not certify that the input differences came from converged runs.
    A margin above the allowed cap requires further resolution; it is not clipped.
    """
    temporal = _finite(temporal_error_K, "Temporal difference")
    spatial = _finite(spatial_error_K, "Spatial difference")
    if min(temporal, spatial) < 0:
        raise ValueError("Resolution differences must be nonnegative")
    margin = max(0.01, 2 * (temporal + spatial))
    if margin > 0.1:
        raise ValueError("Required margin exceeds 0.1 K; refine before optimization")
    return margin


def temperature_bounds(config, *, upper_K=None):
    """Keep physical bounds unchanged while tightening optimization bounds inward."""
    lower = _finite(config["lower_K"], "Physical lower bound")
    upper = _finite(config["upper_K"] if upper_K is None else upper_K, "Physical upper bound")
    margin = _finite(config.get("temperature_margin_K", 0.0), "Temperature margin")
    if not 0 <= margin <= 0.1:
        raise ValueError("Temperature margin must lie between 0 and 0.1 K")
    if lower + margin >= upper - margin:
        raise ValueError("The optimization temperature interval must remain nonempty")
    return {
        "physical_lower_K": lower,
        "physical_upper_K": upper,
        "temperature_margin_K": margin,
        "optimization_lower_K": lower + margin,
        "optimization_upper_K": upper - margin,
    }
