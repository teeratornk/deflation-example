"""Input contracts shared by the numerical implementations."""

from numbers import Real
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


def real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numbers")
    return array.astype(float, copy=False)


def integer(value, name, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def positive_real(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def matrix(value):
    if not sparse.issparse(value) and not isinstance(value, LinearOperator):
        value = real_array(value, "Matrix")
    if len(value.shape) != 2 or np.dtype(value.dtype).kind not in "biuf":
        raise ValueError("Matrix must be two-dimensional and real")
    if not isinstance(value, LinearOperator):
        entries = value.tocsr().data if sparse.issparse(value) else value
        if not np.all(np.isfinite(entries)):
            raise ValueError("Matrix entries must be finite")
    return value
