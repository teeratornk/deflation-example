# Reviewing the example

Start with the CPU quickstart in the README. It builds its own operators, desired
states, bound and reference space. There are no input datasets, absolute source
paths or imports from the manuscript's computational repository.

## Numerical checks

The test suite covers incorrect active indices leaving the mask, an independent
small constrained optimization solution, exact coarse corrections, rank loss,
zero loads, coarse-conditioning fallback and explicit nonconvergence. Small
deterministic operator/target fixtures check the three presets against the
manuscript implementation. End-to-end tests check all three problem classes.

For each completed example, inspect `success` in `results.json`, then the KKT
components, objective difference and independently recomputed kernel residuals.
Do not interpret an iteration count from a failed solve as a result. The
acceptance tolerances and timing boundaries are defined in [method.md](method.md).

```bash
uv sync --locked --extra plot
uv run --locked pytest -m "not gpu"
uv run --locked ruff check .
uv run --locked ruff format --check .
uv build
uv run --locked python tools/check_release.py --source . dist/*.whl dist/*.tar.gz
```

The archive check rejects common credential patterns, user paths, generated
outputs, environment files, unsafe paths and archive links. It prints rule names,
not suspected credential values. Pattern checks cannot prove the absence of every
possible secret; the release file inventory also needs human review.

To exercise the optional implementation on a CUDA device:

```bash
uv sync --locked --extra gpu --extra plot
uv run --locked --extra gpu pytest -m gpu
uv run --locked --extra gpu deflation-example device=cuda problem=cht output=runs/gpu-check
```

CPU CI runs on Python 3.11 and 3.12, builds the distributions, and executes a wheel
from outside the source tree. CUDA tests are marked separately because ordinary
hosted CI has no GPU. Passing CPU CI does not imply that CUDA was exercised.

## Scope

The examples are small Cartesian, linear-quadratic control problems. They do not
validate nonlinear optimization, industrial scalability, unstructured meshes,
automatic reference selection or every manuscript experiment. The default rank
is 20. Thermal eigenmodes are computed directly at that rank, so iteration counts
need not match experiments that selected modes from a larger precomputed pool.

The lockfile fixes the tested dependencies. Timings and last-digit differences
can change with the platform, BLAS library and GPU driver. A kernel comparison
uses the same matrix and right-hand side; it is not an end-to-end CPU/GPU timing
comparison. GPU memory reports cover Torch's allocator, not total process or
device memory. The source archive includes tests and this checklist; the wheel
contains only the installed package, configuration, metadata and license.
