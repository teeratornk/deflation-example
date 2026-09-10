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
They also repeat each default preset, compare the saved fields within numerical
tolerances, and recompute the equations and KKT components from those fields.
Repeatability checks exclude elapsed times and permit floating-point differences.

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

In a full Git clone, also scan all reachable historical blobs:

```bash
uv run --locked python tools/check_release.py --source . --history
```

The check intentionally accepts text-only release contents and limits individual
files to 5 MiB and source archives to 64 MiB / 2,000 entries. It rejects duplicate paths,
Windows drive paths, links, special files and oversized members without extracting
them. History checking refuses shallow clones. Source-archive users can run the
ordinary source/archive checks without Git.

To exercise the optional implementation on a CUDA device:

```bash
uv sync --locked --extra gpu --extra plot
uv run --locked --extra gpu pytest -m gpu
uv run --locked --extra gpu deflation-example device=cuda problem=cht output=runs/gpu-check
```

CPU CI runs on Python 3.11 and 3.12 and builds both distributions. It installs the
wheel into a separate environment with dependencies exported from `uv.lock`,
then runs all three default presets with plots and all three diagnostic tutorials
from outside the source tree. Tutorials receive the benchmark-record directory
explicitly. CUDA tests cover the complete demo adapter, both basis-processing
variants, matrix-free kernels and AmgX warm starts. They are marked separately
because ordinary hosted CI has no GPU. Review skipped tests: a missing CUDA device
or native AmgX binding leaves that implementation unchecked.

## Current benchmark coverage

The release includes matched GPU QR and AmgX kernels, complete sixteen-target
CHT sequences at two grids, scaled Ritz selection, controlled spectral
examples, previous-system recycling, and matrix-free ranks 0, 20 and 200.
The [benchmark guide](gpu-benchmark.md) gives commands, fixed configurations,
timing boundaries and source identifiers. Checked-in raw records retain every
repetition, iteration cap and failed stopping protocol.

The complete-sequence validator checks hashes, distinct targets, requested and
deployed ranks, original residuals, KKT components, resource policies and timing
partitions. Separate startup measurements include fresh Python processes,
scientific/GPU imports, runtime initialization and shutdown. The manuscript
table generator charges these preparation costs without double counting GPU
runtime initialization.

The installed wheel excludes the full raw benchmark records. The repository
and source distribution include those records for independent inspection.
The larger archive limit accommodates all repetitions, including failures.
Earlier release checks remain documented in Git history and the changelog.

## Scope

The optimization examples use Cartesian, linear-quadratic control problems.
Nonlinear optimization, unstructured meshes, industrial-scale optimization and
automatic reference selection require separate validation. Large matrix-free
systems have manufactured solutions and prescribed inactive sets. The default rank
is 20. Thermal eigenmodes are computed directly at that rank, so iteration counts
need not match experiments that selected modes from a larger precomputed pool.

The lockfile fixes the tested dependencies. Timings and last-digit differences
can change with the platform, BLAS library and GPU driver. A kernel comparison
uses the same matrix and right-hand side; it is not an end-to-end CPU/GPU timing
comparison. GPU memory reports cover Torch's allocator, not total process or
device memory. The source archive includes tests and this checklist; the wheel
contains only the installed package, configuration, metadata and license.
