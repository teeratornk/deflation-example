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

In a full Git clone, also scan all reachable historical blobs:

```bash
uv run --locked python tools/check_release.py --source . --history
```

The check intentionally accepts text-only release contents and limits individual
files to 5 MiB and archives to 20 MiB / 2,000 entries. It rejects duplicate paths,
Windows drive paths, links, special files and oversized members without extracting
them. History checking refuses shallow clones. Source-archive users can run the
ordinary source/archive checks without Git.

To exercise the optional implementation on a CUDA device:

```bash
uv sync --locked --extra gpu --extra plot
uv run --locked --extra gpu pytest -m gpu
uv run --locked --extra gpu deflation-example device=cuda problem=cht output=runs/gpu-check
```

CPU CI runs on Python 3.11 and 3.12, builds the distributions, and executes a wheel
from outside the source tree. CUDA tests are marked separately because ordinary
hosted CI has no GPU. Passing CPU CI does not imply that CUDA was exercised.

## Versions 0.1.1–0.1.2 verification

The expanded suite passes 93 CPU tests on Python 3.11.14 and 3.12.3 using the
lockfile. Tests cover integer bounds, extreme load magnitudes, failed-inner-solve
diagnostics, invalid inputs, kernel substitution, atomic report replacement and
historical credentials removed from the current tree and sweep-mode rejection
without metadata files. The historical test uses a synthetic
credential in a temporary test repository; no live credential is included.

All nine default CPU queries and all 18 corresponding CUDA kernels passed their
accuracy checks. CUDA validation used PyTorch 2.10.0 / CUDA 12.8 on an H200, and
the CUDA regression test includes the extreme-scale and coarse-failure cases.
The executed numerical source is commit `c92e9d8`. Version 0.1.2 changes release
packaging only; its numerical implementations are unchanged.

A wheel installed outside the checkout ran the thermal CLI and generated plots.
The unpacked source archive passed all CPU tests in a fresh locked environment.
Source, reachable Git history, distributions and generated numerical reports were
checked for credentials and machine-specific information, with no findings.
This evidence supports the tested contracts and cases; it is not a guarantee of
bug-free behavior for every possible input or environment.

## Version 0.1.0 verification

Linux x86-64 checks on 8 September 2026 passed all 44 CPU tests with Python 3.12.3
(NumPy 2.5.3, SciPy 1.18.1) and Python 3.11.14 (NumPy 2.4.6, SciPy 1.17.1).
The unpacked source archive passed the same tests in a fresh uv environment.
A separately installed wheel ran the Hydra command and generated plots from
outside the checkout, with no Torch installed and no Git source available.

All nine default CPU queries passed: diffusion and thermal at 32 × 32, and
CHT at 12 × 12 × 12, each at three angles. The largest deflated-PDAS KKT
component was below `1.2e-11`, state error against direct PDAS below `1.6e-13`,
and every independently checked kernel residual below `1e-10`. The documented
40 × 40 diffusion override with rank 40 and alpha `0.0001` also passed.
These accuracy results are not performance guarantees for other settings.

The CUDA regression test and all nine default queries also passed with PyTorch
2.10.0, CUDA 12.8 and an NVIDIA H200. All 18 CUDA kernels (Jacobi and deflated
for each query) passed fresh CPU residual checks below `1e-10`. The largest
reported Torch allocation was about 33 MiB; this excludes non-Torch allocations.
The executed package source was commit `50588d5`. Version 0.1.0's solvers and
benchmark definitions were unchanged from that commit.

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
