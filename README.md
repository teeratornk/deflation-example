# Deflation example

A companion to **Reference-Space Reuse for Steady and Transient
State-Constrained Conjugate Heat Transfer**, by Kadeethum et al.
Solve a constrained temperature-tracking
problem, reuse a full-domain reference space, and inspect the resulting states,
optimality checks and linear iterations.

The manuscript contains the analysis and broader experiments. Its
[sources](https://github.com/teeratornk/opt_control) are maintained separately
and may require repository access. The quickstart solves three small quadratic
presets. Separate benchmarks reproduce the matched GPU comparison, complete
CHT sequences, scaled Ritz control, conditional spectral examples,
history-based recycling control and matrix-free rank study.

The [detailed supplementary tables](docs/supplementary-tables.md) provide
per-case accuracy, ranks, cost components, preparation times, memory and
refinement diagnostics. Read them on GitHub or regenerate them from the
bundled data without a GPU.

## Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/teeratornk/deflation-example.git
cd deflation-example
uv run --locked --no-dev --extra plot deflation-example plot=true output=runs/demo
```

This uses the checked-in lockfile and Python 3.12. It requires no GPU, account,
dataset download or other repository. The first run downloads Python packages;
subsequent runs reuse the environment. Python 3.11 is also supported. Omit
`--extra plot` and `plot=true` for a smaller, numerical-only installation.

The three queries rotate a four-source desired state through 0, pi/4 and pi/2.
For diffusion and frozen thermal, the equal-weight target is quarter-turn
symmetric: the first and last queries repeat the same optimization problem.
There are two distinct 2D problems and three solves per preset; the CHT
targets have different source heights and give three distinct problems.
The program solves each optimization problem twice: with direct linear solves
and with deflated CG. It then compares linear kernels on the same verified mask.
Outputs in the new directory are:

- `results.json`: resolved settings, accuracy checks, complete CPU PDAS costs,
  kernel costs, statuses and portable software/source provenance.
- `kernels.csv`: one row per query and linear method.
- `fields-0.npz` through `fields-2.npz`: desired state, both solved states,
  multiplier, active mask and recovered control.
- `fields.png` and `iterations.png`, when plotting is enabled.

A successful run exits with code 0. Failed numerical checks exit with code 1;
invalid requests or execution errors exit nonzero. Existing output directories
are never overwritten. Failed numerical records remain available for inspection.

## Change the problem

Configuration uses [Hydra](https://hydra.cc/docs/intro/): overrides are
`name=value`, not command-line flags such as `--rank`. View the settings with
`uv run --locked --no-dev deflation-example --cfg job`.

```bash
uv run --locked --no-dev deflation-example problem=thermal output=runs/thermal
uv run --locked --no-dev deflation-example problem=cht output=runs/cht
uv run --locked --no-dev deflation-example grid=40 rank=40 alpha=0.0001 output=runs/changed
```

| Preset | Default grid | State operator | Full-domain reference |
|---|---:|---|---|
| `diffusion` | 32 × 32 | Dirichlet Laplacian | Analytical sine modes |
| `thermal` | 32 × 32 | Fixed transport, Ra=100, no nonlinear reaction | Low modes of the full frozen normal operator |
| `cht` | 12 × 12 × 12 | Harmonic conductivity interface, contrast 100, prescribed Re=50 transport | Homogeneous Laplacian sine modes |

Defaults are `rank=20`, `alpha=0.001`, `threads=1`, `maxiter=20000` and
`device=cpu`. Grids count interior nodes per coordinate. The bound is calibrated
at pi/4 against achieved PDAS activity near 20%, then held fixed. Small symmetric
grids need not attain exactly 20%. Each invocation constructs a new reference;
reuse occurs across its three queries and their active-set steps.

The thermal convention is explicitly defined in [the mathematics notes](docs/method.md).
CHT uses a prescribed velocity, not a coupled flow solve. All three presets
are linear-quadratic; no nonlinear optimum is claimed.

## Optional GPU kernels

On a machine with an NVIDIA GPU and a driver compatible with the locked PyTorch
build:

```bash
uv run --locked --no-dev --extra gpu deflation-example device=cuda problem=cht output=runs/cht-gpu
```

The GPU extra is large and is not installed by the CPU quickstart. `device=cuda`
adds float64 Jacobi and deflated kernels on CPU-PDAS-selected masks. Optimization
still runs on the CPU. The returned GPU solutions are checked with the original
CPU matrix. Missing CUDA is an error, not a silent fallback to CPU execution.

Timings include GPU setup and transfers; reference construction is reported
separately. Small problems can be faster on the CPU. Iteration reduction is not
a wall-time speedup, and these examples make no general GPU performance claim.

For the earlier kernel and complete-sequence protocols, use the separate
[GPU benchmark](docs/gpu-benchmark.md). It includes GPU QR, AmgX with fresh
and persistent resources, all repetitions, itemized timing and plot generation.
It also supplies sixteen-target CHT PDAS sequences at two grids, matched warm
starts, scaled-Ritz and history controls, and ranks 0, 20 and 200 through
`128^3`. Checked-in raw records retain all repetitions and failed attempts.
The CPU quickstart and installed wheel remain small. Records for these earlier
protocols are included in the repository and source distribution.

## Complete steady and transient optimization

The [reference-policy study](docs/reference-policy-study.md) extends the companion
to complete steady and transient CHT optimization. It provides an efficient
Jacobi-CG control, reference deflation, recycling with retained coarse vectors
and Jacobi-scaled selection, and persistent-resource AmgX. The transient solver
optimizes a complete backward-Euler trajectory with constraints at every time
level. An independent bounded least-squares reference verifies small problems.
Matched trace replay measures the information retained at released constraints.
The study guide gives pilot commands and the frozen repeated-comparison campaign.
The complete study data have their own
[versioned checkout](https://github.com/teeratornk/deflation-example/tree/reference-policy-data-v1),
which keeps the installable package independent of the larger record collection.
The data include every final sequence, construction and stopping pilots,
matched replays, and independent verification records.

## Body-fitted thermal meshes

The [mesh examples](docs/mesh-showcases.md) provide an axisymmetric transformer
and a three-dimensional bore-in-block model. They include steady and complete transient
optimization, nested mesh refinement, mesh-based reference construction, and
matched solver comparisons. The packaged inputs require no application-data
download. Start with the small CPU bore-in-block example:

```bash
uv run --locked python -m deflation_example.benchmark_mesh
```

## Diagnostic tutorials

Three CPU-only tutorials reproduce the mathematical and verification examples
and summarize the stored performance comparisons:

- [Spectral diagnostics](docs/tutorials/spectral.md): rank loss, cluster cutoffs,
  eigenvalue crossing, controlled SPD restrictions and diffusion angle bounds.
- [Accuracy and stopping](docs/tutorials/verification.md): active-set release,
  original residuals, objective versus stationarity and all CHT stopping protocols.
- [Reuse costs](docs/tutorials/costs.md): resource persistence, preparation costs,
  rank/memory and the specified last-window recycling policy.

Each tutorial writes a fixed protocol, source/environment identifiers,
input hashes, complete numerical results and compact CSV summaries. These
version-1 tutorials are provided in the source revision linked by the manuscript;
the published v0.3.0 distributions retain their original contents.

## Python use and verification

The [temperature-bound example](examples/temperature_bounds/README.md)
compares optimized steady and transient fields at three upper limits on
the same bore-in-block problem. It includes the source, saved fields,
sparse-direct checks and a common-scale figure generator.

See [the short Python example](examples/basic.py) and
[the review checklist](docs/review.md). The [design notes](docs/design.md) explain
module responsibilities and the tested kernel extension point. To run the CPU tests and style checks:

```bash
uv sync --locked --extra plot --extra study
uv run --locked --extra plot --extra study pytest -m "not gpu"
uv run --locked ruff check .
```

CPU execution does not import Torch. After dependencies are installed, neither
the CLI nor the Python API needs network or Git access. Hydra leaves the working
directory unchanged and does not save a separate machine-specific configuration
dump. Output provenance excludes hostnames, user paths and scheduler identifiers.
Hydra's multirun mode is intentionally disabled because its sweeper writes
machine-specific metadata. Run additional configurations as separate invocations.

This repository is MIT-licensed. The manuscript has separate rights; see
[provenance and attribution](PROVENANCE.md). Cite the accompanying paper when
using its method, and identify the example version when reporting results.
