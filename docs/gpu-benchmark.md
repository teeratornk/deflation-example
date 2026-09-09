# Matched GPU benchmark

This self-contained benchmark generates the six Laplacian inactive systems used in the manuscript GPU comparison. It needs no manuscript checkout, stored matrix or private dataset. It includes the GPU-QR variant, CPU-SVD/GPU control, exact AmgX configuration, resource-lifetime controls and every timed repetition.

## Installation

Use Linux, an NVIDIA GPU, a compatible driver and the locked Python environment:

```bash
uv sync --locked --extra gpu --extra plot
```

AmgX and its Python binding are optional native dependencies, not Python packages supplied by this repository. The measurements use AmgX **2.5.0**, PyAMGX reporting **0.1**, PyTorch 2.10.0, CUDA 12.8 and an H200. Build [NVIDIA AmgX](https://github.com/NVIDIA/AMGX/tree/v2.5.0) using the upstream CMake instructions; a single-GPU build can use `CMAKE_NO_MPI=ON`. The public v2.5.0 tag resolves to `cc1cebdbb32b14d33762d4ddabcb2e23c1669f47`. Set `AMGX_DIR` and, if needed, `AMGX_BUILD_DIR`, then build [PyAMGX](https://pyamgx.readthedocs.io/en/latest/install.html) against that library in this environment. Its build requires Cython and SciPy. Record the binding checkout commit when rebuilding: the measured installation reports a version and binary hash, but does not retain its original checkout commit.

After manually installing the binding, use `uv run --no-sync` so uv does not remove the additional native package. Check availability and execute the CUDA tests before measuring:

```bash
uv run --no-sync python -c 'import torch, pyamgx; assert torch.cuda.is_available()'
uv run --no-sync pytest -m gpu
uv run --no-sync python -m deflation_example.benchmark_gpu 'grids=[8]' rank=3 repeats=1 include_cpu=false output=runs/gpu-smoke
uv run --no-sync python -m deflation_example.benchmark_gpu output=runs/gpu-benchmark
uv run --no-sync python -m deflation_example.benchmark_report runs/gpu-benchmark/results.json --output runs/gpu-figures --plot
```

Hydra overrides include `grids`, `rank`, odd `repeats`, `threads` and `include_cpu`. `--cfg job` prints the configuration without CUDA. Output directories must be new. The report command validates every accepted repetition, exports `repetitions.csv` and `cumulative.json`, and optionally plots `cumulative.pdf`. Timings will vary across hardware, builds and runs.

## Problem and acceptance

Each grid has three moving-Gaussian targets at angles 0, 0.35 and 0.7 radians. The normal matrix is `I + 0.001 L.T L`. The bound is calibrated once with eight PDAS bisections, then held fixed. CPU PDAS must meet a normalized KKT tolerance of `1e-8`. Each method receives the same CPU CSR inactive matrix, right-hand side and zero initial guess. Every timed result must meet a newly computed original CPU residual of `1e-10`; AmgX must also report native success. Five repetitions use rotating method order, following one recorded warmup per method and grid. The deflated and AmgX iteration limits are 10,000; the single-run Jacobi control allows 20,000.

The 100 reference columns are analytical tensor sine modes sorted by discrete Laplacian eigenvalue. GPU basis processing uses thin QR followed by an SVD of the small triangular factor. Singular values below `1e-12` times the largest are discarded. A coarse condition above `1e10` causes rank-zero fallback. The record separates requested, orthogonalized and deployed ranks. Cholesky is used only for an accepted SPD coarse matrix. Residual refresh and restart occur every 1,000 iterations or when the recurrence first meets the target.

## Ownership and timing

`amgx.py` contains the complete configuration. Both AmgX controls create a new matrix, two vectors, solver and hierarchy for every inactive system. The persistent control retains **only Config and Resources** across a grid. It does not reuse a hierarchy, matrix sparsity pattern or solver handle. All owned handles are destroyed, including on failure. Resource startup and final cleanup are timed separately for the persistent session. AmgX library initialization/finalization occur once per process.

The measured kernel starts with CPU inputs and ends after the solution is returned, independently verified and temporary objects released. Its nonoverlapping components are:

| Component | Work included |
|---|---|
| Conversion | Validation, CSR/index conversion, RHS and diagonal preparation |
| Resource creation | AmgX Config/Resources when not persistent |
| Handle creation | AmgX matrix, vectors and solver |
| Upload | Matrix, vectors and basis host-to-device copies |
| Basis processing | CPU SVD or GPU thin QR plus small-factor SVD |
| Coarse/hierarchy setup | Deflation coarse assembly/condition/factor or AmgX hierarchy |
| Initialization | Initial deflation correction and residual preparation |
| Iteration | Initial search direction and Krylov loop, including recurrence checks and internal residual refresh |
| Download | Return of the solution to CPU |
| Verification | Independent original CPU residual and native-status query |
| Cleanup | Per-call tensor or handle destruction |
| Synchronization | Time spent in explicit CUDA barriers |
| Host bookkeeping | Remaining measured host intervals |

The components sum to `total_seconds`. They partition wall time, not device-kernel duration: implicit synchronization inside a library call belongs to that call, and explicit barrier waiting is listed separately. Do not add the legacy `setup_seconds`/`solve_seconds` aggregations to these components. PyTorch keeps its allocator cache and AmgX keeps process-level library state; neither method is charged a full runtime reset after each matrix.

Matrix extraction, basis restriction, reference construction, warmups, persistent-session startup/cleanup and process initialization are explicit separate records. Cumulative curves add matrix extraction to all methods, basis restriction and reference construction to deflation, and one warmup per method. Persistent resource creation/destruction are charged once to each plotted prefix. Shared runtime initialization and the CPU optimization that determines the masks are outside these kernel-sequence curves. The curves are **sums of instance medians**, not independently repeated complete-sequence measurements.

## Complete optimization and mathematical controls

```bash
uv run --no-sync python -m deflation_example.benchmark_sequence --output runs/cht-sequences
uv run --locked --no-dev python -m deflation_example.benchmark_structure --output runs/structure
```

The first command compares rank-100 deflated PDAS with persistent-resource AmgX PDAS on a `24^3` heterogeneous CHT grid. Five complete three-target sequences are independently repeated per method. Each measured sequence includes assembly, reference/resource creation, every inactive-set update and inner solve, transfers, verification and cleanup. Both methods use the same outer criterion. Common bound calibration and process initialization are reported separately. All outer histories and inner failures are retained. These totals are not reconstructed from per-instance medians.

The CPU-only second command checks a nontrivial restriction bound and compares Jacobi-scaled with unscaled Ritz selection on the same rank-300 CHT candidate space, retaining rank 100. It records every selection-inclusive repetition, effective rank and original residual. The controlled bound example uses unscaled coordinates throughout; it is an algebraic SPD example, not an additional PDE validation case.

The ordinary 2D demo intentionally retains its original angles for comparison
with the manuscript fixtures: 0 and pi/2 repeat one symmetric problem. The
moving-Gaussian GPU benchmark and all three CHT targets are distinct problems.

## Extended complete CHT protocol

The predeclared configuration is `conf/benchmark_extended.yaml`. It selects
sixteen targets on each of the `24^3` and `32^3` grids, rank 100, five
independent repetitions and three matched initialization policies. Run:

```bash
uv run --no-sync python -m deflation_example.benchmark_extended --cfg job
uv run --no-sync python -m deflation_example.benchmark_extended output=runs/extended-cht
```

Each target is a weighted sum of four Gaussians with width 0.12. For
`t = m/15`, the horizontal rotation is `1.2*t`, the vertical translation is
`0.035*(2*t-1)`, and the weights are `[1, 0.8+0.15*t, 0.65, 0.45]`.
Every target has a different unordered set of Gaussian weights. A spatial
symmetry cannot map one target into another. Exact centers and amplitudes
are written to `protocol.json` before calibration or timing starts.

The physical bound is calibrated once on grid 24 at `t=0.5`, using eight
PDAS bisections toward 20% activity. The resulting scalar is fixed on both
grids and all targets. Each full repetition starts with empty history.
`cold` uses an empty active set for every target and zero inner guesses.
`outer` passes the previous accepted active set and keeps zero inner guesses.
`outer_inner` also passes the previous full state, restricts it to each
new inactive system, and updates it after every inner solve. Both competitors
have identical access to their own previously accepted solutions.

The native AmgX convergence rule is `ABSOLUTE` for this experiment.
The original right-hand side and initial solution are uploaded unchanged.
A per-solve configuration sets the native threshold to the internal relative
tolerance times the right-hand-side norm. Base configuration and resources
remain persistent; the additional solver configuration is included in handle
creation and cleanup costs. This gives RHS-relative stopping for every
initial guess. This uses the documented
[AmgX absolute residual rule](https://github.com/NVIDIA/AMGX/blob/v2.5.0/src/convergence/absolute.cu)
and [PyAMGX supplied-initial-guess interface](https://github.com/shwina/pyamgx/blob/master/pyamgx/Solver.pyx).
Both solvers require an original CPU residual of `1e-10` and every PDAS
KKT component below `1e-8`. The limits are 10,000 inner and 100 outer steps.
Deflation triggers a fresh residual check when its recurrence reaches
`1e-10`. AmgX uses a native target of `1e-11` because its independent CPU
check follows the native solve. Both methods use the same final original
residual limit of `1e-10`.
GPU QR, the `1e-12` numerical-rank threshold, the `1e10` coarse-condition
limit and the 1,000-iteration residual refresh follow the current method.

Each sequence includes assembly, reference or persistent-resource creation,
every target, inactive-set update, inner solve, transfer, verification and
final cleanup. Components partition this whole interval; `remaining_host_work`
contains callback, loop and report-assembly overhead. Kernel components are
expanded once in this partition. Calibration, process initialization and
finalization are separate records. Add these three common costs to a
sequence total to obtain a single-deployment total. JSON disk writes occur
between measured sequences. No warmup solve is omitted from these totals.

The benchmark retains every target, including failures. A failed target
clears warm-start history for the next target. Every sequence has its own
raw file, hash, target cumulative costs, outer history, inner diagnostics
and all five final KKT components. Summary medians use independently timed
complete sequences. The original three-target benchmark remains available
through `benchmark_sequence` and the version 0.2.0 source.

Protocol version 1 normalized the AmgX right-hand side and solution and used
the same `1e-10` internal and acceptance tolerances. Some native-success
solves failed the original CPU residual check. Its complete raw experiment
is retained. Protocol version 2 removes this rescaling and predeclares the
same factor-0.1 internal stopping margin for both methods. Targets, bounds,
ranks, warm-start policies, failure handling and final acceptance criteria
remain unchanged. At `32^3`, the tighter deflation trigger exposed breakdown
before accepted convergence in some solves. Protocol version 3 retains the
`1e-10` deflation trigger and fresh checks during iteration, and keeps the
AmgX native margin. This matches the final accuracy criterion through the
two backends' different verification mechanisms. Every attempt from all
three protocols is retained and reported separately.

## Matrix-free rank controls

```bash
uv run --no-sync python -m deflation_example.benchmark_matrix_free --output runs/matrix-free
```

The fixed protocol uses grids 32, 64, 102 and 128 with ranks 0, 20 and 200,
one run per pair, a 10,000-iteration cap and residual tolerance `1e-10`.
Every rank receives the same supplied inactive slab, random manufactured
solution, right-hand side and zero initial guess. Hashes verify these matches.
The normal operator uses full-domain stencil applications and zero extension.
An independently assembled CPU Laplacian and its transpose verify the final
original residual at every size. The GPU residual is also recomputed.
Rank zero uses Jacobi-CG. All ranks use the current projected-direction
recurrence, residual refresh and rank/conditioning safeguards. Failed and
capped runs remain in `results.json`.

The total includes operator/basis construction, QR, coarse setup, iteration,
return transfer, independent CPU verification and cleanup. Initialization is
separate. GPU memory is the per-run PyTorch allocator peak. Host RSS is the
process high-water mark and can include an earlier run's allocation. These
are manufactured kernel tests with prescribed masks.

## PDE separation and previous-system recycling

```bash
uv run --locked python -m deflation_example.benchmark_controls --output runs/controls
```

This command writes its protocol before calculations. The PDE example uses
`I + 0.001 L.T L`, one reference mode, grids 4, 6 and 12, and corner squares
of widths 1, 2 and 3. It reports the sufficient bound and actual angle in
unscaled coordinates, including inconclusive cases. These masks are prescribed.

The recycling control uses seven distinct targets in each of diffusion at
`16^2`, frozen thermal at `16^2`, and CHT at `8^3`. CPU PDAS supplies each
fixed inactive system. Both methods use the same projected-direction CG,
zero initial guesses and residual acceptance. Three independently repeated
kernel sequences include all solver calls. The first target uses rank zero
for both methods. The history control retains up to 100 latest CG search
directions, orthonormalizes them, and retains their 20 lowest previous-system
Ritz directions. It embeds them by zero extension and restricts them to the
next inactive set. Each pair uses the smaller of 20 and the transferred
numerical rank. Any additional coarse-condition fallback is explicitly
recorded and marks the pair's deployed ranks as unmatched.

Timing includes transfer, every CG call and construction of the next history
space. Reference construction is charged once to fixed-reference reuse.
Calibration and common PDAS mask construction are separate. These costs are
sums of measured kernels within each repeated control sequence; they exclude
complete optimization. A failed kernel clears the retained history. All
attempts remain visible. The comparison tests this bounded-history policy.

## Check recorded evidence and regenerate presentation files

The repository contains complete records under `examples/benchmarks/extended`.
The `cht_v1`, `cht_v2` and `cht_v3` directories keep the three protocols
separate. Each contains its specification, sequence index and every raw
sequence file. The validator accepts a complete experiment containing failures
and reports the failed counts explicitly. It rejects missing repetitions,
changed hashes, mismatched targets and invalid acceptance or timing claims.

```bash
uv run --locked --extra plot python -m deflation_example.benchmark_extended_report \
  examples/benchmarks/extended/cht_v3/results.json \
  --startup examples/benchmarks/extended/startup.json \
  --output runs/cht-report --plot --require-clean-source
```

This exports sequence-level CSV, checked summary JSON and a cumulative PDF.
Thin curves are individual complete measurements; thick curves are pointwise
medians of their measured cumulative prefixes. The final point includes
cleanup. These curves are distinct from sums of per-instance median kernels
in the separate Laplacian report.
The clean-source flag enforces a recorded clean Git source for publication
checks. Omit it for reports from an installed wheel; source-module hashes
remain in those reports even when Git metadata is unavailable.

To regenerate manuscript tables from a clean checkout of both repositories,
run the following in `opt_control` (adjust only the sibling repository path):

```bash
python3 tools/extended_results.py \
  --controls ../deflation-example/examples/benchmarks/extended/controls.json \
  --scale ../deflation-example/examples/benchmarks/extended/scale.json \
  --startup ../deflation-example/examples/benchmarks/extended/startup.json \
  --cht-v1 ../deflation-example/examples/benchmarks/extended/cht_v1/results.json \
  --cht-v2 ../deflation-example/examples/benchmarks/extended/cht_v2/results.json \
  --cht-v3 ../deflation-example/examples/benchmarks/extended/cht_v3/results.json
python3 tools/extended_results.py --check
```

The importer verifies hashes and refuses to replace a different existing raw
record. CPU controls and the scale study include every declared configuration;
rank-zero iteration caps remain in the resulting tables. Timing reproduction
requires the stated hardware and software environment. Numerical acceptance
and report validation remain independently checkable.

## Fresh-process startup and shutdown

```bash
uv run --no-sync python -m deflation_example.benchmark_startup --output runs/startup
```

Five fresh child processes load the same scientific and GPU modules, initialize
the CUDA and AmgX runtimes, finalize them, and exit. The parent measures each
whole child lifetime. Child intervals identify scientific imports, GPU loading,
runtime initialization and finalization; the remaining parent interval covers
Python launch, serialization and exit. These components sum to each total.
This separate measurement includes imports absent from the CHT runtime timer.
The parent launcher is outside the measured child workload.

For a cost including preparation, add this separately measured median startup
and shutdown cost, the once-per-study bound calibration, and a measured complete
CHT sequence. The smaller CUDA/AmgX initialization/finalization intervals already
inside the startup experiment are excluded from that sum to avoid double counting.
The resulting sum is distinct from the independently repeated sequence interval.
