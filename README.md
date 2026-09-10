# Body-fitted CHT reference-space study: data version 1

This collection supports complete state-constrained thermal optimization on
an axisymmetric transformer mesh and a three-dimensional bore-in-block mesh.
The numerical records accompany
[deflation-example v0.5.0](https://github.com/teeratornk/deflation-example/tree/v0.5.0)
and its [mesh study guide](https://github.com/teeratornk/deflation-example/blob/v0.5.0/docs/mesh-showcases.md).

## Final comparisons

`final/` contains six primary populations and four rank or time controls.
Each population has 16 desired-temperature targets, four methods and five
independently timed complete sequences per method. All 200 sequences satisfy
the independently recomputed relative linear residual of 1e-10 and all five
KKT thresholds of 1e-8. These sequences contain 3200 accepted target solves;
timing repetitions and rank variants repeat their declared physical problems.

The methods are Jacobi-CG, fixed-reference deflation, recycling that retains
existing coarse vectors and new Jacobi-scaled Ritz directions, and
persistent-resource AmgX. Both outer PDAS and inner solves use matched warm
starts. Each repetition starts with empty solution and recycling history.
Operators, physical bounds and material properties remain fixed within each
sequence. A transient query optimizes all its time levels together.

Reference deflation has the lowest median complete time in the six primary
populations. On the largest engine four-slab trajectory, with 144788 optimized
degrees of freedom, its median is 133.791 s. The corresponding Jacobi,
recycling and AmgX medians are 153.957, 166.499 and 521.214 s. All raw
repetitions, including the close eight-slab engine comparison, are retained.

Complete time includes mesh loading and preparation, assembly, reference or
resource construction, every PDAS step, all inner solves, transfers,
verification and solver cleanup. Common library initialization and warmup
after Python imports, and native finalization, have separate fields. Parent
process wall time additionally includes imports, memory-monitor startup,
serialization and optional field output. The fixed mesh bounds require no
calibration. Sampled host RSS includes fields retained for output; sampled
GPU memory is process allocation, including library caches and AMG memory.

The clean numerical source is
`dc89ae8518ff6a975eadf2f6f3bae1acbd17b83b`. The GPU environment reports installed
distribution metadata 0.3.0; the selected source declares 0.4.0. The recorded
Git identifier and module hashes specify the executed implementation. The
v0.5.0 release distinguishes installed metadata from selected source metadata.
Every final worker uses NVIDIA H200 and Intel Xeon Platinum 8468 hardware,
double precision, four numerical-library CPU threads, and the same native
configuration. `diagnostic/native-environment/native.json` identifies the
AmgX and PyAMGX source and binary versions.

## Independent checks and transfer mechanism

`diagnostic/independent-validation/` contains 12 small physical-parameter
checks against independent bounded-variable least squares. They cover the
complete coarse engine and a transformer inlet-region subproblem in steady,
uniform-step and unequal-step form. The source is `d0b96ab`. The separately
retained algebraic validation uses its earlier parameter choices.

The three named transfer directories contain replay records from complete
optimization traces. Both policies start from the same reference restriction.
Subsequent restrictions either read the full reference or sequentially
zero-extend the preceding restricted values. Both use the same matrices,
right-hand sides, zero initial guesses and independent acceptance. Eight
evenly spaced systems have three alternating timed repetitions each.
These CPU-SVD replay timings include basis transfer and the kernel; independent
direct energy diagnostics are outside that interval. The complete GPU study
uses GPU QR. Both policies retain rank 100 in these three replays. The engine
traces show substantial energy-removal differences; the transformer trace
shows a much smaller difference. Small engine records additionally retain
remaining nonzero deflated spectra where their dimension permits.

## Pilots and outcomes

`pilot/` retains all numerical pilot configurations and attempts. The
`sources/pilot-v1/` through `sources/pilot-v4/` directories freeze their
numerical implementations and packaged inputs. These snapshots preserve
earlier implementation choices and form a separate exploratory population.

- Version 1 tests the initial mesh implementation and records missing native
  initialization and capped transformer solves.
- Version 2 initializes AmgX and compares spatial sizes and trajectories.
  Difficult transformer settings fail the prescribed residual acceptance.
- Version 3 varies rank, temporal construction and regularization. Three
  preparation failures retain their original records alongside separate
  retries. An additional large steady engine pilot is retained.
- Version 4 places reference and recycling factors on the GPU and uses
  Jacobi-scaled temporal selection. It compares mode-dependent and tensor
  constructions and records finer transformer residual failures.

`pilot/finer-transformer-steady/` and `pilot/finer-transformer-transient/`
provide the compact supporting-information failure table. Their unsuccessful
attempt times are not completed-solve speedups. The final repeated protocol
is frozen separately; pilot results are not pooled into its timing summaries.

## Reproduction and file integrity

In a clean code checkout of v0.5.0, with this collection in `mesh-evidence`:

```bash
uv run --locked --extra plot python -m deflation_example.mesh_evidence \
  --records mesh-evidence --output runs/reproduced-mesh-evidence
```

The command checks every listed file against `manifest.json` before
regenerating acceptance-checked tables, complete repetition curves, timing
components, memory screens and geometry/active-set figures. The figures use
packaged numeric meshes and recorded active masks. Optional full state,
control and multiplier arrays are produced by rerunning the benchmark and
are not needed for these submitted figures. The guide gives all GPU rerun
commands and the required native build. Hardware-dependent timings can vary.

The manifest lists filenames, byte counts and SHA-256 hashes. Records retain
source versions, resolved configurations, target hashes, timing repetitions,
numerical ranks, fallbacks, residuals and final KKT components. Numerical
source snapshots retain the MIT license and mesh attribution. Root README
text and the manifest itself are outside the manifest's file hash list.
