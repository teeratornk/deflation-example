# Optimization on body-fitted thermal meshes

The mesh examples extend the Cartesian examples with a 2D axisymmetric
transformer and a 3D gas bore inside a conducting metal block. Each query
optimizes temperature and a distributed thermal source. The transient example
optimizes the complete trajectory, with an upper temperature bound at every
positive time level. Velocity and material properties remain fixed during
optimization.

A separate [forward-assessment example](../examples/coupled_assessment/README.md)
prepares fixed optimized controls and verifies axisymmetric momentum
components for a proposed two-way flow–temperature assessment. Coupled
application results and the material-model check remain pending.

## Run the examples

The small engine example runs on a CPU and uses the packaged mesh:

```bash
uv run --locked python -m deflation_example.benchmark_mesh
uv run --locked python -m deflation_example.benchmark_mesh transient=true output=runs/engine-transient
```

The transformer has 10,849 nodes. Its prescribed through-flow produces a
substantially more difficult thermal operator. Use its separate parameter
preset and retain the complete acceptance report:

```bash
uv run --locked python -m deflation_example.benchmark_mesh --config-name mesh_transformer output=runs/transformer
```

`level=1` subdivides each triangle into four or each tetrahedron into eight.
The mesh hierarchy preserves the material interfaces, baffle thickness, and
boundary labels. The reference uses the packaged coarse mesh and P1
prolongation. `rank` denotes the total deployed candidate count, including
space and time together in transient runs. Numerical rank loss is reported.

Use `spatial_reference=diffusion` for generalized diffusion modes or
`spatial_reference=scaled_schur` for modes of the coarse Jacobi-scaled reduced
operator. The latter includes the prescribed transport. Temporal construction
uses `construction=mode_dependent` or `construction=tensor`.
`temporal_metric=jacobi` selects the temporal directions in Jacobi-scaled
coordinates. `reference_device=cuda` retains reference factors on the GPU and
forms restrictions there. Recycling then retains and transfers its selected
vectors on the same device. Its candidate pool includes the existing coarse
vectors and new projected search directions.

The GPU comparison requires the existing optional PyTorch and native AmgX
installation described in the companion's GPU instructions. Install the
Python extras first, then install the native binding. Use `--no-sync` after
that installation to preserve the binding:

```bash
uv sync --locked --extra gpu --extra plot --extra study
```

After installing the native binding in this environment, run:

```bash
uv run --no-sync python -m deflation_example.benchmark_mesh \
  --config-name mesh_final \
  output=runs/engine-gpu
```

Every method and repetition runs in a fresh process. The four-way GPU workers
perform the same small PyTorch and AmgX warmup. AmgX configuration and resources
persist within the sequence; its hierarchy is constructed for each inactive
matrix. Its native absolute stopping target is
`amgx_factor * rtol * norm(rhs)`. CG uses `cg_factor * rtol` internally. The
default factors are both 0.1. Every method must satisfy the separate original
residual and KKT acceptance checks.

The complete Hessian is applied through its factors. The default GPU path
assembles each inactive Hessian for the shared sparse GPU solver interface.
`matrix_free_inner=true` supports CPU Jacobi, reference, and recycling solves
without assembling the inactive Hessian. These are distinct implementations
and their timings must remain identifiable.

## Finer-transformer residual correction

For new GPU body-fitted studies, we recommend the explicit
`residual_policy=refine` option used by `mesh_refinement_transformer`.
The six primary comparison presets keep `residual_policy=terminal` to reproduce
their measured procedure. The Python `StudySolver` default also remains
`terminal`; callers select `refine` explicitly. This recommendation concerns
the tested residual-gap correction and its unchanged final accuracy checks.

The finer transformer has 42,180 spatial state degrees of freedom, or
168,720 for a four-slab trajectory. Instrumented projected-CG runs identify
negative `r @ z` with positive `p @ B @ p`, full retained rank, and moderate
coarse conditioning. The recurrence residual differs from the independently
recomputed residual near convergence. The diagnostics record both residuals,
their gap, coarse orthogonality, and the exact failing scalar.

The correction preset returns a candidate for independent verification before
restarting the projected recurrence. A rejected candidate defines an error
equation with right-hand side `b - B @ x`. Each error solve starts from zero
and requests relative residual 0.1, with internal target 0.01. The local
acceptance test is `norm(e - B @ delta) / norm(e) <= 0.1`, where
`e = b - B @ x`. The factor-ten internal margin allows for the difference
between internal and independently evaluated residuals. A locally rejected
candidate can still be retained if it improves the original residual.
The local status governs recycling updates; the independently evaluated
original residual governs final acceptance. The updated
solution must satisfy the original `1e-10` residual criterion. PDAS retains
its `1e-8` KKT criterion. This is residual-based
[iterative refinement](https://epubs.siam.org/doi/10.1137/1.9780898718027.ch12).
At most four error equations share the original 50,000-iteration budget.
Stagnation, exhausted budgets, and all local termination statuses are recorded.
A nonpositive `r @ z` permits another error solve after an improving candidate.
Other scalar breakdowns terminate the solve when final acceptance fails.
The original-residual check precedes these continuation decisions, so a
candidate that already meets the final criterion requires no correction.

All four solvers receive this policy. Each error equation rebuilds the
matrix-specific GPU objects and coarse or hierarchy setup. AmgX retains its
resources and receives the remaining cap and local stopping target through
a per-solve configuration. Recycling selects directions after locally
converged Krylov solves. The complete clock charges every correction,
transfer, setup, and independent check.

These commands repeat the original four-target problems five times per method:

```bash
uv run --no-sync python -m deflation_example.benchmark_mesh \
  --config-name mesh_refinement_transformer transient=false output=runs/refined-steady
uv run --no-sync python -m deflation_example.benchmark_mesh \
  --config-name mesh_refinement_transformer transient=true \
  horizon=0.000048444553995823255 output=runs/refined-transient
```

The repeated correction study uses numerical source
`926b37c40935fbbe7134a3ab5f87ebdfddb68837`. Its records remain separate from
the six original sixteen-target benchmark cases. Their `terminal` policy and
frozen timings remain available. The `refine` policy changes the finite-precision
stopping procedure and retains the original discrete equations and final checks.

Run `mesh_breakdown` with a saved protocol to reconstruct the diagnostic:

```bash
git clone --branch mesh-cht-refinement-data-v1 --single-branch \
  https://github.com/teeratornk/deflation-example.git mesh-refinement-evidence
uv run --no-sync python -m deflation_example.mesh_breakdown \
  --protocol mesh-refinement-evidence/diagnostic/steady/protocol.json \
  --output runs/breakdown-steady
```

The instrumented timing includes diagnostic work and snapshot serialization.
Each rejected solve saves its exact CSR matrix, right-hand side, initial guess,
candidate basis, and returned state as a numerical NPZ file. The compact public
evidence contains the JSON traces; the command regenerates the larger snapshots.
The report generator checks the unchanged physical settings, target hashes,
assembly/reference/PDAS source hashes, iteration budgets, and timing partitions.

Regenerate the diagnostic and repeated-comparison tables on a CPU:

```bash
uv run --locked --extra study python tools/summarize_mesh_refinement.py \
  --diagnostic-steady mesh-refinement-evidence/diagnostic/steady \
  --diagnostic-transient mesh-refinement-evidence/diagnostic/transient \
  --repair-steady mesh-refinement-evidence/final/steady \
  --repair-transient mesh-refinement-evidence/final/transient \
  --output runs/refinement-tables
```

## Discretization and units

The temperature equation is

\[
K y=M_\ell u+f.
\]

Here, `K` contains diffusion, prescribed transport, and the declared streamline
diffusion. The positive lumped mass `M_l` defines both tracking and control
weights. Dirichlet temperature values are lifted before elimination. The
reduced operator is

\[
H=W+\alpha A^\top W A,\qquad A=M_\ell^{-1}K.
\]

Restriction retains every row of `A`. The implementation forms `A[:, I]`
when constructing an inactive operator. Backward Euler replaces `A` with its
coupled block temporal operator and uses right-endpoint time weights. Recovery
uses this same equation and its exact transpose.

The transformer uses `(r,z)` coordinates and the measure `2*pi*r dr dz`.
Conductivity is anisotropic in the winding. Oil properties are frozen at
323.15 K. Temperature represents excess above the inlet temperature, scaled
by 20 K. The input manifest records source, capacity, velocity, and time units.
The quadratic Stokes field is integrated with degree-five triangle quadrature
and evaluated exactly on each refined child element.
Temperature is prescribed at the inlet. The other exterior boundaries use
homogeneous natural conditions for the declared thermal bilinear form.

The engine follows the documented bore dimensions, conductivity ratio of
1000, inward interface heat flux, and fixed outer coolant temperature. Its
interface load enters once as a surface heat source in the conforming
temperature equation, giving a conductive-flux jump. Its
prescribed circulation is independent of temperature. Time uses the fluid
diffusion scale; both materials have unit dimensionless thermal capacity.
The temperature scale is `q*L/k_s`, with the numerical physical parameters in
the input manifest. Desired profiles and bounds define idealized tracking
problems; they do not represent measured engine operating trajectories.

Both examples use conforming P1 temperatures and nodal upper bounds. A constant
upper bound at every node also bounds the P1 field throughout each element.
The scalar control acts throughout the optimized spatial domain.

## Verification and output

```bash
uv run --locked python -m deflation_example.validate_mesh --output runs/mesh-verification
uv run --locked --extra plot python -m deflation_example.mesh_presentation \
  --records runs/engine-transient --output runs/engine-figures --fields
```

The independent verification compares PDAS with bounded-variable least squares
on the weighted tracking/control factor. It covers the complete coarse engine
mesh and a transformer inlet-region subproblem, including unequal time steps
and nonzero initial temperature. Unit tests additionally check integration,
affine fields, refinement, the transpose, control recovery, and restriction.

Each sequence writes its resolved configuration, source identifiers, input
hashes, timing components, numerical ranks, active masks, and acceptance
components. Optional field files store desired and optimized temperatures,
controls, multipliers, and active degrees of freedom. Failed sequences retain
their outcomes. Summary generation checks populations, distinct target hashes,
timing sums, and accuracy before reporting accepted timings.

Mesh loading, refinement, assembly, reference construction, every PDAS update,
inner solve, transfer, verification, and solver cleanup belong to complete
sequence cost. Common initialization and finalization are recorded separately.
Field serialization and plotting occur after the timed sequence. Memory uses
sampled process RSS and NVML process allocation when enabled.
The library-preparation-inclusive total adds the recorded initialization
after Python module imports and native finalization. The separately measured
whole-worker wall time also includes module imports, memory-monitor startup,
report serialization, and field output. These boundaries have distinct labels
in the generated summary.

The optional `mesh` dependency regenerates the engine mesh. It is unnecessary
for running the packaged inputs:

```bash
uv run --locked --extra mesh python -m deflation_example.generate_engine_mesh \
  --output runs/regenerated-engine --size 0.12
```

Gmsh may require platform OpenGL system libraries. The packaged examples have
no Gmsh runtime dependency and require no access to AI4PDEs or application data.

## Repeated comparison protocol

The declared population comprises engine refinement levels 1 and 2, each in
steady and four-slab transient form, and the packaged transformer mesh in both
forms. Every configuration uses 16 distinct desired-temperature profiles,
five complete repetitions, four solvers, rank 100, and matched outer and inner
warm starts. The engine uses a dimensionless horizon of 0.1; the transformer
uses a 600-second horizon. Bounds, material properties, and prescribed
velocities remain fixed within each sequence. The profiles use the same
physical-coordinate functions across refinement.

Run the six configurations in separate output directories:

```bash
uv run --no-sync python -m deflation_example.benchmark_mesh --config-name mesh_final level=1 transient=false
uv run --no-sync python -m deflation_example.benchmark_mesh --config-name mesh_final level=1 transient=true
uv run --no-sync python -m deflation_example.benchmark_mesh --config-name mesh_final level=2 transient=false
uv run --no-sync python -m deflation_example.benchmark_mesh --config-name mesh_final level=2 transient=true
uv run --no-sync python -m deflation_example.benchmark_mesh --config-name mesh_final_transformer transient=false
uv run --no-sync python -m deflation_example.benchmark_mesh --config-name mesh_final_transformer transient=true
```

The temporal construction was selected after a four-target pilot compared
mode-dependent and tensor candidates with identical spatial modes, rank,
operator, and accuracy. Their engine costs were close; the final study uses
mode-dependent compression consistently for both geometries. Pilot records
remain separate from the repeated population. Larger-mesh failures and
alternative ranks remain identifiable in the supporting scaling records.

Each worker records actual elapsed cumulative time after each target and the
complete time after cleanup. Cumulative plots use these timestamps from each
independent repetition. `recycle_rank` permits an explicitly declared
alternative recycling budget; its default equals the reference rank.

The matched transfer replay reconstructs the recorded inactive matrices and
right-hand sides. It checks their hashes, the input bundle, and numerical
source files before comparison. Both policies start from the same reference
restriction. Later steps use either full-reference restriction or sequential
zero extension, with no added directions. The following small CPU example
also measures the remaining deflated spectrum:

```bash
uv run --locked python -m deflation_example.benchmark_mesh \
  'methods=[reference]' spatial_reference=scaled_schur output=runs/mesh-trace
uv run --locked python -m deflation_example.mesh_transfer \
  --record runs/mesh-trace/reference-0/record.json --output runs/mesh-transfer
```

The replay reports independent energy-error diagnostics, actual numerical
ranks, constraint activations and releases, and alternating repeated kernel
measurements. Its transfer-inclusive times are distinct from complete
optimization timings.

The complete numerical studies use source commit
`dc89ae8518ff6a975eadf2f6f3bae1acbd17b83b`; the first checked transfer
and validation tools use `774c5beccf1dd0b8788b9471b90440c0504ea424`.
Workers select these source checkouts directly. Their Git identifiers and
per-module hashes identify the implementation. The original GPU environment's
installed-distribution metadata reports version 0.3.0; the selected numerical
source declares version 0.4.0. Version 0.5.0 reports these two metadata fields
separately and preserves the numerical source used in those records.

## Rebuild the comparison artifacts

Install the plotting extra for CPU-only artifact generation. The evidence
checkout contains the six primary populations, four rank/time controls,
independent verification, matched transfer records, and the pilot outcomes.
It also retains the numerical sources used by the pilot implementations.
The manifest covers their exact bytes. This command verifies that manifest
before generating the complete-cost tables, repetition curves, timing
components, and geometry/constraint figure:

```bash
git clone --branch mesh-cht-data-v1 --single-branch \
  https://github.com/teeratornk/deflation-example.git mesh-evidence
uv run --locked --extra plot python -m deflation_example.mesh_evidence \
  --records mesh-evidence --output runs/reproduced-mesh-figures
```

The field figure reconstructs desired temperatures from the packaged inputs
and reads the accepted active masks from the records. It checks the mesh-input
and target hashes. It requires neither a GPU nor the optional full state-field
archives. The full state, control, and multiplier fields remain available
after a new benchmark run for additional inspection.

The code release is `v0.6.2`. The separate `mesh-cht-data-v1` tag freezes
the numerical records and source snapshots without installing them as package
dependencies. The generated memory-budget table is a retrospective screen
over measured ranks 20, 100, and 200. It uses the largest sampled GPU process
allocation over all five repetitions. Execution did not impose these memory
budgets. Host RSS is reported separately and includes retained output fields.

Version 0.5.1 applies source formatting without changing the numerical Python
syntax trees. Frozen records retain their original byte hashes. To rerun the
frozen mesh traces with `mesh_transfer`, use `v0.5.0`, whose checked mesh
modules match those recorded hashes. New runs and their replays use the
current checkout together. The
`mesh_evidence` command regenerates the submitted artifacts from either
release and the frozen data without rerunning optimization.

The rank controls repeat the engine level-2 steady sequence at ranks 20 and
200, matching the recycling rank to each reference rank. The time controls
repeat the transformer and engine level-1 sequences with eight slabs over
their existing horizons. Each control retains all four methods, 16 targets,
five independent repetitions, and the primary accuracy criteria.

```bash
uv run --no-sync python -m deflation_example.benchmark_mesh --config-name mesh_final level=2 rank=20 recycle_rank=20
uv run --no-sync python -m deflation_example.benchmark_mesh --config-name mesh_final level=2 rank=200 recycle_rank=200
uv run --no-sync python -m deflation_example.benchmark_mesh --config-name mesh_final level=1 transient=true slabs=8
uv run --no-sync python -m deflation_example.benchmark_mesh --config-name mesh_final_transformer transient=true slabs=8
```
