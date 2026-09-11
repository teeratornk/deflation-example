# Coupled thermal optimization

This example is under verification. It adds temperature-dependent buoyancy to
the axisymmetric transformer model and recomputes the distributed heat source.
The prescribed-flow timing records and their numerical implementations remain
unchanged. No coupled application speedup has been established.
These commands use the development source, rather than the frozen v0.6.2
implementation underlying the earlier timing release.

The trial variable is the complete temperature trajectory. Each evaluation
solves momentum and continuity, then recovers the source from the thermal
equations. Sparse factorizations of the converged momentum Jacobians supply
tangent and transpose solves. The positive-definite Gauss--Newton model contains
the tracking weights and the weighted normal product of the source Jacobian.
Both temporal couplings and velocity-dependent streamline diffusion are included.
The stabilization formula is differentiable within each branch; exact branch
switches are rejected during linearization.

## Verification

From the repository root, run on an allocated compute node:

```bash
uv sync --frozen
uv run pytest tests/test_coupled_derivatives.py tests/test_coupled_optimizer.py tests/test_coupled_flow_solve.py
```

These tests check the steady and nonuniform-time-step tangent, transpose,
objective gradient, positive definiteness, feedback-disabled limit, and failed
flow evaluations. They verify discrete derivatives on small problems. Application
resolution, nonlinear optimality, and solver timings require separate evidence.

## Operating-point pilots

```bash
uv run python -m deflation_example.coupled_pilot inlet_factor=1.0 flow_tolerance=1e-9 output=runs/inlet-1
```

The declared inlet factors are 1, 0.5, 0.25, and 0.125, in that order. Each run
retains viscosity-continuation attempts, the physical steady residuals, and mass
balance. A converged isothermal baseline is a prerequisite for further checks.
Selection also requires coupled convergence and resolution checks, independently
of optimizer performance. Output directories must be new.

A verified stabilized baseline and a short nonlinear solver check can be run as
follows. This short time interval is a discrete verification example; it does
not meet the final study's physical-resolution requirements.

```bash
uv run python -m deflation_example.coupled_pilot inlet_factor=0.125 grad_div_scale=1 flow_tolerance=1e-9 output=runs/stabilized-baseline
uv run python -m deflation_example.coupled_optimize baseline_directory=runs/stabilized-baseline transient=true slabs=1 horizon_s=9.375 mode=derivatives output=runs/short-derivatives
uv run python -m deflation_example.coupled_optimize baseline_directory=runs/stabilized-baseline transient=true slabs=1 horizon_s=9.375 mode=optimize methods=[reference] rank=100 backtracking=quadratic secant_memory=10 output=runs/short-optimization
```

Oil momentum properties use the correlations in
[Li et al., High Voltage 9 (2024), 230–240](https://doi.org/10.1049/hve2.12345).
Thermal properties retain the declared transformer coefficients. These inputs
define a computational model, with prescribed inlet conditions and frozen
properties apart from the Boussinesq forcing.

## Coupled derivative and optimization pilots

After independent baseline verification:

```bash
uv run python -m deflation_example.coupled_optimize baseline_directory=runs/stabilized-baseline mode=derivatives output=runs/derivatives
uv run python -m deflation_example.coupled_optimize baseline_directory=runs/stabilized-baseline mode=optimize output=runs/steady-control
```

The second command requires the application derivative checks to pass first.
Use `transient=true slabs=64 horizon_s=600 target_startup_s=60` for the new
complete 600 s trajectory. These commands
write pilots, not final repeated timing populations. The
[declared study](protocol.json) identifies the final targets, temperature bounds,
comparators, tolerances and resolution checks.

### Smooth target startup

The new coupled study blends the initial field into each desired trajectory
over the first 60 physical seconds. With `s = min(t / 60, 1)`, the weight is
`r = s^2 * (3 - 2*s)` and the desired field is
`initial + r * (original_desired(t) - initial)`. The weight has zero slope at
both endpoints. Every desired value at and after 60 s is unchanged, including
the later peaks and the 600 s endpoint. The same rule applies to all solvers
and temperature bounds.

`coupled_sequence` defaults to `target_startup_s=60`. The pilot command
`coupled_optimize` retains `target_startup_s=0` for reproduction of the original
targets; set it explicitly to 60 for a new transient pilot. Steady runs require
zero startup. Original-target pilots remain separate from ramped pilots.
Saved configurations record the duration, and replay commands evaluate that
same desired field on refined spatial and temporal grids. Records predating
this option retain the original target through a zero-duration default.

For example, verify the derivatives for the new target before optimization:

```bash
uv run python -m deflation_example.coupled_optimize baseline_directory=runs/stabilized-baseline transient=true slabs=64 horizon_s=600 target_startup_s=60 mode=derivatives flow_tolerance=1e-12 flow_cap=40 output=runs/startup60-derivatives
uv run python -m deflation_example.coupled_optimize baseline_directory=runs/stabilized-baseline transient=true slabs=64 horizon_s=600 target_startup_s=60 mode=optimize methods=[reference] rank=100 backtracking=quadratic secant_memory=10 flow_tolerance=1e-12 flow_cap=40 output=runs/startup60-reference-pilot
```

The smooth startup changes the desired field only. It leaves the physical
equations, source recovery, temperature bounds and accuracy criteria unchanged.
Resolution and complete-cost comparisons remain separate verification steps.

The optional `flow_continuation=true` setting introduces a residual load that
makes the initial flow an equilibrium, then removes this artificial load in
verified stages. Its final test uses the original momentum equations. Every
failed stage remains visible. `grad_div_scale` in the baseline pilot defines a
separate momentum discretization and must be included in comparisons and source
identification.

For a separate nested-initialization check, `level=1
seed_directory=runs/coarse-baseline` prolongs a verified matching coarse P2
velocity and P1 pressure before solving the original fine-mesh momentum
equations. Its physical coefficients and inlet speed must match the seed.
The seed changes the initial iterate, not the final equations or tolerance.

## Numerical-policy pilots

The default nonlinear policy remains damped Gauss--Newton with halving
backtracking. `backtracking=quadratic` uses safeguarded quadratic interpolation
after an unsuccessful trial. The separate `secant_memory=10` pilot augments the
current Gauss--Newton operator with ten damped Broyden--Fletcher--Goldfarb--Shanno
(BFGS) secants. Powell damping
maintains positive curvature in exact arithmetic. These settings change the
nonlinear model or line search and must be stated alongside their measurements.
They retain the original-residual and nonlinear KKT criteria. They have not been
selected for the final comparison.

The current quadratic subproblem solver also checks for an unchanged active
set before repeating a linear solve. If its Karush--Kuhn--Tucker (KKT) test still
fails, it solves an error equation and verifies the updated original system.
At most four consecutive fixed-mask corrections are allowed. A correction that
does not improve the quadratic KKT residual terminates with the preceding state
retained. This addresses a measured norm mismatch in the initial 64-slab
subproblem: the original linear residual satisfied its relative tolerance while
the quadratic stationarity test remained above its threshold. The original
attempts remain separate from computations using this correction.

`initial_quadratic.py` reproduces that initial, isothermal quadratic subproblem
from a saved configuration. It measures the quadratic solve alone; it excludes
the subsequent coupled flow evaluations and complete optimization cost.

The coupled line search evaluates the change in the quadratic tracking and
source penalties directly from the two trajectories. This avoids subtracting
nearly equal objective totals. Armijo backtracking supplies ordinary steps. A
separately recorded roundoff safeguard applies when the current normalized KKT
residual is at most `1e-4`, the predicted decrease is below `64 * eps` times the
larger of one and the least-squares objective, and the measured increase stays
within that allowance. It requires a further 10% KKT reduction. The final
`1e-8` KKT check is unchanged. Saved-state diagnostics separately test the effect
of momentum-solve accuracy on objective reproducibility near stationarity.

For a one-level refinement, `reference_baseline_directory` selects a verified
coarse computed-flow baseline. The code checks the material labels, boundary
nodes, spatial measure and nested mesh transfer. It prolongs coarse spatial
directions and constructs temporal factors from the fine frozen thermal
operator. Construction and baseline loading enter the sequence timer; the
additional coarse calibration enters the preparation-inclusive total.

For an exactly zero temperature-rise trajectory with zero thermal boundary
values, the source derivative reduces to the thermal operator. The evaluator
then skips momentum derivative factorizations while still solving and verifying
every original momentum equation. Nonzero trajectories construct the full
coupled derivative factors.

The optional CUDA backend uses the same CPU-factored momentum derivatives and
independently verifies the original system on the CPU:

```bash
uv sync --frozen --extra coupled-gpu
uv run --extra coupled-gpu pytest tests/test_coupled_cuda.py -q
uv run --extra coupled-gpu python -m deflation_example.coupled_optimize baseline_directory=runs/stabilized-baseline device=cuda output=runs/cuda-pilot
```

It requires Linux, a CUDA-compatible GPU and the pinned CuPy dependency.
Triangular analyses are retained while the momentum Jacobian is unchanged;
vector solves use cuSPARSE SpSV and column blocks use SpSM. Each plan retains
its descriptors and analysis workspace. The vector binding uses the cuSPARSE
library already loaded by CuPy, avoiding incompatible opaque descriptors from
different library versions. Plans are confined to their original CUDA stream.
The basis and coarse factorization are rebuilt for each inactive system.
Final optimization checks also reassemble the momentum transpose equations
independently. `evaluation_progress=true` writes per-slab progress for development
pilots; leave it disabled for final timing comparisons.

The diagnostic scripts in this directory have `--help` interfaces:

- `check_saved_gradient.py` tests an objective gradient at a saved nonlinear iterate.
- `momentum_spectrum.py` examines selected linearized momentum eigenvalues. It
  does not certify the stability of the complete spectrum or the physical flow.
- `benchmark_actions.py` compares CPU and CUDA applications of the same fixed
  normal operator, with two warmups and five timings at each block size.

Operator timings exclude complete optimization and must remain separate from
its cost comparisons. All pilot failures and iteration caps remain recorded.

## Complete sequences

After freezing a verified operating point, discretization and nonlinear policy,
run each method and repetition in a fresh process:

```bash
uv sync --frozen --extra study
uv run --extra study python -m deflation_example.coupled_sequence baseline_directory=runs/stabilized-baseline method=reference repetition=0 output=runs/sequence-reference-0
```

The configuration declares five target/bound pairs over the complete 600 s
trajectory. Its current defaults are a development starting point, not a frozen
performance protocol. `method=jacobi` and `method=recycling` select the matched
alternatives. The three methods must use identical nonlinear-policy settings,
accuracy, discretization and start policy. Add `--extra coupled-gpu` and
`device=cuda` for a separate CUDA comparison.

The wall interval includes assembly, reference construction, all optimization
steps, verification and cleanup. Calibration and process preparation are
reported separately. Output serialization follows the timer. All requested
targets appear, including numerical failures and targets skipped after an
unrecoverable error. Only verified solutions supply outer warm starts; recycling
history persists across successful queries under both outer-start policies.
The memory sampler observes host resident memory and the same process's GPU
allocation through NVML. Peaks include retained output arrays and allocator
caches, and sampling can miss short-lived peaks. Repetition numbers identify
independently launched sequences; the runner never substitutes sums of
per-instance medians for complete timings.

Generate a comparison from complete-sequence records with:

```bash
uv run python -m deflation_example.coupled_report runs/sequence-jacobi-0 runs/sequence-reference-0 runs/sequence-recycling-0 --repetitions 1 --output runs/comparison-pilot
```

For final comparisons, supply every method and repetition and use
`--repetitions 5`. The summary checks source hashes, problem settings, accuracy,
timing sums and repetition identifiers. It retains unsuccessful outcomes and
reports a complete-population ratio only when every declared sequence meets
the checks. A single-repetition summary remains a pilot comparison.

Plot stored temperature, source and velocity fields with:

```bash
uv run --extra plot python -m deflation_example.coupled_figures --baseline runs/stabilized-baseline --optimization runs/transient-control --method reference --output runs/coupled-fields
```

For sequence data, add `--target-position 0`. The default time levels are the
first, middle and last stored levels. The figure metadata records their actual
physical times, units, field checksum and interpolation conventions.

## Fixed-control resolution checks

```bash
uv run python -m deflation_example.coupled_resolution --baseline runs/stabilized-baseline --optimization runs/startup60-reference-pilot --method reference --subdivision 1 --output runs/replay-original
uv run python -m deflation_example.coupled_resolution --baseline runs/stabilized-baseline --optimization runs/startup60-reference-pilot --method reference --subdivision 2 --output runs/replay-refined
uv run python -m deflation_example.coupled_spatial_resolution --baseline runs/stabilized-baseline --fine-baseline runs/refined-baseline --optimization runs/transient-control --method reference --output runs/replay-fine-mesh
```

The first command checks agreement with a verified discrete optimizer result.
The second divides every original physical time interval into two substeps.
Both apply the saved source unchanged, using a piecewise-constant temporal
representation and zero source values at prescribed-temperature nodes.
The spatial command uses the immediate nested refinement with the same physical
time steps. It evaluates the original piecewise-linear source on the refined
mesh. Both baselines must use identical physical inputs and momentum forms.
The original baseline must match the checksum stored with the optimized source.
For a complete-sequence output, add `--target-position 0` (or another verified
position); the selected target and bound then come from that sequence record.
The forward solver enforces both momentum and thermal equations at every
substep. It neither reoptimizes the source nor clips the temperature. Endpoint
differences and bound violations are resolution diagnostics; refined forward
responses are not new constrained optima. Spatial refinement and tracking-error
resolution are additional requirements of the declared final protocol.
