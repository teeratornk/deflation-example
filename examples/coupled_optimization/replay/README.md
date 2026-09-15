# Coupled trajectory consistency

These checks compare a saved optimization trajectory with the response to its
unchanged heat source. They use the same mesh, thermal properties, boundaries,
initial conditions and physical time intervals. This example supplies numerical
diagnostics; the original optimization timings remain attached to their sources.

Run from a checkout containing `deflation_example.coupled_replay_check`, on an
allocated compute node. Replace the input paths with the verified baseline and
optimization outputs described in the [parent example](../README.md).

```bash
uv sync --frozen
uv run python -m deflation_example.coupled_replay_check \
  --baseline runs/stabilized-baseline --optimization runs/optimized \
  --method reference --threads 8 --output runs/saved-equations
```

The command recomputes momentum, continuity, thermal residuals, mass balance and
discrete energy balance at every saved time slab. A direct thermal solve with the
saved velocity and preceding state checks source recovery independently of the
optimization recurrence. It leaves the saved fields unchanged.

## Fixed-source tolerance study

The original forward protocol uses a coupled residual target of `1e-8` and an
internal momentum target of `1e-9`. The following matched runs tighten both.
The optimizer's final residual and KKT criteria remain unchanged. Temperature is
never clipped, and the control is never recovered from the replayed response.

```bash
for tolerance in 1e-8 1e-10 1e-11; do
  uv run python -m deflation_example.coupled_resolution \
    --baseline runs/stabilized-baseline --optimization runs/optimized \
    --method reference --subdivision 1 --threads 8 \
    --tolerance "$tolerance" --relaxation 0.5 --coupling-cap 100 \
    --output "runs/replay-$tolerance"
done
uv run python -m deflation_example.coupled_replay_check \
  --baseline runs/stabilized-baseline --optimization runs/optimized \
  --method reference --replay runs/replay-1e-8 --threads 8 \
  --output runs/saved-versus-replay
```

Each output records the source hashes, stopping targets, every step's status,
residual histories and temperature differences. The CLI permits stricter targets
only. The internal momentum target is one tenth of the coupled target. Iteration
caps, nonconvergence and incomplete trajectories remain visible.

With `--replay`, the diagnostic also solves the thermal equation using the saved
velocity and the replay's preceding temperature. It reports that propagated
temperature difference separately from the remaining coupled difference. Their
maximum norms are not additive. This decomposition does not establish which
nonlinear solution branch was reached.

For a local initial-guess comparison, append `--local-slabs 0 15 27` and
`--local-tolerance 1e-10` to the saved-equation command and select a new output
directory. Each selected step uses the exact saved preceding temperature and
velocity. One solve starts from the saved current fields; the other starts from
the saved preceding fields. Both retain the same control and convergence rules.
The output retains every local solve, its residual history, and the temperature
and velocity differences. These local checks separate errors accumulated over
earlier steps from sensitivity to the current initial guess.

Compare the complete tolerance sequence before interpreting temperature-bound
violations or tracking changes. Small equation residuals alone do not bound
trajectory error. A remaining unexplained discrepancy requires further diagnosis
before the coupled example can support an application claim. Temporal or spatial
refinement is a separate check of discretization error.

## Generate the comparison

The transformer example uses a temperature scale of 20 K. After the replays,
regenerate the summary and temperature-difference curves from their saved fields:

```bash
uv run --extra plot python -m deflation_example.coupled_replay_report \
  --optimization-fields runs/optimized/reference-fields.npz \
  --replays runs/replay-1e-8 runs/replay-1e-10 runs/replay-1e-11 \
  --temperature-scale 20 --plot --output runs/replay-summary
```

The report checks the common optimized source, time grid and configuration. It
recomputes each temperature difference and checks it against the saved summary.
Incomplete replays retain their termination status and partial trace; their
full-trajectory difference remains unavailable. A missing run remains a row in
the summary. Scheduler timeouts must be recorded separately when a killed
process leaves its last numerical status as `running`.

## Linearized propagation

For selected saved steps, examine the coupled current/history Jacobian pencil
and the thermal pencil with velocity fixed:

```bash
uv run python -m deflation_example.coupled_step_spectrum \
  --baseline runs/stabilized-baseline --optimization runs/optimized \
  --method reference --slabs 0 15 27 --modes 4 --threads 8 \
  --output runs/step-amplification
```

For the current-state Jacobian `H` and preceding-state derivative `-C`, the map
`H^{-1} C` propagates first-order perturbations while the control is fixed.
The command reports its largest-modulus eigenvalues and original pencil
residuals. A modulus above one identifies a growing mode of that particular
discrete linearization. Changing time-step maps and nonnormal amplification
require separate analysis; these local eigenvalues do not certify stability
or instability of the physical system or the complete trajectory. Eigensolver
iteration caps remain visible alongside any converged eigenpairs.

## Monolithic Newton verification

The separate `coupled_newton_replay` command uses the verified full step
Jacobian with sparse LU and backtracking. It applies the saved source without
reoptimization. An initial-residual check precedes Newton; rejected candidates
leave the previous fields and their residuals intact. Row equilibration and at
most two error corrections verify each linear Newton equation. Final coupled,
mass and energy checks use the original equations.

```bash
uv run python -m deflation_example.coupled_newton_replay --baseline runs/stabilized-baseline --optimization runs/optimized --tolerance 1e-12 --output runs/newton-forward
```

This command checks the original temporal grid and saves all completed steps,
including a terminating failed step. It reports each step's temperature
difference from the optimized trajectory. Its implementation and cost remain
separate from the segregated replays and the optimization timings. Include its
output directory in the replay-summary command to compare both forward
procedures. Agreement on the original grid still requires a separate physical
time-resolution assessment. Add `--subdivision 2` or `--subdivision 4` for
the declared forward time refinements. The source remains piecewise constant
on each original interval and is copied into its substeps. The output compares
original endpoints and the piecewise-linear original trajectory separately,
and evaluates tracking against the same physical target at every new time
level. It reports the predeclared 0.05 K and 1% resolution checks for complete
refined trajectories, while retaining every unsuccessful solve.

Compare completed refinements directly, independently of their differences
from the original optimized trajectory:

```bash
uv run --extra plot python -m deflation_example.coupled_time_resolution_report --replays runs/newton-subdivision2 runs/newton-subdivision4 --temperature-scale 20 --output runs/newton-time-summary --plot
```

Use the temperature scale declared for the saved optimization. This command
checks the common source and physical configuration, recomputes differences
at shared time levels, and compares the tracking integrals using the finer
integral as denominator. It preserves missing and unsuccessful computations.
Agreement at these time levels does not certify feasibility at every
intermediate physical time.

The same monolithic Newton procedure also supports the declared spatial
refinement with an unchanged physical time grid:

```bash
uv run python -m deflation_example.coupled_spatial_resolution --baseline runs/stabilized-baseline --fine-baseline runs/refined-baseline --optimization runs/startup60-reference-pilot --procedure monolithic_newton --tolerance 1e-12 --newton-cap 30 --output runs/newton-spatial
```

The spatial driver checks matching material data, boundary conditions and
physical scales, then transfers the saved signed source by nested P1
interpolation. It retains the computed temperatures, velocities, pressures,
conservation checks and any unsuccessful step. Its default segregated
procedure remains available for reproducing earlier computations.

## Diagnose a stopped spatial replay

The following read-only check uses the saved temperature, velocity, pressure,
preceding fields and applied source at the last stored time step:

```bash
uv run python -m deflation_example.coupled_newton_diagnostic \
  --baseline runs/stabilized-baseline --fine-baseline runs/refined-baseline \
  --optimization runs/optimized --replay runs/newton-spatial \
  --output runs/newton-step-diagnostic
```

It checks the analytic Jacobian against centered differences in a random
direction and the Newton direction. It also reports the original linear
residual, the proposed temperature and velocity increments, and the residual
along that direction. The fixed row-scaled residual and the solver's normalized
merit are reported separately. These quantities help distinguish derivative
errors, inaccurate linear solves and difficulties with nonlinear steps. The
command neither changes a saved field nor continues an unsuccessful trajectory.
Use `--step` to inspect a different stored step.
