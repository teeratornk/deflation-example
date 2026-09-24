# Matched linear space–time control

This comparison freezes the verified isothermal computed velocity used to
initialize the nonlinear transformer study. Temperature-to-flow feedback is
disabled. Geometry, material coefficients, consistent thermal source and
storage actions, boundary data and objective weights are unchanged. The
temperature trajectory remains one all-at-once constrained problem, with
backward-Euler coupling between successive slabs and the same initial field.

Each query is an exact linear-quadratic optimization. PDAS solves it directly;
there are no nonlinear outer iterations or repeated momentum solves. The
source action is factored and applied as an operator, not replaced by a
lumped source in the optimization equations. The lumped approximation is
used only to construct the preconditioner. The final control is checked in
independently assembled thermal equations, and the adjoint transpose and
weighted KKT conditions are verified separately.

## Declared comparison

The bounded [regularization screen](../coupled_regularization/README.md)
supplies the single alpha/rank choice. The linear comparison does not select
a new setting after observing its results. It retains targets 7, 15 and 14,
64 slabs over 600 seconds, the smooth 60-second target startup, fixed
temperature bounds, and the same assessed initial temperature supplied to
the nonlinear comparison. Saved coupled velocities are ignored.

Before freezing the isothermal velocity, the loader independently verifies it
in the actual time-discrete momentum equations. If needed, single Newton
corrections solve the same steady isothermal equations, retaining candidates
that improve the independently recomputed time-discrete residual. The final
criterion remains `1e-12`; the internal step requests a factor-ten margin.
The steady-baseline check remains `1e-8`, as in the nonlinear loader. A relative
velocity change greater than `1e-6`, stagnation above final accuracy or an
exhausted correction budget stops the comparison. The resulting field, all
attempts, residuals and cost are recorded; preparation is charged equally
within each complete sequence. The saved baseline remains unchanged.

Four arms compare Jacobi-CG, three-sweep frozen preconditioning, improved
recycling with that inverse, and a fixed full-domain reference with that
inverse. Reference construction is charged once per sequence. Recycling
starts without history and retains existing coarse vectors alongside new
directions, with selection in the preconditioned energy metric. Only verified
previous optima supply warm starts. All methods meet the same original
residual and weighted KKT thresholds, `1e-10` and `1e-8`, respectively.
The thermal-equation threshold is `1e-12`; conservation uses `1e-6`.

Use fresh eight-thread CPU processes, at most four concurrently, for three
development repetitions per arm. Preserve all outcomes and scheduler states.
Five new confirmation repetitions require verified complete-population
solution agreement and lower reference median time than the fastest tested
alternative. Linear and nonlinear optimization timings have separate
comparators; their times are never pooled into a solver speedup.

## Reproduction

The baseline, nominal optimization and assessed initial snapshot are the
same checksum-verified inputs as the complete nonlinear comparison described
in the linked regularization example. Run in a compute-node allocation:

```sh
uv sync --frozen --extra study --extra plot
uv run --no-sync pytest tests/test_linear_spacetime.py
uv run --no-sync python -m deflation_example.linear_spacetime_complete settings \
  --screen regularization-summary/summary.json --output linear-settings.json
uv run --no-sync python -m deflation_example.linear_spacetime_complete run \
  --screen regularization-summary/summary.json --settings linear-settings.json \
  --optimization NOMINAL_OPTIMIZATION --baseline BASELINE \
  --initial-snapshot INITIAL_SNAPSHOT --initial-assessment ASSESSMENT.json \
  --arm reference --repetition 0 --output linear-reference-repeat0
```

Run `jacobi`, `frozen`, `recycling` and `reference` for repetitions 0–2, with
cyclic arm ordering. Every complete record includes direct PDAS histories,
original-system residuals, retained ranks, independent equation and KKT
checks, component times and sampled process allocation. Failed or capped
outcomes cannot supply a completed-solve speedup. Generate the matched summary:

```sh
uv run --no-sync python -m deflation_example.coupled_confirmation_report \
  --settings linear-settings.json --records RUN1/record.json RUN2/record.json \
  RUN3/record.json RUN4/record.json RUN5/record.json RUN6/record.json \
  RUN7/record.json RUN8/record.json RUN9/record.json RUN10/record.json \
  RUN11/record.json RUN12/record.json --output linear-development-summary
```

The complete timer includes model assembly, reference construction, all
active-set solves, transfers, verification and cleanup. Common flow
calibration and process preparation are reported separately. Field output
follows the timer; all targets and slabs are saved. Predeclared illustrations
use targets 7 and 14 at levels 7, 32 and 64, corresponding to 65.625, 300 and
600 seconds. Each illustration must label its actual time, temperature units
and active bounds. No linear results are asserted before these runs finish.

After the complete comparison, use the same `--settings` and twelve `--records`
arguments with `deflation_example.coupled_confirmation_figures` to draw every
repetition, an additive cost breakdown and sampled memory. Its title identifies
prescribed-flow physics. For saved reference repetition 0, draw the predeclared
target positions 0 and 2 using:

```sh
uv run --no-sync python -m deflation_example.coupled_figures \
  --baseline BASELINE --optimization linear-reference-repeat0 --method reference \
  --target-position 0 --time-indices 6 31 63 --show-active-sets \
  --format pdf --output linear-target7-fields
uv run --no-sync python -m deflation_example.coupled_figures \
  --baseline BASELINE --optimization linear-reference-repeat0 --method reference \
  --target-position 2 --time-indices 6 31 63 --show-active-sets \
  --format pdf --output linear-target14-fields
```

Command-line time indices are zero-based. These figures retain the saved
temperature and signed control without clipping. Separate panels show lower
and upper active constraints. If a selected target failed verification, the
plot command refuses it; another successful target is not substituted.
