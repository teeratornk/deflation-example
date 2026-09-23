# Temporal-resolution diagnostics

These diagnostics extend the fixed-source checks in the completion protocol.
They preserve the original source, physical bounds and coupled-equation tolerances.
They do not constitute reoptimization or a complete-trajectory resolution certificate.
The original 256-, 512- and 1024-step attempts remain part of the comparison.

## Verified prefixes

The prefix comparison uses only consecutive states whose independently evaluated
momentum, continuity and thermal residuals meet `1e-12`, with mass and energy
defects below `1e-6`. Both grids are truncated to a shared verified endpoint.
Temperature differences include intermediate fine-grid times, using linear
interpolation from the actual initial field. Tracking uses the same physical
target and spatial weights on that common interval. Failed fields are excluded
from these quantities, while the failed run and its stopping time remain visible.

After installing the locked environment with plotting support, run:

```sh
uv sync --locked --group dev --extra plot --extra study
uv run python -m deflation_example.coupled_prefix_diagnostics \
  --baseline "$DATA_ROOT/baseline" \
  --optimization "$DATA_ROOT/optimization/reference/rep-0" \
  --replays "$DATA_ROOT/forward-256" "$DATA_ROOT/forward-512" "$DATA_ROOT/forward-1024" \
  --output "$RESULT_ROOT/prefix-diagnostic" --threads 4
```

`DATA_ROOT` contains the exact baseline, saved control, replay records and arrays;
`RESULT_ROOT` is a new results directory. The tool checks source, configuration
and field identities before comparing results. It writes the complete diagnostic
values and a four-panel plot. Prefix comparisons are labeled as such and cannot
replace the complete-horizon resolution test.

## A failed backward-Euler step

Three bounded diagnostics use the saved first failed step. Each requires the
replay's immutable `checkpoints/` records and rechecks the preceding state against
its original equations.

```sh
for policy in diagnose half_predictor anderson_previous; do
  uv run python -m deflation_example.coupled_late_step \
    --baseline "$DATA_ROOT/baseline" \
    --optimization "$DATA_ROOT/optimization/reference/rep-0" \
    --replay "$DATA_ROOT/forward-512" --policy "$policy" \
    --output "$RESULT_ROOT/late-$policy" --threads 8
done
```

`diagnose` compares Jacobian actions against centered finite differences at the
previous and stalled states. The output includes absolute, action-relative and
matrix/direction-normalized errors. Action-relative errors can be large for a
nearly null direction even when its absolute discrepancy is small.

`half_predictor` solves two half steps to obtain an initial guess. It then solves
the **original full-step equations** with the original previous state and source.
Half-step convergence alone never qualifies as a repaired full step. A failed
predictor stops this control and remains in its output.

`anderson_previous` starts complete-field Anderson iteration from the preceding
verified temperature and flow, rather than from the stalled Newton fields.
It uses depth 5, relaxation 0.5, at most 300 coupled iterations and 100 momentum
iterations. Newton uses at most 100 iterations and 21 backtracking trials.
Both controls use `1e-12` original-equation and `1e-6` conservation criteria.

Every returned attempt has a hash-bound field snapshot, residual checks,
termination status, history and elapsed time. Diagnostic time includes all
predictor work and is separate from complete-trajectory timing. These commands
neither modify the old replay nor resume a complete trajectory automatically.

## Local propagation

The prefix comparison motivates a separate diagnostic at 75, 150 and 225 s on
the 512- and 1024-step trajectories. These physical times precede, cross and
follow the interval of large temperature differences. They are exploratory
sampling choices based on the prefix results, not an independent validation set.

```sh
uv run python -m deflation_example.coupled_replay_spectrum \
  --baseline "$DATA_ROOT/baseline" \
  --optimization "$DATA_ROOT/optimization/reference/rep-0" \
  --replay "$DATA_ROOT/forward-512" --time 150 --modes 4 \
  --output "$RESULT_ROOT/local-512-150" --threads 8
```

Repeat this command for both time grids and all three times. It rechecks the
selected original equations and records the coupled propagation pencil and
its frozen-temperature momentum and frozen-velocity thermal blocks. All
blocks include the same storage, source weighting and physical time step as
the replay. Each computed eigenpair has an independently evaluated pencil
residual. An eigensolver cap remains visible. A modulus above one indicates
amplification by that local linearized step; it does not establish physical
instability or the growth of the full sequence of changing step maps.

## Second-order temporal control

The nondecreasing backward-Euler differences and growing coupled local modes
motivate a matched BDF2 assessment at 256, 512 and 1024 steps. This changes only
the assessment time integrator. It preserves the 600 s horizon, signed saved
source, material properties, initial fields and final residual criteria. The
optimizer and manuscript timing evidence remain unchanged. Backward Euler
starts the first substep of each original piecewise-constant source interval;
all other substeps use BDF2 for both flow and thermal storage. The tests verify
both storage histories and manufactured second-order convergence with the
consistent thermal weighting before these assessments are run.

```sh
for subdivision in 4 8 16; do
  uv run python -m deflation_example.coupled_newton_replay \
    --baseline "$DATA_ROOT/baseline" \
    --optimization "$DATA_ROOT/optimization/reference/rep-0" \
    --method reference --target-position 0 --subdivision "$subdivision" \
    --time-scheme bdf2 --consistent-stabilization --forward-policy newton_anderson \
    --line-search fixed_scaled --backtrack-cap 21 --cap 100 --tolerance 1e-12 \
    --threads 8 --output "$RESULT_ROOT/bdf2-$subdivision"
done
uv run python -m deflation_example.coupled_time_resolution_report \
  --replays "$RESULT_ROOT/bdf2-4" "$RESULT_ROOT/bdf2-8" "$RESULT_ROOT/bdf2-16" \
  --temperature-scale 20 --initial-value 0 --plot \
  --output "$RESULT_ROOT/bdf2-resolution"
```

Retain every declared run, including nonlinear failures and time limits. The
complete-horizon comparison uses both successive refinement changes, with
0.05 K maximum temperature and 1% relative tracking thresholds. A converged
forward solve alone establishes neither time resolution nor bound satisfaction.
This control does not change the predeclared optimization feasibility margin.
