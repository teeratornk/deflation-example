# Coupled completion checks

These examples diagnose and assess the corrected coupled formulation before
new optimization comparisons. They preserve the prescribed-flow study and all
earlier coupled attempts. The [protocol](completion_protocol.json) specifies
the physical problem, numerical criteria and subsequent gates. It is a plan,
not a statement that the final coupled comparison is complete.

## Numerical source and tests

Use the pinned numerical source for both generation and restart:

```sh
git clone https://github.com/teeratornk/deflation-example.git
cd deflation-example
git checkout 1126746918d2f66c74d04de62bcb32f9c4aa1d98
uv sync --frozen --group dev --extra study --extra plot
uv run pytest tests/test_coupled_bounds.py \
  tests/test_coupled_forward_checkpoint.py tests/test_coupled_forward_restart.py \
  tests/test_coupled_local_diagnostics.py tests/test_coupled_step_spectrum.py \
  tests/test_newton_trust_region.py tests/test_coupled_hybrid_forward.py
```

Run numerical checks on allocated compute resources. The checkpoint tests
cover interrupted writes, changed fields or records, incompatible protocols,
and equality between restarted and uninterrupted small trajectories. The step
tests distinguish pressure from velocity in the optional field limiter. The
full CPU suite and the relevant CuPy tests provide additional regression checks.

## Required fields

The application commands below require a converged baseline and a verified
saved optimization sequence. `runs/optimization/target-00.npz` must contain the
selected source, state and desired trajectory; its `record.json` identifies
the method, target, time grid and matching baseline. These application fields
are not embedded in the wheel. The commands in the parent example generate
baseline and optimization inputs. To reproduce a particular archived run,
use its original fields and hashes rather than a newly optimized substitute.

The following names are user-supplied data locations:

- `runs/baseline`: matching `baseline.npz` and `record.json`.
- `runs/optimization`: saved reference-method target at position zero.
- `runs/history-64`: corrected-formulation forward fields and records on 64 steps.
- `runs/history-256`: the complete corrected-formulation trajectory on 256 steps.

For the declared application, the original saved control has 64 intervals over
600 seconds. Its physical temperature offset is 341.3 K and its scale is 20 K.
The target rises smoothly over the first 60 seconds. The assessment copies each
signed control value unchanged into the subdivisions of its original interval.

## Diagnose the stalled step

```sh
for policy in original deep limited; do
  uv run python -m deflation_example.coupled_local_diagnostics \
    --baseline runs/baseline --optimization runs/optimization \
    --history runs/history-64 --target-position 0 \
    --mode diagnose --interval 11 --policy "$policy" --threads 8 \
    --output "runs/step-diagnosis-$policy"
done
```

The original policy uses 21 backtracking trials. `deep` permits 40 trials.
`limited` starts each search within a 0.05 nondimensional temperature change
and a velocity change of at most 5% of the current peak speed. Pressure has
different units and does not enter the velocity limit; its effect remains in
the original momentum and continuity checks. All procedures use the same
100-iteration Newton cap and final accuracy criteria.

Each run saves the returned Newton fields before checking its Jacobian by
centered finite differences. The initial and returned states have separate
diagnostics. Branch-change counts accompany each perturbation. Large relative
errors in a nearly null direction require interpretation alongside absolute
scales and cancellation; they alone do not establish a derivative error.

## Compare local time steps

```sh
for policy in original deep limited; do
  for interval in 0 11 22 63; do
    uv run python -m deflation_example.coupled_local_diagnostics \
      --baseline runs/baseline --optimization runs/optimization \
      --history runs/history-256 --target-position 0 \
      --mode window --interval "$interval" --policy "$policy" --threads 8 \
      --output "runs/window-$interval-$policy"
  done
done
```

Within each window, all three temporal grids start from the same verified
temperature and flow. The source remains fixed. One full step, two half steps
and four quarter steps cover the identical physical interval. Failed coarse
steps remain visible. Endpoint differences from these windows diagnose local
sensitivity; the complete-trajectory comparison below assesses accumulated
differences.

## Complete temporal refinement and restart

After reviewing the local diagnostics, the retained original-search procedure
can be evaluated on 256, 512 and 1024 steps:

```sh
for subdivision in 4 8 16; do
  uv run python -m deflation_example.coupled_newton_replay \
    --baseline runs/baseline --optimization runs/optimization \
    --method reference --target-position 0 --subdivision "$subdivision" \
    --consistent-stabilization --forward-policy newton_anderson \
    --line-search fixed_scaled --backtrack-cap 21 --cap 100 \
    --tolerance 1e-12 --threads 8 --output "runs/refinement-$subdivision"
done
uv run python -m deflation_example.coupled_time_resolution_report \
  --replays runs/refinement-4 runs/refinement-8 runs/refinement-16 \
  --temperature-scale 20 --initial-value 0 --plot \
  --output runs/refinement-comparison
```

Every returned Newton or Anderson attempt has its own field archive and
verification record. A time-step checkpoint becomes visible only after its
fields and metadata have been written atomically. Field and record checksums
are checked before restart. Source, control, physical configuration and
numerical policy must match. The last retained step is independently verified
again before continuing.

To resume an interrupted 512-step trajectory into a new output directory:

```sh
uv run python -m deflation_example.coupled_newton_replay \
  --baseline runs/baseline --optimization runs/optimization \
  --method reference --target-position 0 --subdivision 8 \
  --consistent-stabilization --forward-policy newton_anderson \
  --line-search fixed_scaled --backtrack-cap 21 --cap 100 \
  --tolerance 1e-12 --threads 8 --resume-from runs/refinement-8 \
  --output runs/refinement-8-resumed
```

Use the resumed directory in the comparison when appropriate. Its metadata
identifies the preceding record and restored steps. Its new elapsed time is
the continuation interval, not an uninterrupted complete-run time. Saved
earlier attempts remain available; do not erase their work when accounting
for interrupted or failed runs.

## Resolution-based optimization margin

The optimization configuration adds `temperature_margin_K`, default zero.
It tightens both optimization bounds inward and records the physical and
design bounds separately. Forward assessment always uses the physical bounds.
After verified temporal and spatial comparisons, the declared empirical rule is

```text
margin_K = max(0.01, 2 * (temporal_difference_K + spatial_difference_K))
```

The maximum permitted margin is 0.1 K. If the computed value is larger, the
helper raises an error and further resolution is required; it does not clip
the value. This empirical margin is not a rigorous continuum-error bound.
No margin is inferred automatically from an incomplete comparison.

Reoptimization and the rank/time/memory study remain subsequent gates. A
complete coupled result requires the stated equation, conservation, KKT,
resolution and independent physical-bound checks. The optimization evidence
concerns stationary points of the coupled discrete problem.
