# Final coupled comparison

The [study design](protocol.json) fixes the physical problem, comparison
population, numerical targets and selection rules. Spatial resolution, temporal
resolution and rank selection are still in progress. `frozen_selection` is
therefore `null`; the existing screening times are not final-study measurements.
The prescribed-flow results remain a separate population.

Each sequence contains five constrained optimization problems and three distinct
desired trajectories. The lower bound is 337.3 K. The nominal upper bound is
357.3 K, with 353.3 K and 361.3 K variants of target 7. All methods use the same
600-second horizon and smooth 60-second startup. No safety margin is applied.
All methods use the positive frozen-velocity preconditioning diagonal, including
damping. The rank-zero method is identified as `jacobi` in the configuration.
The same diagonal defines the coordinates used for recycling selection.

## Resolution and feasibility

Run the [saved-source verification and refinement example](../replay/README.md)
on allocated compute resources. The monolithic forward solve preserves the
optimized source and original physical model. For example, compare three nested
time grids using the same forward tolerance. Use the versioned reporting source
and keep the sources used to generate earlier measurements unchanged:

```bash
git clone https://github.com/teeratornk/deflation-example.git
cd deflation-example
git checkout coupled-final-design-v1
uv sync --frozen --extra study --extra plot
for subdivision in 2 4 8; do
  uv run python -m deflation_example.coupled_newton_replay \
    --baseline runs/stabilized-baseline --optimization runs/optimized \
    --method reference --subdivision "$subdivision" --tolerance 1e-12 \
    --cap 30 --threads 8 --output "runs/time-$subdivision"
done
uv run --extra plot python -m deflation_example.coupled_time_resolution_report \
  --replays runs/time-2 runs/time-4 runs/time-8 --temperature-scale 20 --initial-value 0 \
  --output runs/time-comparison --plot
```

The initial temperature in this model is uniformly zero in dimensionless
coordinates (341.3 K). The report recomputes differences at shared physical
times and at every finer level against linear interpolation of the coarse
trajectory, including the initial condition. It requires the
finest comparison to meet both 0.05 K and 1% tracking thresholds, with decreasing
changes across the last two comparisons. At least three complete grids are
needed to assess that trend. Supply every declared run, including missing or
unsuccessful refinements. The summary retains their status and does not mark an
incomplete design as resolved. It also retains recorded upper and lower bound
excesses; a missing measurement remains missing.

Check spatial refinement at matching physical times using
`coupled_spatial_resolution --coarse-replay`, as described in the linked example.
Assess space and time separately. A bound excess on a refined forward grid
requires refinement and reoptimization under the same physical bounds. Repeat
same-grid and finer-grid forward checks with the new source. The resolution
thresholds describe discrete differences, not a bound on the continuum solution.

## Freeze the complete comparison

Before timed sequences, record the resolved mesh and time grid, baseline hashes,
derivative checks, source version, numerical backends, memory measurements and
rank-screen outputs in a versioned selection record. The declared rank screen
uses the nominal problem with ranks 20, 100 and 200, together with rank-zero
diagonally preconditioned CG. Select each coarse method by verified setup-inclusive time and retain
all screening outcomes. If different ranks are selected, retain a matched-rank
comparison as supporting evidence.

Run each method in a fresh process with the same isolated resources. Follow the
five declared solver orders and keep all scheduler outcomes. The existing
`deflation_example.coupled_sequence` runner supplies the complete sequence,
independent equation and optimality checks, timing partition, fields and sampled
process memory. Merge the design's `common` settings and the frozen grid and
rank choices with its Hydra configuration. Do not use its default 64-slab value
as a substitute for a resolved selection. Reproduction commands for the final
selected discretization will accompany the frozen selection and results.

## Summarize complete sequences

The standard reporter requires identical rank and window settings. For a
predeclared method-specific rank comparison, supply the frozen mapping instead:

```bash
uv run python -m deflation_example.coupled_report \
  runs/final/jacobi/rep-0 runs/final/reference/rep-0 runs/final/recycling/rep-0 \
  --repetitions 5 --rank-policy runs/final/rank-policy.json \
  --output runs/final-summary
```

Append the other twelve run directories when they are available. With only
three records, this command explicitly reports an incomplete population and
gives no complete-population speedup. The rank-policy JSON must name `jacobi`,
`reference` and `recycling`, each with integer `rank` and `recycle_window`.
The `jacobi` method has rank zero; its positive window setting is unused. The two coarse
methods require positive ranks. The reporter permits only these two declared
budget differences. Sources, hardware, physical inputs, tolerances, starts and
timing boundaries must still match. It stores the policy's checksum with the
summary. Do not infer or change the policy after examining the final timings.

Every verified label requires all five optimality components and the independent
equation checks. Medians and observed ranges are computed from complete verified
sequences. Failed, capped and missing repetitions remain visible, and the
fastest-tested-alternative ratio requires all fifteen declared sequences to
meet the final criteria. Such a ratio concerns the frozen discrete comparison;
the source-resolution assessment remains a separate requirement.
