# Coupled reference retention

This study separates reference quality from processing cost. The physical
comparison retains the existing transformer model, 600 s horizon, smooth
60 s startup, temperature bounds, material inputs, and final accuracy criteria.
Feedback multipliers 0 and 0.5 are diagnostic controls only; the application
comparison uses multiplier 1. Prior measurements remain unchanged.

## Install and verify

Use the source identifier recorded with each output. From a clean checkout:

```sh
uv sync --frozen --extra study --extra plot
uv run --no-sync pytest tests/test_coupled_trace.py tests/test_coupled_selected_reference.py tests/test_coupled_retention_replay.py tests/test_coupled_retention_report.py -m 'not gpu'
```

On an allocated NVIDIA GPU, add `--extra coupled-gpu` to `uv sync` and run
`uv run --no-sync pytest tests/test_coupled_selected_reference.py -m gpu`.
Numerical runs belong on compute nodes. The input names below are directories
or files supplied by the user; no machine-local paths are embedded in presets.

## Capture the actual inner problems

```sh
uv run --no-sync python -m deflation_example.coupled_retention capture \
  --baseline BASELINE --optimization VERIFIED_JACOBI_RUN \
  --initial-snapshot INITIAL_SNAPSHOT --initial-assessment INITIAL_ASSESSMENT.json \
  --output CAPTURE
```

This reruns the verified single-target initialization and records every
attempted inactive solve, including failed and correction solves. Temperatures,
flow guesses, masks, loads, initial iterates, damping, and secants are checksum
bound. Capture includes I/O overhead and supplies no publication timing.
Updates 0–2 form the selection set; all subsequent updates are held out.

## Construct fixed references

```sh
uv run --no-sync python -m deflation_example.coupled_retention_replay bank \
  --trace CAPTURE/linear-systems --baseline BASELINE --device hybrid --width 5 \
  --output BANK
```

The bank contains the original thermal policies at ranks 20, 50, 100, and 200,
plus a 400-direction candidate pool. Coupled selection uses the initial full
Gauss–Newton operator and Jacobi scaling. A thin QR and small SVD reveal rank;
Rayleigh–Ritz selection retains the lowest directions within this candidate
span. Coefficients and the candidate factors remain fixed during later solves.
Rank loss creates no replacement vectors. Nominal evaluation and selection
costs are charged to this policy. These are approximate coupled reference
directions, not fine-grid eigenvectors.

## Matched replays

```sh
uv run --no-sync python -m deflation_example.coupled_retention_replay replay \
  --trace CAPTURE/linear-systems --baseline BASELINE --bank BANK \
  --quadratic 0 --policy nominal_coupled --rank 20 --width 5 \
  --device hybrid --repetitions 3 --output REPLAY
```

Use `--policy jacobi --rank 0` for the baseline and `--policy thermal` for the
existing reference. Compare both reference policies at all four declared ranks
on every selection quadratic. Use three repetitions; preserve all outcomes.
Only after selecting the lowest-cost reference/rank, compare block widths
1, 5, 10, and 20 for that policy. Choose by verified setup-inclusive cost,
then freeze the choice before evaluating every held-out quadratic against
Jacobi and thermal rank 200. Device factors persist within a quadratic and are
released between repeated groups. The rank-zero path performs no coarse work.

Add `--diagnostics` for independently verified initial-error energy removal.
These additional solves and CPU-SVD coarse diagnostics are outside the timing
interval. Small systems receive exact scaled and deflated spectral diagnostics;
large systems receive no unsupported spectral certificate. The recorded warm
start, rather than a zero initial guess, defines the error.

Add `--feedback 0`, `--feedback 0.5`, or `--feedback 1` for the separate sensitivity
control. Each value recomputes flow, source recovery, and derivatives at the
same temperature. All three omit secant corrections and retain the trace masks,
loads, initial guesses, and fixed nominal reference. This fixed-load experiment
isolates operator changes; it is not a complete optimization of the modified
physical model. Keep these records out of the physical-model selection summary.

```sh
uv run --no-sync python -m deflation_example.coupled_retention_report \
  --trace CAPTURE/linear-systems --records REPLAY/record.json OTHER_REPLAY/record.json \
  --partition selection --plot --output SUMMARY
```

The summary requires every declared system and repetition, charges construction
once, and excludes incomplete groups from speedup claims. Its sums of replay
timings are explicitly distinct from independently timed complete optimization.

## Complete optimization and gates

For a fresh single-target initialized run, `coupled_sequence` accepts
`reference_selection=nominal_coupled`, `reference_candidates=400`, and
`reference_selection_tolerance=1e-12` through the `coupled_retention` preset.
Set `capture_linear_systems=false` for timings. Supply the same initial snapshot
and assessment to every method. Nominal flow evaluation and reference selection
are included in construction time, with no extra optimizer history supplied.

After a held-out cost improvement, compare Jacobi, thermal rank 200, the frozen
selected policy, and improved recycling at the selected rank. Run three complete
pilot repetitions. If all methods satisfy the existing residual, equation and
KKT criteria, retain all pilot outcomes and freeze five new complete repetitions.
Report timing ranges, all failures, requested/deployed ranks, and sampled process
memory. A replay benefit alone is insufficient evidence for a complete speedup.

The parallel fixed-source temporal-resolution study remains a separate gate:
the last two refinement pairs must meet 0.05 K temperature and 1% tracking
criteria with nonincreasing changes. Reoptimization requires new forward checks
of its new control. A memory screen precedes any refined GPU optimization.
This example does not claim that the 64-slab trajectory is temporally resolved.
