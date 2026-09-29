# Complete coupled optimization on nested time grids

This study reduces the number of time slabs before returning to longer discrete
trajectories. The physical horizon remains 600 seconds, with a smooth 60-second
target startup. The prescribed physical bounds, full thermal feedback and
spatial mesh remain unchanged. These are discrete optimization tests; reducing
the slab count does not establish temporal resolution of the physical response.

The predeclared development grid uses 16 slabs, regularization parameters
`1e-14`, `1e-12`, `1e-11`, and reference ranks `0`, `8`, `16`, `32`.
Rank zero uses the same three-sweep velocity-frozen preconditioner without
coarse-space processing. The nonzero references use 48 energy-metric Krylov
steps at the common initial temperature. Their construction is charged to each
complete run. A positive complete-time result requires verified optimization,
not only fewer inner iterations.

## Inputs and verification

Use the checksum-bound initial snapshot, its derivative assessment, physical
baseline and source optimization record from the associated study dataset.
The commands do not download missing data. `SOURCE_RECORD` identifies the
original 64-slab attempt; it does not imply that attempt reached optimality.

```sh
uv sync --locked --extra study
uv run --no-sync python -m deflation_example.coupled_small_study verify \
  --source-record SOURCE_RECORD --baseline BASELINE \
  --initial-snapshot SNAPSHOT --initial-assessment ASSESSMENT \
  --slabs 16 --alpha 1e-11 --budget-seconds 7200 \
  --output OUTPUT/verification
```

The transfer selects coincident endpoints of nested uniform backward-Euler
grids. It imports temperature and flow initial guesses only. The new problem
recomputes flow, recovered control, derivatives and adjoints. A source-snapshot
assessment does not certify the new discretization. Each alpha receives a new
verification record, including absolute and normalized stationarity.

## Matched complete runs

```sh
uv run --no-sync python -m deflation_example.coupled_small_study run \
  --source-record SOURCE_RECORD --baseline BASELINE \
  --initial-snapshot SNAPSHOT --initial-assessment ASSESSMENT \
  --slabs 16 --alpha 1e-11 --rank 0 --device cpu \
  --verification OUTPUT/verification/record.json --budget-seconds 14400 \
  --output OUTPUT/rank-zero
```

Repeat with each declared nonzero `--rank`, a separate output directory and
otherwise identical settings. The gate checks the numerical source and physical
configuration. `--device hybrid` uses CPU sparse operations and GPU coarse
operations; `--device cuda` uses the resident GPU implementation. Both require
the optional `coupled-gpu` environment and an allocated GPU.

The four-hour optimization allowance starts after initialization and reference
construction; these costs remain in the complete elapsed time. Preparation has
its own four-hour cap. Cooperative momentum deadlines act between sparse solves,
not inside an in-progress external factorization. Every incomplete run retains
its termination status. Check verified runs with `coupled_projected_report` as
described in [projected quadratic verification](projected_repair.md).

Development uses target 7. After choosing and freezing a verified setting,
`--sequence nearby` selects 7, 8, 9 and `--sequence stress` selects 7, 15, 14.
Stress cases remain separate from the primary sequence. The scale-up grids use
32 and 64 slabs at the same horizon and require their own verification.

Retain all expected output paths when generating the development summary:

```sh
uv run --no-sync python -m deflation_example.coupled_small_report \
  --records OUTPUT/rank-zero/record.json OUTPUT/rank-eight/record.json \
  OUTPUT/rank-sixteen/record.json OUTPUT/rank-thirty-two/record.json \
  --plot --output OUTPUT/summary
```

Missing, failed and capped records remain visible. Ratios require verified
paired solves with matching physical settings, source, initialization and
backend. The initial single-run ratios guide the next experiment; they do not
replace independent complete-sequence timing repetitions.

## Momentum and derivative diagnostics

```sh
uv run --no-sync python -m deflation_example.coupled_flow_replay trial \
  --record SOURCE_RECORD --trace OPTIMIZATION/inactive-trace-00 \
  --baseline BASELINE --quadratic 1 --trial 0 --budget-seconds 7200 \
  --output OUTPUT/flow-replay
uv run --no-sync python -m deflation_example.coupled_flow_replay derivatives \
  --record SOURCE_RECORD --trace OPTIMIZATION/inactive-trace-00 \
  --baseline BASELINE --quadratic 0 --budget-seconds 7200 \
  --output OUTPUT/initial-derivatives
```

The replay reconstructs a trial from consecutive retained temperatures and the
recorded accepted step fraction. It verifies the increment, physical bounds and
input checksums. This reconstructed field is distinguished from a saved trial.
Direct Newton and residual-load continuation receive identical inputs and final
equation tolerances. Continuation must remove its artificial load completely.
All unsuccessful stages remain in the output. Its use in optimization is opt-in
through `--continuation` and requires a matching initial verification record.

The separate replay option `--polish-load` solves the zero-fraction artificial
load problem when its initial residual exceeds the unchanged flow tolerance.
This tests a failure observed before continuation could advance. It does not
relax the initialization or final physical-equation criteria, and it is disabled
in the frozen direct-Newton optimization study. Its additional Newton solve and
all continuation stages are included in the replay time.
