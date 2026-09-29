# Cost, retention, GPU execution and repeated coupled sequences

This follow-up uses the small-study physical model, alpha `1e-14`, 16 time
slabs over 600 seconds, the smooth 60-second target startup, and the unchanged
337.3–357.3 K bounds. The baseline and both coarse-space methods use three
velocity-frozen preconditioner sweeps. Reference and recycling ranks are eight.
The reference uses the same 48-step initial-trajectory construction throughout.

The stages distinguish four questions: where execution time is spent, what
information direct restriction retains, how the GPU implementations perform,
and whether the chosen configuration reduces repeated complete-sequence time.
All outcomes remain part of the comparison. A favorable ranking is not a
condition for finishing the study.

## Environment and source

Use an allocated compute node and a clean, versioned checkout. GPU memory
measurement requires PyTorch as well as the CuPy numerical backend:

```sh
uv sync --locked --extra study --extra plot --extra gpu --extra coupled-gpu
uv run --no-sync pytest tests/test_memory.py tests/test_coupled_cuda_frozen.py \
  tests/test_coupled_hybrid_solver.py tests/test_coupled_corrected_ablation_gpu.py -q
```

The subsequent GPU-startup correction establishes a compute context before
starting process-allocation sampling. It is charged to process preparation.
It leaves the numerical equations and stopping criteria unchanged. Frozen CPU
profiles at source `71ee3c7` retain that identifier; new source-matched GPU
runs and their verification use the corrected source recorded in their outputs.
Do not attribute earlier results to the startup correction.

The matched GPU follow-up also uses the host recycling policy with the frozen
preconditioner: it retains old coarse vectors and new directions and selects
alternating extremal Ritz directions from the energy-metric pencil. The previous
resident implementation selected in Jacobi coordinates even with frozen sweeps.
That earlier selection remains attached to its source; the follow-up records the
matched policy explicitly. GPU model and backend versions accompany new runs.

Set the four input paths described in [small_to_large.md](small_to_large.md).
Each new numerical source requires its own `coupled_small_study verify` result.
The following examples assume that verification is `RUNS/verification/record.json`.

## 1. Profile a separate diagnostic run

```sh
common=(--source-record SOURCE_RECORD --baseline BASELINE
        --initial-snapshot SNAPSHOT --initial-assessment ASSESSMENT
        --slabs 16 --alpha 1e-14 --sequence development
        --verification RUNS/verification/record.json)
uv run --no-sync python -m cProfile -o RUNS/reference.pstats \
  -m deflation_example.coupled_small_study run "${common[@]}" \
  --device cpu --rank 8 --variant standard --capture-trace \
  --budget-seconds 14400 --output RUNS/reference-capture
uv run --no-sync python -m deflation_example.coupled_cost_report profile \
  --profiles RUNS/reference.pstats --output RUNS/profile-summary
```

Run the corresponding rank-zero profile without `--capture-trace`. Instrumented
execution includes profiler and capture overhead and stays outside the timing
comparison. The summary partitions exclusive function time and lists nested
call times separately. Nested times must not be summed. Read only locally
generated, trusted `.pstats` files.

## 2. Replay the same recorded systems

```sh
uv run --no-sync python -m deflation_example.coupled_matched_transfer \
  --optimization RUNS/reference-capture --baseline BASELINE \
  --budget-seconds 14400 --output RUNS/matched-transfer
```

The replay reads the saved initial basis and checks its checksum. Both policies
receive each recorded inactive operator, right-hand side and initial guess in
chronological order. Operator reassembly retains the saved temperature and flow
fields and independently verifies their momentum and continuity residuals.
It does not run an additional momentum update: changing an already converged
flow can alter a late secant-corrected system enough to invalidate its saved
linear residual. The saved solution must still satisfy its original tolerance
under the reconstructed operator. Sequential transfer zero-extends the previous restricted
basis without replacing lost directions. All masks advance its history,
including systems that already satisfy accuracy at their initial guess.
An independently recomputed relative residual of `1e-10` applies to both
policies. This is stricter than some adaptive solves in the captured optimizer;
the record states both thresholds.

The output includes newly active and newly inactive counts, numerical rank,
fallbacks, solver work and coarse energy removal. A separate error equation
uses a `1e-12` residual target. Its cost and the coarse diagnostics are excluded
from the matched solve timers. These measurements describe the captured
initial guesses; they do not describe the errors of another optimization path.
The captured inactive sets belong to the quadratic subproblems, whose bounds
include the trust region as well as the physical temperature limits. Their
transition counts alone do not identify release from a physical temperature bound.

## 3. Compare the GPU implementations

For each of `--device hybrid` and `--device cuda`, run fresh processes with
`--rank 0 --variant standard`, `--rank 8 --variant standard`, and
`--rank 8 --variant recycling`. Use target 7, the same verified inputs,
`--budget-seconds 14400`, and separate output directories. Allocate one GPU
and eight bound CPU cores per run. Include construction, transfer, selection,
verification and cleanup. The hybrid implementation retains host sparse
operations; the resident implementation moves the inner operator to the GPU.
The coupled nonlinear forward evaluations remain part of complete cost.

Measure matched operator actions separately from complete optimization:

```sh
uv run --no-sync python -m deflation_example.coupled_device_profile \
  --trace RUNS/reference-capture/inactive-trace-00 --baseline BASELINE \
  --quadratic 2 --output RUNS/device-profile
```

This checks CPU/GPU action agreement and measures five warmed applications of
the tangent, transpose, inactive operator and frozen preconditioner. The serial
and block-diagonal GPU preconditioner layouts remain separate. These intervals
are diagnostic costs; they are not additive components of sequence time.

The confirmation backend is the fastest full-reference GPU backend for which
all three methods pass the declared accuracy checks. If neither GPU backend
qualifies, retain its failures and use the verified CPU implementation for
confirmation. This development selection does not establish a speedup.

## 4. Confirm on three distinct targets

Freeze the selected backend, source, rank and tolerances before confirmation.
Use `--sequence nearby` for targets 7, 8, 9, and `--repetition 0` through `4`.
For every repetition, run all three methods in fresh processes with identical
initial data and a 21,600-second optimization budget per complete sequence.
Rotate their execution order across repetitions within the allocation. Record
all statuses, including failed or capped sequences, and keep warm-start access
equivalent. Common calibration and process preparation are reported separately.

Pass all fifteen expected output paths, including missing paths, to:

```sh
uv run --no-sync python -m deflation_example.coupled_cost_report confirmation \
  --records RECORD_01 RECORD_02 RECORD_03 RECORD_04 RECORD_05 \
            RECORD_06 RECORD_07 RECORD_08 RECORD_09 RECORD_10 \
            RECORD_11 RECORD_12 RECORD_13 RECORD_14 RECORD_15 \
  --output RUNS/confirmation-summary
```

The report retains each outcome and gives complete-sequence medians and all
observed timing ranges only for complete populations. A headline ratio also
requires matching source, physical settings, initial data, hardware, timing
boundary and repetition. Five timing repetitions still represent three distinct
physical targets. Memory values are sampled process allocations, not enforced
memory budgets. These discrete optimization comparisons do not establish
thermal mesh or time resolution.

Failed-attempt durations remain in `attempt_seconds_by_repetition`. Their
`complete_seconds_by_repetition` entries are null: termination time is not time
to an accurate solution. The plots use elapsed-attempt labels when failures
occur, without reporting a complete-sequence speedup for that population.

## Figures

Generate figures directly from the retained summaries and records:

```sh
uv run --no-sync python -m deflation_example.coupled_followup_figures \
  --transfer RUNS/matched-transfer/record.json \
  --confirmation RUNS/confirmation-summary/summary.json \
  --device-profile RUNS/device-profile/record.json \
  --backend-records HYBRID_BASELINE HYBRID_REFERENCE HYBRID_RECYCLING \
                    CUDA_BASELINE CUDA_REFERENCE CUDA_RECYCLING \
  --output RUNS/figures
```

The figures show both transfer policies, all three methods on each backend,
all five confirmation repetitions, termination statuses and sampled memory.
The recorded input hashes bind each plot to its numerical evidence.

Audit every saved trajectory against the same physical baseline and the
recorded objective without modifying the optimized source:

```sh
uv run --no-sync python -m deflation_example.coupled_confirmation_fields \
  --records RECORD_01 RECORD_02 RECORD_03 RECORD_04 RECORD_05 \
            RECORD_06 RECORD_07 RECORD_08 RECORD_09 RECORD_10 \
            RECORD_11 RECORD_12 RECORD_13 RECORD_14 RECORD_15 \
  --baseline BASELINE --output RUNS/field-audit
```

This reconstructs the weighted objective from saved fields, checks declared
discrete temperature bounds, and compares each method with its matched baseline.
Its comparison thresholds are 0.001 K for temperature and `1e-6` for relative
objective difference. The original residual and KKT requirements remain those
of the optimizer; field agreement does not replace them.
