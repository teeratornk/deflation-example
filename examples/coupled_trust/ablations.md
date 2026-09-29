# Ablations for complete coupled optimization

The rank and regularization study uses the protocol in
[small_to_large.md](small_to_large.md). The additional controls below isolate
reference retention, learning from previous solves, and spectral selection.
They use the same nonlinear equations, initial temperature, physical bounds,
preconditioner, warm starts, and final residual and optimality criteria.
The older [construction study](../coupled_optimization/corrected_study/ablations.md)
uses a different numerical procedure and remains separate evidence.
The [four-stage follow-up](four_stage_study.md) supplies matched trace replay,
cost profiling, GPU comparisons and repeated complete-sequence commands.

| Comparison | Settings | Quantity it tests |
| --- | --- | --- |
| Deflation and rank | `--rank 0`, `8`, `16`, `32`; `--variant standard` | Whether iteration savings repay reference construction and application |
| Regularization | `--alpha 1e-14`, `1e-12`, `1e-11`, at matched ranks | Sensitivity to control regularization; each value defines a different objective |
| Information retention | `--variant standard` versus `sequential`, at rank 8 | Direct restriction versus sequential zero extension of the same initial reference |
| Learned correction space | `--variant recycling`, at rank 8 | Retained coarse vectors plus new search directions, with selection and transfer charged |
| Spectral selection | `--variant lowest` versus `standard`, at rank 8 | Lowest Ritz directions versus alternating low/high directions from the same 48-step construction |
| Execution backend | `--device cpu`, `hybrid`, `cuda` | Measured implementation cost; compare methods within each backend first |
| Temporal size | `--slabs 16`, `32`, `64` at the same horizon | Complete cost and memory as the coupled space–time system grows |

The additional policy comparison is declared at alpha `1e-14`, 16 slabs,
development target 7, and rank 8, with rank zero as its common baseline. This
setting was chosen after all four original rank runs met final accuracy. The
original rank-8 run had the lowest complete cost among those references; it
was slower than rank zero. These additional runs are a development comparison,
not an independent confirmation of a previously selected speedup.

## Run the matched policy comparison

Use one clean numerical checkout for all five methods and the equation gate.
The additional options belong to the source recorded by the new runs. Existing
measurements from `a19e66d` remain attached to that source; do not pool them with
new timings. Run on an allocated compute node with the locked environment:

```sh
uv sync --locked --extra study --extra plot
common=(--source-record SOURCE_RECORD --baseline BASELINE
        --initial-snapshot SNAPSHOT --initial-assessment ASSESSMENT
        --slabs 16 --alpha 1e-14 --sequence development --device cpu)
uv run --no-sync python -m deflation_example.coupled_small_study verify \
  "${common[@]}" --output RUNS/verification --budget-seconds 7200
```

After that verification succeeds, run each arm in a separate fresh process:

```sh
uv run --no-sync python -m deflation_example.coupled_small_study run \
  "${common[@]}" --verification RUNS/verification/record.json \
  --rank 0 --variant standard --repetition 0 --budget-seconds 14400 \
  --output RUNS/rank-zero
for variant in standard sequential recycling lowest; do
  if ! uv run --no-sync python -m deflation_example.coupled_small_study run \
    "${common[@]}" --verification RUNS/verification/record.json \
    --rank 8 --variant "$variant" --repetition 0 --budget-seconds 14400 \
    --output "RUNS/$variant"; then
    printf 'Unsuccessful arm retained: %s\n' "$variant"
  fi
done
```

Keep the rank-zero result even if it fails, and run the remaining declared arms.
An unsuccessful baseline cannot supply a speedup denominator. Include all five
paths in the summary, including missing or unsuccessful outputs:

```sh
uv run --no-sync python -m deflation_example.coupled_small_report \
  --records RUNS/rank-zero/record.json RUNS/standard/record.json \
  RUNS/sequential/record.json RUNS/recycling/record.json RUNS/lowest/record.json \
  --plot --output RUNS/summary
```

All reference arms charge the initial full-space construction. Sequential
transfer releases the in-memory full basis after the first restriction. Later
released degrees of freedom receive zero entries; the solver adds no replacement
directions. The initial reference is archived as a numeric array for inspection.
This control requires an uninterrupted run; restoring the full reference after
an interruption would change its information history, so resume is rejected.

Recycling starts without a coarse space. Its candidate pool includes the
deployed coarse vectors and the last `2 * rank` new search directions. With the
velocity-frozen preconditioner, selection uses the implemented preconditioned
Ritz procedure. This differs from the Jacobi-scaled selection used with a
diagonal preconditioner. Equal requested rank does not imply equal deployed
rank or equal memory. Retain candidate storage, numerical ranks, fallbacks, and
selection cost alongside total time.

## Scope and subsequent controls

The sequential and recycling optimizers may follow different nonlinear paths.
Their complete times measure whole algorithms. A transfer-mechanism claim also
needs a replay of identical inactive operators, right-hand sides and initial
guesses. `--capture-trace` saves checksum-bound inner systems for that diagnostic;
keep capture runs separate because they include additional I/O. The existing
[replay tools](../coupled_retention_v8.md) provide equation and warm-start error
diagnostics, but their older reference-bank presets do not constitute a matched
replay of the new rank-8 space. Such a replay remains a separate required step.

The backend, temporal-size and target-sequence controls are staged after verified
development runs. GPU modes require both the `gpu` and `coupled-gpu` extras and an allocated GPU.
Each temporal grid needs its own source-matched equation gate. These refinements
test optimization cost; they do not by themselves establish physical temporal
resolution. Keep nearby targets 7, 8, 9 separate from stress targets 7, 15, 14.

Freeze the comparison settings before five new complete-sequence repetitions.
Use `--repetition 0` through `4` and fresh output directories. The report pairs
only matching repetitions, sources, physical settings, initial states, hardware
metadata and timing boundaries. It does not pool capture and timing runs.
Report all termination statuses, original residuals, final KKT components,
outer and inner iterations, requested and deployed ranks, complete elapsed
time, and sampled process memory. Tests and runnable commands establish software
readiness; only completed verified runs establish numerical results.
