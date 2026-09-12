# Guarded residual correction: complete GPU comparisons

These separate measurements use the recommended initial-residual guard on the finer transformer.
Each method solves four targets per sequence, with five complete repetitions per form.
The published primary and correction-study timings remain unchanged.

| Form | Method | Completed / attempted | Median time [full range] (s) | Maximum original residual | Maximum KKT component |
| --- | --- | --- | --- | --- | --- |
| Steady | jacobi | 5 / 5 | 40.277 [39.790, 41.098] | 9.475e-11 | 1.862e-09 |
| Steady | reference | 5 / 5 | 30.015 [29.575, 30.610] | 9.561e-11 | 8.016e-09 |
| Steady | recycling | 5 / 5 | 66.003 [65.066, 67.891] | 9.090e-11 | 2.400e-09 |
| Steady | amgx | 5 / 5 | 23.613 [23.543, 23.631] | 7.057e-11 | 1.409e-09 |
| Four slabs | jacobi | 5 / 5 | 64.075 [63.828, 65.725] | 4.823e-11 | 4.567e-10 |
| Four slabs | reference | 5 / 5 | 65.225 [64.752, 66.328] | 9.128e-11 | 5.131e-09 |
| Four slabs | recycling | 5 / 5 | 109.717 [108.541, 113.341] | 9.609e-11 | 5.608e-10 |
| Four slabs | amgx | 5 / 5 | 91.730 [91.677, 91.784] | 9.706e-11 | 2.382e-09 |

Times include reference construction, all active-set steps, transfers, verification and solver cleanup.
Library preparation and full process times remain separately identified in the records.
KKT denotes the Karush–Kuhn–Tucker conditions; primal feasibility is absolute, and the other components follow the normalized definitions in the protocol.
All methods use the same final residual and KKT criteria. The records retain each iteration, correction, rank and sampled process-memory measurement.

The initial-residual guard is a numerical safeguard; these timings do not isolate its causal effect from changes in hardware load between studies.
AmgX has the lowest steady median; Jacobi-CG has the lowest transient median, with overlapping reference/Jacobi timing ranges.

## Reproduce

The [GPU commands](../temporal_resolution/guarded_gpu.md) pin the measured numerical source and configurations.
To regenerate the tables and figures from these raw records, use a checkout containing this archive:

```bash
uv sync --frozen --extra study --extra plot
uv run --frozen python -m deflation_example.mesh_report \
  --studies examples/guarded_refinement/records/steady examples/guarded_refinement/records/transient \
  --output runs/guarded-refinement-summary
```

The SHA-256 manifest covers the complete comparison records. No temperature or control field is needed to regenerate the timing and accuracy summaries; the numerical reproduction commands regenerate those fields.
