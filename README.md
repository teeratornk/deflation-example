# Finer-transformer residual-correction evidence, version 1

This data release supports the separate finer-transformer study in
*Reference-Space Reuse for Steady and Transient State-Constrained Conjugate
Heat Transfer*. It preserves original scalar diagnostics, initial correction
runs and all repeated comparisons. The six primary sixteen-target comparisons
retain their original implementation and separate `mesh-cht-data-v1` release.

The transformer has 42,180 spatial state degrees of freedom. A four-slab
trajectory has 168,720. The follow-up preserves the four distinct target
functions, physical bound, operator, reference and final accuracy of the
original finer-mesh attempts. Five complete repetitions per method give
40 sequences and 160 target solves across the two forms and four solvers.
All meet the original relative residual threshold `1e-10` and the five KKT
thresholds `1e-8`. Repetitions measure timing variability on the same targets.

## Diagnosis and comparison

Seven instrumented reference solves terminate with negative `r @ z` and
positive `p @ B @ p`, retained rank 100 and moderate coarse conditioning.
The records include recurrence and independent residuals, their gap, coarse
orthogonality and the scalar test. One steady candidate already satisfies
the independent final criterion before continuation toward the tighter
internal target. The transient example also requires residual correction.

The follow-up verifies each candidate independently before a projected-CG
restart. A rejected candidate defines an error equation with right-hand side
`b - B @ x`. At most four zero-start error solves request relative residual
0.1, with internal target 0.01. Every iteration shares the original cap of
50,000. All four methods receive this policy. Complete time includes all
correction equations, setup, transfers and verification. AmgX resources
persist; matrix-specific solvers and hierarchies are rebuilt. Recycling
updates after locally converged inner solves. Final acceptance always checks
the original inactive equation and complete KKT conditions.

Median complete seconds and ranges over all five repetitions are:

| Form | Jacobi-CG | Reference | Recycling | AmgX |
|---|---:|---:|---:|---:|
| Steady | 39.300 [39.166, 40.001] | 29.251 [29.057, 29.578] | 64.140 [63.960, 64.638] | 23.109 [23.063, 23.172] |
| Four slabs | 65.404 [64.958, 65.741] | 64.986 [64.874, 65.931] | 110.645 [109.706, 114.649] | 93.350 [92.750, 93.869] |

AmgX is fastest for the finer steady case. Reference and Jacobi trajectory
timings overlap. The follow-up establishes successful optimization for these
four-target problems under the corrected stopping procedure. Original attempts
remain available; their failed timings do not enter completed-solve summaries.

## Files and reproduction

- `diagnostic/steady` and `diagnostic/transient` contain instrumented original
  runs, all scalar events and complete sequence records.
- `pilot/steady` and `pilot/transient` contain the initial four-way correction
  comparisons, with one repetition per method.
- `final/steady` and `final/transient` contain every repeated correction run,
  resolved protocol and attempt status. Their protocol phase remains `pilot`
  to distinguish this four-target follow-up from the six primary benchmarks.
- `summary/refinement/summary.json` contains the formula-generated diagnostic
  and comparison summaries with input hashes.
- `manifest.json` records SHA-256 hashes and sizes for all selected evidence.

Numerical source identifiers are
`e56b7fc969079b65bff7d7c0221fcc6abc456460` for instrumentation and
`926b37c40935fbbe7134a3ab5f87ebdfddb68837` for the repeated correction study.
Per-module hashes identify the numerical implementation in every record.
Environment records specify CPU, GPU, numerical backends and native AmgX.
Instrumented timings include diagnostic and snapshot work; use the separate
uninstrumented repeated comparisons for performance.

Use the [v0.6.0 companion](https://github.com/teeratornk/deflation-example/tree/v0.6.0)
and its [mesh guide](https://github.com/teeratornk/deflation-example/blob/v0.6.0/docs/mesh-showcases.md).
The guide supplies uv installation, Hydra configurations, GPU dependencies,
benchmark commands and CPU-only table regeneration. The function
`deflation_example.mesh_evidence.verify` checks this release's manifest.
The script `tools/summarize_mesh_refinement.py` checks unchanged target hashes,
physical and reference modules, shared budgets and final accuracy before
generating the two SI tables.

Large numerical snapshots of rejected inactive systems are regenerated with
`python -m deflation_example.mesh_breakdown` and the diagnostic protocol.
They are excluded from this compact JSON release. The command saves the exact
CSR matrix, right-hand side, initial guess, candidate reference and returned
state for each rejected solve. The original primary data release preserves
earlier implementation and protocol outcomes separately.
