# Detailed supplementary tables

These tables support the prescribed-flow CHT optimization study. They give
per-case accuracy and ranks, cost components, preparation-inclusive times,
sampled memory, and the finer-transformer scalar diagnostics. Every row and
compared method from these table groups is included. The manuscript and SI
retain the principal complete comparisons, transfer tests, verification,
stopping sensitivity, and conditional-bound example.

The [body-fitted reproduction examples](mesh-showcases.md#choose-the-numerical-procedure)
distinguish the primary projected recurrence, measured residual-correction
study, and subsequent guarded solver. The guard is separate from the GPU timings.
The [Cartesian examples](reference-policy-study.md#frozen-complete-optimization-campaign)
provide the refinement and rank comparisons.

## Read or regenerate the tables

Read the tables below directly on GitHub. From a checkout of this revision:

```bash
uv run --locked python tools/supplementary_tables.py
uv run --locked python tools/supplementary_tables.py --write
```

The first command verifies the source checksums, row counts and displayed tables.
The second regenerates this page. Both use the bundled data and Python's standard
library; table verification itself needs no accelerator, optional libraries,
or network. `python3 tools/supplementary_tables.py` also works without installation.
Each section links to commands for regenerating its numerical evidence.

## Table index

- [Body-fitted accuracy](#body-fitted-accuracy)
- [Body-fitted cost components](#body-fitted-cost-components)
- [Body-fitted preparation](#body-fitted-preparation)
- [Body-fitted memory budgets](#body-fitted-memory-budgets)
- [Finer-transformer diagnostics](#finer-transformer-diagnostics)
- [Cartesian deployed ranks](#cartesian-deployed-ranks)
- [Cartesian steady preparation](#cartesian-steady-preparation)
- [Cartesian transient preparation](#cartesian-transient-preparation)
- [Cartesian steady memory](#cartesian-steady-memory)
- [Cartesian transient memory](#cartesian-transient-memory)
- [Cartesian memory budgets](#cartesian-memory-budgets)

## Data and measurement definitions

The [manifest](../examples/supplementary_tables/manifest.json) identifies the
numerical implementations, data versions and exact source-table checksums.
The row files come from manuscript revision
[`c2d10a5`](https://github.com/teeratornk/opt_control/tree/c2d10a54727c5fa9e681d2491ec663c2ba2d0381).
Earlier SI table numbers in the manifest identify that fixed manuscript revision.
They do not change when the current SI is renumbered.

Bore 1 and Bore 2 denote the two bore-in-block mesh levels. Body-fitted
comparisons use 16 targets and five complete repetitions unless the section
specifies the separate four-target diagnostic study. Unmarked body-fitted
reference ranks are 100. Warm starts apply to both PDAS and inner solves.
All completed optimization sequences meet the independent original-residual
and KKT criteria of 1e-10 and 1e-8. Diagnostic breakdowns are identified explicitly.
Repeated timings cover the same target sequence. They do not represent additional
distinct physical problems. Sampled process peaks include all numerical backends.

## Body-fitted accuracy

Maxima over five complete 16-target repetitions. The original linear residual is relative; primal feasibility is absolute and the other KKT components follow the manuscript normalization. Recycling starts with rank zero. Fallbacks count conditioning-safeguard rejections.

[Reproduction example](mesh-showcases.md#rebuild-the-comparison-artifacts).

| Geometry | Form | Method | Original residual | KKT | Deployed rank | Fallbacks |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer | Steady | Jacobi-CG | 1.26e-11 | 9.10e-11 | 0--0 | 0 |
| Transformer | Steady | Reference | 1.12e-11 | 1.16e-10 | 100--100 | 0 |
| Transformer | Steady | Recycling | 1.76e-11 | 1.03e-10 | 0--100 | 0 |
| Transformer | Steady | AmgX | 8.66e-11 | 7.30e-10 | 0--0 | 0 |
| Transformer | 4 slabs | Jacobi-CG | 1.07e-11 | 1.27e-10 | 0--0 | 0 |
| Transformer | 4 slabs | Reference | 1.38e-11 | 1.32e-10 | 100--100 | 0 |
| Transformer | 4 slabs | Recycling | 1.23e-11 | 1.95e-10 | 0--100 | 0 |
| Transformer | 4 slabs | AmgX | 5.90e-11 | 5.09e-10 | 0--0 | 0 |
| Bore 1 | Steady | Jacobi-CG | 9.99e-12 | 7.89e-09 | 0--0 | 0 |
| Bore 1 | Steady | Reference | 9.98e-12 | 7.89e-09 | 100--100 | 0 |
| Bore 1 | Steady | Recycling | 9.99e-12 | 7.89e-09 | 0--100 | 0 |
| Bore 1 | Steady | AmgX | 9.92e-12 | 7.89e-09 | 0--0 | 0 |
| Bore 1 | 4 slabs | Jacobi-CG | 9.97e-12 | 6.82e-11 | 0--0 | 0 |
| Bore 1 | 4 slabs | Reference | 1.00e-11 | 8.82e-11 | 100--100 | 0 |
| Bore 1 | 4 slabs | Recycling | 9.99e-12 | 6.55e-11 | 0--100 | 0 |
| Bore 1 | 4 slabs | AmgX | 9.98e-12 | 6.09e-11 | 0--0 | 0 |
| Bore 2 | Steady | Jacobi-CG | 9.99e-12 | 6.54e-09 | 0--0 | 0 |
| Bore 2 | Steady | Reference | 9.99e-12 | 6.54e-09 | 100--100 | 0 |
| Bore 2 | Steady | Recycling | 1.00e-11 | 6.54e-09 | 0--100 | 0 |
| Bore 2 | Steady | AmgX | 9.98e-12 | 6.54e-09 | 0--0 | 0 |
| Bore 2 | 4 slabs | Jacobi-CG | 1.00e-11 | 8.90e-09 | 0--0 | 0 |
| Bore 2 | 4 slabs | Reference | 1.00e-11 | 8.90e-09 | 100--100 | 0 |
| Bore 2 | 4 slabs | Recycling | 1.00e-11 | 8.90e-09 | 0--100 | 0 |
| Bore 2 | 4 slabs | AmgX | 9.95e-12 | 8.90e-09 | 0--0 | 0 |
| Bore 2, $r=20$ | Steady | Jacobi-CG | 9.99e-12 | 6.54e-09 | 0--0 | 0 |
| Bore 2, $r=20$ | Steady | Reference | 1.00e-11 | 6.54e-09 | 20--20 | 0 |
| Bore 2, $r=20$ | Steady | Recycling | 1.00e-11 | 6.54e-09 | 0--20 | 0 |
| Bore 2, $r=20$ | Steady | AmgX | 9.98e-12 | 6.54e-09 | 0--0 | 0 |
| Bore 2, $r=200$ | Steady | Jacobi-CG | 9.99e-12 | 6.54e-09 | 0--0 | 0 |
| Bore 2, $r=200$ | Steady | Reference | 1.00e-11 | 6.54e-09 | 200--200 | 0 |
| Bore 2, $r=200$ | Steady | Recycling | 1.00e-11 | 6.54e-09 | 0--200 | 0 |
| Bore 2, $r=200$ | Steady | AmgX | 9.98e-12 | 6.54e-09 | 0--0 | 0 |
| Bore 1 | 8 slabs | Jacobi-CG | 1.00e-11 | 6.90e-11 | 0--0 | 0 |
| Bore 1 | 8 slabs | Reference | 1.00e-11 | 7.33e-11 | 100--100 | 0 |
| Bore 1 | 8 slabs | Recycling | 1.00e-11 | 7.25e-11 | 0--100 | 0 |
| Bore 1 | 8 slabs | AmgX | 9.98e-12 | 1.04e-10 | 0--0 | 0 |
| Transformer | 8 slabs | Jacobi-CG | 1.25e-11 | 1.42e-10 | 0--0 | 0 |
| Transformer | 8 slabs | Reference | 1.17e-11 | 1.89e-10 | 100--100 | 0 |
| Transformer | 8 slabs | Recycling | 2.85e-11 | 2.19e-10 | 0--100 | 0 |
| Transformer | 8 slabs | AmgX | 7.88e-11 | 4.09e-10 | 0--0 | 0 |

## Body-fitted cost components

Disjoint components of the actual median-time repetition. Setup includes assembly, reference construction, inactive extraction, basis processing, and coarse or hierarchy setup. Verification includes independent residuals and KKT checks. Rows sum to complete time before rounding.

[Reproduction example](mesh-showcases.md#rebuild-the-comparison-artifacts).

| Geometry | Form | Method | Setup (s) | Iteration (s) | Transfer (s) | Verification (s) | Other (s) | Total (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | Steady | Jacobi-CG | 0.510 | 24.889 | 0.039 | 0.070 | 0.147 | 25.655 |
| Transformer | Steady | Reference | 1.743 | 9.036 | 0.045 | 0.075 | 0.150 | 11.049 |
| Transformer | Steady | Recycling | 2.180 | 36.836 | 0.103 | 0.075 | 0.107 | 39.301 |
| Transformer | Steady | AmgX | 1.014 | 15.694 | 0.066 | 0.079 | 0.272 | 17.124 |
| Transformer | 4 slabs | Jacobi-CG | 1.856 | 36.356 | 0.097 | 0.320 | 0.227 | 38.857 |
| Transformer | 4 slabs | Reference | 3.086 | 18.647 | 0.100 | 0.230 | 0.298 | 22.361 |
| Transformer | 4 slabs | Recycling | 3.593 | 69.312 | 0.101 | 0.271 | 0.227 | 73.504 |
| Transformer | 4 slabs | AmgX | 2.691 | 48.437 | 0.291 | 0.277 | 0.400 | 52.096 |
| Bore 1 | Steady | Jacobi-CG | 1.431 | 4.615 | 0.050 | 0.078 | 0.070 | 6.244 |
| Bore 1 | Steady | Reference | 2.129 | 1.409 | 0.056 | 0.077 | 0.072 | 3.743 |
| Bore 1 | Steady | Recycling | 3.407 | 1.951 | 0.057 | 0.087 | 0.074 | 5.576 |
| Bore 1 | Steady | AmgX | 2.542 | 5.515 | 0.107 | 0.108 | 0.223 | 8.496 |
| Bore 1 | 4 slabs | Jacobi-CG | 4.739 | 6.148 | 0.155 | 0.283 | 0.175 | 11.500 |
| Bore 1 | 4 slabs | Reference | 6.019 | 3.274 | 0.167 | 0.290 | 0.176 | 9.927 |
| Bore 1 | 4 slabs | Recycling | 7.889 | 5.211 | 0.159 | 0.291 | 0.196 | 13.747 |
| Bore 1 | 4 slabs | AmgX | 7.998 | 25.307 | 0.490 | 0.364 | 0.832 | 34.991 |
| Bore 2 | Steady | Jacobi-CG | 16.988 | 38.437 | 0.385 | 0.904 | 0.385 | 57.099 |
| Bore 2 | Steady | Reference | 18.779 | 11.473 | 0.410 | 0.904 | 0.399 | 31.966 |
| Bore 2 | Steady | Recycling | 21.962 | 24.485 | 0.411 | 0.965 | 0.391 | 48.214 |
| Bore 2 | Steady | AmgX | 21.269 | 78.628 | 1.297 | 0.989 | 0.754 | 102.937 |
| Bore 2 | 4 slabs | Jacobi-CG | 89.056 | 55.404 | 2.972 | 5.339 | 1.186 | 153.957 |
| Bore 2 | 4 slabs | Reference | 93.327 | 30.882 | 3.066 | 5.330 | 1.186 | 133.791 |
| Bore 2 | 4 slabs | Recycling | 97.263 | 60.002 | 2.993 | 5.065 | 1.176 | 166.499 |
| Bore 2 | 4 slabs | AmgX | 109.613 | 393.841 | 10.695 | 5.431 | 1.634 | 521.214 |
| Bore 2, $r=20$ | Steady | Jacobi-CG | 17.140 | 38.481 | 0.393 | 0.857 | 0.406 | 57.277 |
| Bore 2, $r=20$ | Steady | Reference | 17.308 | 24.509 | 0.395 | 0.848 | 0.371 | 43.431 |
| Bore 2, $r=20$ | Steady | Recycling | 18.584 | 38.122 | 0.411 | 1.005 | 0.393 | 58.515 |
| Bore 2, $r=20$ | Steady | AmgX | 21.494 | 80.201 | 1.323 | 1.019 | 0.548 | 104.585 |
| Bore 2, $r=200$ | Steady | Jacobi-CG | 17.191 | 36.620 | 0.406 | 0.921 | 0.408 | 55.546 |
| Bore 2, $r=200$ | Steady | Reference | 20.193 | 10.070 | 0.417 | 0.892 | 0.375 | 31.948 |
| Bore 2, $r=200$ | Steady | Recycling | 26.471 | 15.257 | 0.401 | 0.888 | 0.374 | 43.391 |
| Bore 2, $r=200$ | Steady | AmgX | 21.141 | 78.545 | 1.330 | 1.004 | 0.575 | 102.594 |
| Bore 1 | 8 slabs | Jacobi-CG | 10.015 | 6.769 | 0.323 | 0.610 | 0.266 | 17.983 |
| Bore 1 | 8 slabs | Reference | 11.519 | 4.909 | 0.335 | 0.596 | 0.299 | 17.658 |
| Bore 1 | 8 slabs | Recycling | 13.330 | 7.417 | 0.341 | 0.667 | 0.273 | 22.028 |
| Bore 1 | 8 slabs | AmgX | 14.051 | 33.820 | 1.169 | 0.729 | 0.516 | 50.285 |
| Transformer | 8 slabs | Jacobi-CG | 3.999 | 41.818 | 0.257 | 0.525 | 0.405 | 47.004 |
| Transformer | 8 slabs | Reference | 5.494 | 22.631 | 0.204 | 0.501 | 0.447 | 29.276 |
| Transformer | 8 slabs | Recycling | 6.112 | 95.768 | 0.194 | 0.567 | 0.395 | 103.036 |
| Transformer | 8 slabs | AmgX | 4.939 | 75.793 | 0.661 | 0.576 | 0.465 | 82.434 |

## Body-fitted preparation

Medians of five independently timed sequences. Library total adds initialization and finalization to complete time. Process time also includes imports, monitoring, serialization, and field output. Medians of separate intervals need not sum to the median of their sum. These fixed-bound cases require no calibration.

[Reproduction example](mesh-showcases.md#rebuild-the-comparison-artifacts).

| Geometry | Form | Method | Complete (s) | Initialize (s) | Finalize (s) | Library total (s) | Process (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | Steady | Jacobi-CG | 25.655 | 6.369 | 0.001 | 32.026 | 41.473 |
| Transformer | Steady | Reference | 11.049 | 6.221 | 0.000 | 17.405 | 26.109 |
| Transformer | Steady | Recycling | 39.301 | 6.471 | 0.001 | 46.237 | 55.091 |
| Transformer | Steady | AmgX | 17.124 | 6.944 | 0.000 | 24.058 | 33.919 |
| Transformer | 4 slabs | Jacobi-CG | 38.857 | 6.298 | 0.001 | 45.235 | 54.239 |
| Transformer | 4 slabs | Reference | 22.361 | 7.351 | 0.001 | 29.713 | 39.478 |
| Transformer | 4 slabs | Recycling | 73.504 | 6.422 | 0.001 | 79.622 | 94.246 |
| Transformer | 4 slabs | AmgX | 52.096 | 6.109 | 0.001 | 58.342 | 68.226 |
| Bore 1 | Steady | Jacobi-CG | 6.244 | 6.324 | 0.001 | 12.543 | 22.005 |
| Bore 1 | Steady | Reference | 3.743 | 6.157 | 0.001 | 10.078 | 20.697 |
| Bore 1 | Steady | Recycling | 5.576 | 8.540 | 0.001 | 14.027 | 21.962 |
| Bore 1 | Steady | AmgX | 8.496 | 6.232 | 0.001 | 14.886 | 23.858 |
| Bore 1 | 4 slabs | Jacobi-CG | 11.500 | 12.688 | 0.001 | 25.323 | 33.911 |
| Bore 1 | 4 slabs | Reference | 9.927 | 6.304 | 0.000 | 16.150 | 29.919 |
| Bore 1 | 4 slabs | Recycling | 13.747 | 6.519 | 0.001 | 20.130 | 28.656 |
| Bore 1 | 4 slabs | AmgX | 34.991 | 6.071 | 0.001 | 41.865 | 54.399 |
| Bore 2 | Steady | Jacobi-CG | 57.099 | 7.876 | 0.001 | 64.525 | 75.975 |
| Bore 2 | Steady | Reference | 31.966 | 7.283 | 0.001 | 39.322 | 50.148 |
| Bore 2 | Steady | Recycling | 48.214 | 6.381 | 0.001 | 55.183 | 66.757 |
| Bore 2 | Steady | AmgX | 102.937 | 8.469 | 0.001 | 113.374 | 124.696 |
| Bore 2 | 4 slabs | Jacobi-CG | 153.957 | 6.641 | 0.001 | 160.664 | 173.613 |
| Bore 2 | 4 slabs | Reference | 133.791 | 7.550 | 0.001 | 141.403 | 153.884 |
| Bore 2 | 4 slabs | Recycling | 166.499 | 7.764 | 0.001 | 174.817 | 193.275 |
| Bore 2 | 4 slabs | AmgX | 521.214 | 8.012 | 0.001 | 529.558 | 542.542 |
| Bore 2, $r=20$ | Steady | Jacobi-CG | 57.277 | 7.609 | 0.001 | 65.117 | 78.933 |
| Bore 2, $r=20$ | Steady | Reference | 43.431 | 7.986 | 0.001 | 51.388 | 61.465 |
| Bore 2, $r=20$ | Steady | Recycling | 58.515 | 7.515 | 0.001 | 65.723 | 77.234 |
| Bore 2, $r=20$ | Steady | AmgX | 104.585 | 7.671 | 0.001 | 112.290 | 122.399 |
| Bore 2, $r=200$ | Steady | Jacobi-CG | 55.546 | 7.290 | 0.001 | 62.788 | 73.523 |
| Bore 2, $r=200$ | Steady | Reference | 31.948 | 9.722 | 0.001 | 41.611 | 55.586 |
| Bore 2, $r=200$ | Steady | Recycling | 43.391 | 6.418 | 0.001 | 49.821 | 61.259 |
| Bore 2, $r=200$ | Steady | AmgX | 102.594 | 7.334 | 0.001 | 109.993 | 120.744 |
| Bore 1 | 8 slabs | Jacobi-CG | 17.983 | 8.208 | 0.001 | 26.359 | 35.916 |
| Bore 1 | 8 slabs | Reference | 17.658 | 6.423 | 0.001 | 24.095 | 35.632 |
| Bore 1 | 8 slabs | Recycling | 22.028 | 6.732 | 0.001 | 28.787 | 39.339 |
| Bore 1 | 8 slabs | AmgX | 50.285 | 7.730 | 0.001 | 57.733 | 69.621 |
| Transformer | 8 slabs | Jacobi-CG | 47.004 | 8.069 | 0.001 | 54.775 | 70.274 |
| Transformer | 8 slabs | Reference | 29.276 | 7.921 | 0.001 | 36.893 | 51.402 |
| Transformer | 8 slabs | Recycling | 103.036 | 6.580 | 0.001 | 109.617 | 120.938 |
| Transformer | 8 slabs | AmgX | 82.434 | 6.276 | 0.001 | 88.531 | 104.215 |

## Body-fitted memory budgets

Retrospective screens of the larger steady bore-in-block mesh at ranks 20, 100, and 200. Eligible choices satisfy the budget in every repetition; the selected choice has the lowest median complete time. A dash means no measured choice fits. Execution imposed no allocation cap.

[Reproduction example](mesh-showcases.md#rebuild-the-comparison-artifacts).

| GPU budget (GiB) | Method | Selected rank | Complete time (s) | GPU peak (GiB) |
| --- | --- | --- | --- | --- |
| 1.5 | Jacobi-CG | 0 | 57.099 | 1.016 |
| 1.5 | Reference | 100 | 31.966 | 1.385 |
| 1.5 | Recycling | 20 | 58.515 | 1.166 |
| 1.5 | AmgX | --- | --- | --- |
| 2.5 | Jacobi-CG | 0 | 57.099 | 1.016 |
| 2.5 | Reference | 200 | 31.948 | 2.314 |
| 2.5 | Recycling | 20 | 58.515 | 1.166 |
| 2.5 | AmgX | 0 | 102.937 | 2.209 |
| 5.0 | Jacobi-CG | 0 | 57.099 | 1.016 |
| 5.0 | Reference | 200 | 31.948 | 2.314 |
| 5.0 | Recycling | 200 | 43.391 | 4.559 |
| 5.0 | AmgX | 0 | 102.937 | 2.209 |

## Finer-transformer diagnostics

All seven scalar-check breakdowns from the two original four-target reference sequences. Query and PDAS step are one-based. Every basis retains rank 100. Residual is independently recomputed relative to the original inactive equations. Positive curvature and negative projected scalars support the residual-gap diagnosis. These are the original attempts, separate from the corrected timing comparisons.

[Reproduction example](mesh-showcases.md#finer-transformer-residual-correction).

| Form | Query | PDAS step | Inactive degrees of freedom | CG iterations | Coarse condition number | Original residual | r^T z | p^T B p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| steady | 2 | 1 | 41817 | 5010 | 73.84 | 2.03e-10 | -5.20e-25 | 3.62e-22 |
| steady | 3 | 10 | 41729 | 5507 | 73.81 | 2.51e-10 | -1.50e-36 | 3.04e-22 |
| steady | 4 | 11 | 41656 | 5026 | 73.85 | 2.85e-10 | -8.65e-24 | 2.50e-21 |
| transient | 1 | 14 | 168327 | 10001 | 25.55 | 8.08e-10 | -1.83e-22 | 1.60e-21 |
| transient | 2 | 15 | 168332 | 10000 | 25.49 | 3.15e-10 | -6.25e-22 | 3.25e-21 |
| transient | 3 | 14 | 168258 | 8538 | 25.49 | 2.75e-10 | -2.41e-35 | 6.19e-19 |
| transient | 4 | 14 | 168200 | 20217 | 25.49 | 3.04e-10 | -2.84e-21 | 9.66e-21 |

## Cartesian deployed ranks

Every inner solve from five complete repetitions contributes to the deployed-rank range. Recycling begins with empty history. Fallback counts refer to conditioning-safeguard rejections.

[Reproduction example](reference-policy-study.md#regenerate-the-submitted-figures-and-tables).

| Grid and slabs | Start | Method | Requested rank | Deployed range | Fallbacks |
| --- | --- | --- | --- | --- | --- |
| $ 24^3$ | Warm | Reference | 100 | 100--100 | 0 |
| $ 24^3$ | Warm | Recycling | 100 | 0--100 | 0 |
| $ 32^3$ | Warm | Reference | 20 | 20--20 | 0 |
| $ 32^3$ | Warm | Recycling | 20 | 0--20 | 0 |
| $ 32^3$ | Warm | Reference | 100 | 100--100 | 0 |
| $ 32^3$ | Warm | Recycling | 100 | 0--100 | 0 |
| $ 32^3$ | Warm | Reference | 200 | 200--200 | 0 |
| $ 32^3$ | Warm | Recycling | 200 | 0--200 | 0 |
| $ 32^3$ | Cold | Reference | 100 | 100--100 | 0 |
| $ 32^3$ | Cold | Recycling | 100 | 0--100 | 0 |
| $ 48^3$ | Warm | Reference | 100 | 100--100 | 0 |
| $ 48^3$ | Warm | Recycling | 100 | 0--100 | 0 |
| $ 64^3$ | Warm | Reference | 100 | 100--100 | 0 |
| $ 64^3$ | Warm | Recycling | 100 | 0--100 | 0 |
| $ 12^3\times 8$ | Warm | Reference | 100 | 100--100 | 0 |
| $ 12^3\times 8$ | Warm | Recycling | 100 | 0--100 | 0 |
| $ 16^3\times 4$ | Warm | Reference | 100 | 100--100 | 0 |
| $ 16^3\times 4$ | Warm | Recycling | 100 | 0--100 | 0 |
| $ 16^3\times 8$ | Warm | Reference | 100 | 100--100 | 0 |
| $ 16^3\times 8$ | Warm | Recycling | 100 | 0--100 | 0 |
| $ 16^3\times 16$ | Warm | Reference | 100 | 100--100 | 0 |
| $ 16^3\times 16$ | Warm | Recycling | 100 | 0--100 | 0 |
| $ 16^3\times 32$ | Warm | Reference | 100 | 100--100 | 0 |
| $ 16^3\times 32$ | Warm | Recycling | 100 | 0--100 | 0 |
| $ 24^3\times 8$ | Warm | Reference | 20 | 20--20 | 0 |
| $ 24^3\times 8$ | Warm | Recycling | 20 | 0--20 | 0 |
| $ 24^3\times 8$ | Warm | Reference | 100 | 100--100 | 0 |
| $ 24^3\times 8$ | Warm | Recycling | 100 | 0--100 | 0 |
| $ 24^3\times 8$ | Warm | Reference | 200 | 200--200 | 0 |
| $ 24^3\times 8$ | Warm | Recycling | 200 | 0--200 | 0 |
| $ 24^3\times 8$ | Cold | Reference | 100 | 100--100 | 0 |
| $ 24^3\times 8$ | Cold | Recycling | 100 | 0--100 | 0 |
| $ 32^3\times 8$ | Warm | Reference | 100 | 100--100 | 0 |
| $ 32^3\times 8$ | Warm | Recycling | 100 | 0--100 | 0 |

## Cartesian steady preparation

Preparation-inclusive medians add the common steady calibration once to the full process lifetime. The process interval already contains optimization time; complete time is not added again. Each comparison has five 16-target repetitions.

[Reproduction example](reference-policy-study.md#regenerate-the-submitted-figures-and-tables).

| Grid | Start | Method | Rank | Complete (s) | With preparation (s) |
| --- | --- | --- | --- | --- | --- |
| $ 24^3$ | Warm | Jacobi-CG | 0 | 13.217 | 239.323 |
| $ 24^3$ | Warm | Reference | 100 | 6.778 | 234.421 |
| $ 24^3$ | Warm | Recycling | 100 | 10.910 | 237.397 |
| $ 24^3$ | Warm | AmgX | 0 | 9.159 | 235.445 |
| $ 32^3$ | Warm | Reference | 20 | 20.072 | 246.897 |
| $ 32^3$ | Warm | Recycling | 20 | 38.208 | 264.748 |
| $ 32^3$ | Warm | Jacobi-CG | 0 | 28.813 | 254.825 |
| $ 32^3$ | Warm | Reference | 100 | 14.648 | 240.225 |
| $ 32^3$ | Warm | Recycling | 100 | 31.067 | 255.935 |
| $ 32^3$ | Warm | AmgX | 0 | 25.664 | 250.760 |
| $ 32^3$ | Warm | Reference | 200 | 14.280 | 241.135 |
| $ 32^3$ | Warm | Recycling | 200 | 25.856 | 252.516 |
| $ 32^3$ | Cold | Jacobi-CG | 0 | 107.800 | 337.327 |
| $ 32^3$ | Cold | Reference | 100 | 57.141 | 286.324 |
| $ 32^3$ | Cold | Recycling | 100 | 165.841 | 392.468 |
| $ 32^3$ | Cold | AmgX | 0 | 105.109 | 334.125 |
| $ 48^3$ | Warm | Jacobi-CG | 0 | 91.065 | 322.224 |
| $ 48^3$ | Warm | Reference | 100 | 49.548 | 276.464 |
| $ 48^3$ | Warm | Recycling | 100 | 106.266 | 336.827 |
| $ 48^3$ | Warm | AmgX | 0 | 88.669 | 318.529 |
| $ 64^3$ | Warm | Jacobi-CG | 0 | 204.460 | 432.913 |
| $ 64^3$ | Warm | Reference | 100 | 118.370 | 346.603 |
| $ 64^3$ | Warm | Recycling | 100 | 309.930 | 536.868 |
| $ 64^3$ | Warm | AmgX | 0 | 181.939 | 410.237 |

## Cartesian transient preparation

Preparation-inclusive medians add the common transient calibration once to the full process lifetime. Every query optimizes a complete trajectory. The physical horizon and bound remain fixed during refinement.

[Reproduction example](reference-policy-study.md#regenerate-the-submitted-figures-and-tables).

| Grid and slabs | Start | Method | Rank | Complete (s) | With preparation (s) |
| --- | --- | --- | --- | --- | --- |
| $ 12^3\times 8$ | Warm | Jacobi-CG | 0 | 6.021 | 718.614 |
| $ 12^3\times 8$ | Warm | Reference | 100 | 6.460 | 719.477 |
| $ 12^3\times 8$ | Warm | Recycling | 100 | 6.874 | 720.433 |
| $ 12^3\times 8$ | Warm | AmgX | 0 | 12.829 | 725.849 |
| $ 16^3\times 4$ | Warm | Jacobi-CG | 0 | 9.420 | 730.578 |
| $ 16^3\times 4$ | Warm | Reference | 100 | 6.938 | 719.768 |
| $ 16^3\times 4$ | Warm | Recycling | 100 | 9.142 | 723.032 |
| $ 16^3\times 4$ | Warm | AmgX | 0 | 17.482 | 730.732 |
| $ 16^3\times 8$ | Warm | Jacobi-CG | 0 | 11.747 | 724.906 |
| $ 16^3\times 8$ | Warm | Reference | 100 | 13.442 | 726.671 |
| $ 16^3\times 8$ | Warm | Recycling | 100 | 14.407 | 726.871 |
| $ 16^3\times 8$ | Warm | AmgX | 0 | 25.037 | 737.829 |
| $ 16^3\times 16$ | Warm | Jacobi-CG | 0 | 17.012 | 732.199 |
| $ 16^3\times 16$ | Warm | Reference | 100 | 29.041 | 740.362 |
| $ 16^3\times 16$ | Warm | Recycling | 100 | 29.466 | 740.626 |
| $ 16^3\times 16$ | Warm | AmgX | 0 | 61.252 | 774.151 |
| $ 16^3\times 32$ | Warm | Jacobi-CG | 0 | 23.516 | 734.105 |
| $ 16^3\times 32$ | Warm | Reference | 100 | 45.083 | 757.001 |
| $ 16^3\times 32$ | Warm | Recycling | 100 | 41.070 | 752.993 |
| $ 16^3\times 32$ | Warm | AmgX | 0 | 91.566 | 802.598 |
| $ 24^3\times 8$ | Warm | Reference | 20 | 49.658 | 760.401 |
| $ 24^3\times 8$ | Warm | Recycling | 20 | 51.887 | 763.658 |
| $ 24^3\times 8$ | Warm | Jacobi-CG | 0 | 38.106 | 749.852 |
| $ 24^3\times 8$ | Warm | Reference | 100 | 47.788 | 761.409 |
| $ 24^3\times 8$ | Warm | Recycling | 100 | 60.225 | 776.664 |
| $ 24^3\times 8$ | Warm | AmgX | 0 | 91.872 | 805.941 |
| $ 24^3\times 8$ | Warm | Reference | 200 | 56.855 | 767.816 |
| $ 24^3\times 8$ | Warm | Recycling | 200 | 62.346 | 774.063 |
| $ 24^3\times 8$ | Cold | Jacobi-CG | 0 | 144.785 | 860.964 |
| $ 24^3\times 8$ | Cold | Reference | 100 | 193.181 | 918.798 |
| $ 24^3\times 8$ | Cold | Recycling | 100 | 273.143 | 990.031 |
| $ 24^3\times 8$ | Cold | AmgX | 0 | 384.815 | 1099.490 |
| $ 32^3\times 8$ | Warm | Jacobi-CG | 0 | 101.031 | 816.422 |
| $ 32^3\times 8$ | Warm | Reference | 100 | 146.020 | 864.533 |
| $ 32^3\times 8$ | Warm | Recycling | 100 | 158.401 | 873.696 |
| $ 32^3\times 8$ | Warm | AmgX | 0 | 230.574 | 951.571 |

## Cartesian steady memory

Largest sampled full-process allocations across all five repetitions. Both host and GPU values include numerical-library allocation and caches. The final column restores the sampling diagnostics from the original data. These sampled peaks are lower bounds on instantaneous peaks.

[Reproduction example](reference-policy-study.md#regenerate-the-submitted-figures-and-tables).

| Grid | Slabs | Start | Method | Rank | Host peak (GiB) | GPU peak (GiB) | Maximum sampling gap (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $ 24^3$ | 1 | Warm | Jacobi-CG | 0 | 1.304 | 1.000 | 0.131 |
| $ 24^3$ | 1 | Warm | Reference | 100 | 1.335 | 1.023 | 0.138 |
| $ 24^3$ | 1 | Warm | Recycling | 100 | 1.427 | 1.281 | 0.147 |
| $ 24^3$ | 1 | Warm | AmgX | 0 | 1.474 | 1.656 | 0.111 |
| $ 32^3$ | 1 | Warm | Reference | 20 | 1.354 | 1.021 | 0.134 |
| $ 32^3$ | 1 | Warm | Recycling | 20 | 1.466 | 1.066 | 0.148 |
| $ 32^3$ | 1 | Warm | Jacobi-CG | 0 | 1.337 | 1.002 | 0.129 |
| $ 32^3$ | 1 | Warm | Reference | 100 | 1.411 | 1.166 | 0.176 |
| $ 32^3$ | 1 | Warm | Recycling | 100 | 1.544 | 2.105 | 0.145 |
| $ 32^3$ | 1 | Warm | AmgX | 0 | 1.500 | 1.656 | 0.148 |
| $ 32^3$ | 1 | Warm | Reference | 200 | 1.467 | 1.941 | 0.144 |
| $ 32^3$ | 1 | Warm | Recycling | 200 | 1.633 | 4.010 | 0.144 |
| $ 32^3$ | 1 | Cold | Jacobi-CG | 0 | 1.372 | 1.002 | 0.145 |
| $ 32^3$ | 1 | Cold | Reference | 100 | 1.441 | 1.166 | 0.146 |
| $ 32^3$ | 1 | Cold | Recycling | 100 | 1.565 | 2.105 | 0.145 |
| $ 32^3$ | 1 | Cold | AmgX | 0 | 1.484 | 1.656 | 0.142 |
| $ 48^3$ | 1 | Warm | Jacobi-CG | 0 | 1.471 | 1.021 | 0.132 |
| $ 48^3$ | 1 | Warm | Reference | 100 | 1.664 | 3.766 | 0.144 |
| $ 48^3$ | 1 | Warm | Recycling | 100 | 1.851 | 4.920 | 0.142 |
| $ 48^3$ | 1 | Warm | AmgX | 0 | 1.632 | 1.867 | 0.142 |
| $ 64^3$ | 1 | Warm | Jacobi-CG | 0 | 1.716 | 1.344 | 0.148 |
| $ 64^3$ | 1 | Warm | Reference | 100 | 2.078 | 11.535 | 0.153 |
| $ 64^3$ | 1 | Warm | Recycling | 100 | 2.354 | 53.309 | 0.147 |
| $ 64^3$ | 1 | Warm | AmgX | 0 | 1.893 | 2.594 | 0.146 |

## Cartesian transient memory

The same process-level measurement covers the coupled space-time operator, compact reference factors, deployed bases, cached products, and recycling candidates. Each row includes all five repetitions, including rank and cold-start controls.

[Reproduction example](reference-policy-study.md#regenerate-the-submitted-figures-and-tables).

| Grid | Slabs | Start | Method | Rank | Host peak (GiB) | GPU peak (GiB) | Maximum sampling gap (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $ 12^3$ | 8 | Warm | Jacobi-CG | 0 | 1.306 | 1.000 | 0.144 |
| $ 12^3$ | 8 | Warm | Reference | 100 | 1.324 | 1.023 | 0.129 |
| $ 12^3$ | 8 | Warm | Recycling | 100 | 1.449 | 1.254 | 0.138 |
| $ 12^3$ | 8 | Warm | AmgX | 0 | 1.427 | 1.795 | 0.130 |
| $ 16^3$ | 4 | Warm | Jacobi-CG | 0 | 1.317 | 1.000 | 0.159 |
| $ 16^3$ | 4 | Warm | Reference | 100 | 1.339 | 1.047 | 0.128 |
| $ 16^3$ | 4 | Warm | Recycling | 100 | 1.451 | 1.414 | 0.146 |
| $ 16^3$ | 4 | Warm | AmgX | 0 | 1.432 | 1.795 | 0.132 |
| $ 16^3$ | 8 | Warm | Jacobi-CG | 0 | 1.342 | 1.002 | 0.132 |
| $ 16^3$ | 8 | Warm | Reference | 100 | 1.391 | 1.154 | 0.149 |
| $ 16^3$ | 8 | Warm | Recycling | 100 | 1.561 | 2.090 | 0.125 |
| $ 16^3$ | 8 | Warm | AmgX | 0 | 1.469 | 1.795 | 0.143 |
| $ 16^3$ | 16 | Warm | Jacobi-CG | 0 | 1.408 | 1.018 | 0.140 |
| $ 16^3$ | 16 | Warm | Reference | 100 | 1.473 | 1.814 | 0.141 |
| $ 16^3$ | 16 | Warm | Recycling | 100 | 1.656 | 3.746 | 0.176 |
| $ 16^3$ | 16 | Warm | AmgX | 0 | 1.540 | 2.209 | 0.157 |
| $ 16^3$ | 32 | Warm | Jacobi-CG | 0 | 1.542 | 1.148 | 0.142 |
| $ 16^3$ | 32 | Warm | Reference | 100 | 1.657 | 3.068 | 0.145 |
| $ 16^3$ | 32 | Warm | Recycling | 100 | 1.917 | 8.074 | 0.142 |
| $ 16^3$ | 32 | Warm | AmgX | 0 | 1.719 | 2.900 | 0.145 |
| $ 24^3$ | 8 | Warm | Reference | 20 | 1.523 | 1.154 | 0.176 |
| $ 24^3$ | 8 | Warm | Recycling | 20 | 1.683 | 1.963 | 0.158 |
| $ 24^3$ | 8 | Warm | Jacobi-CG | 0 | 1.507 | 1.094 | 0.134 |
| $ 24^3$ | 8 | Warm | Reference | 100 | 1.620 | 3.129 | 0.148 |
| $ 24^3$ | 8 | Warm | Recycling | 100 | 1.862 | 6.988 | 0.144 |
| $ 24^3$ | 8 | Warm | AmgX | 0 | 1.646 | 2.969 | 0.141 |
| $ 24^3$ | 8 | Warm | Reference | 200 | 1.712 | 6.750 | 0.148 |
| $ 24^3$ | 8 | Warm | Recycling | 200 | 2.055 | 29.848 | 0.142 |
| $ 24^3$ | 8 | Cold | Jacobi-CG | 0 | 1.535 | 1.094 | 0.142 |
| $ 24^3$ | 8 | Cold | Reference | 100 | 1.640 | 3.129 | 0.174 |
| $ 24^3$ | 8 | Cold | Recycling | 100 | 1.892 | 6.988 | 0.151 |
| $ 24^3$ | 8 | Cold | AmgX | 0 | 1.714 | 2.969 | 0.124 |
| $ 32^3$ | 8 | Warm | Jacobi-CG | 0 | 1.837 | 1.598 | 0.148 |
| $ 32^3$ | 8 | Warm | Reference | 100 | 1.946 | 9.518 | 0.146 |
| $ 32^3$ | 8 | Warm | Recycling | 100 | 2.396 | 44.832 | 0.190 |
| $ 32^3$ | 8 | Warm | AmgX | 0 | 1.978 | 4.760 | 0.146 |

## Cartesian memory budgets

Retrospective rank choices at steady 32^3 and transient 24^3 with eight slabs. The host budget is 16 GiB. Choices must meet the residual and KKT criteria and fit the sampled-memory budget in all repetitions. A dash means no measured rank qualifies; no allocator cap was enforced.

[Reproduction example](reference-policy-study.md#regenerate-the-submitted-figures-and-tables).

| Problem | GPU budget (GiB) | Method | Selected rank | Complete time (s) |
| --- | --- | --- | --- | --- |
| Steady | 8 | Reference | 200 | 14.280 |
| Steady | 16 | Reference | 200 | 14.280 |
| Steady | 64 | Reference | 200 | 14.280 |
| Steady | 8 | Recycling | 200 | 25.856 |
| Steady | 16 | Recycling | 200 | 25.856 |
| Steady | 64 | Recycling | 200 | 25.856 |
| Steady | 8 | Jacobi-CG | 0 | 28.813 |
| Steady | 16 | Jacobi-CG | 0 | 28.813 |
| Steady | 64 | Jacobi-CG | 0 | 28.813 |
| Steady | 8 | AmgX | 0 | 25.664 |
| Steady | 16 | AmgX | 0 | 25.664 |
| Steady | 64 | AmgX | 0 | 25.664 |
| Transient | 8 | Reference | 100 | 47.788 |
| Transient | 16 | Reference | 100 | 47.788 |
| Transient | 64 | Reference | 100 | 47.788 |
| Transient | 8 | Recycling | 20 | 51.887 |
| Transient | 16 | Recycling | 20 | 51.887 |
| Transient | 64 | Recycling | 20 | 51.887 |
| Transient | 8 | Jacobi-CG | 0 | 38.106 |
| Transient | 16 | Jacobi-CG | 0 | 38.106 |
| Transient | 64 | Jacobi-CG | 0 | 38.106 |
| Transient | 8 | AmgX | 0 | 91.872 |
| Transient | 16 | AmgX | 0 | 91.872 |
| Transient | 64 | AmgX | 0 | 91.872 |
