# Complete manuscript tables

This archive preserves all six article tables and nineteen supporting-information
tables from the 22-page article and 21-page SI at manuscript revision
`ce40ee5b133905a55eb9511ee9bfb2d286a5c7e6`. The shorter article and SI retain the
scientific argument and selected comparisons. Original table numbers below remain
fixed so a relocated table has a stable destination after manuscript renumbering.
Captions, units, failed outcomes, ranges and numerical precision are preserved.
Section, equation and algorithm numbers in these captions refer to that frozen version.

## Read and reproduce

The tables below can be read directly on GitHub. The
[manifest](../examples/manuscript_tables/manifest.json) maps every original table
label to its source files, original manuscript paths, SHA-256 checksums and row count.
Each table links to the numerical reproduction guide for its measurements.

To verify the complete source inventory and regenerate this reader page from the
archived table rows, run from a checkout of this repository revision:

```bash
python3 tools/manuscript_tables.py
python3 tools/manuscript_tables.py --write
```

These commands use only the Python standard library and need no GPU or network.
They check and render the preserved table rows; they do not rerun the thermal solves.
To recompute the numerical rows from recorded measurements, follow the
[v21 evidence export](prescribed-speedup-v21.md#reproduce-the-tables), the
[Cartesian reproduction guide](reference-policy-study.md#frozen-complete-optimization-campaign),
and the linked temperature-bound and residual-correction examples. Those guides
identify the measured source revisions and data archives. The v21 export checks
the raw-record manifest before reconstructing the tables. New thermal timings
require the hardware and settings specified in the corresponding guide.

## Measurement and interpretation

The primary comparison uses one NVIDIA H200 GPU and four CPU threads, double
precision, 16 distinct targets and five complete sequence repetitions per method.
Complete time includes operator and reference construction, all active-set updates
and inner solves, transfers, independent verification and cleanup. Process startup,
common warmup and calibration have their separately stated boundaries. Reference
construction is charged once per sequence. Timing envelopes combine observed
extrema; they are not confidence intervals. Single-repetition and separate-job
comparisons are identified in their captions.

The original inactive-system residual threshold is 1e-10 and the discrete KKT
threshold is 1e-8, except the explicitly identified tolerance variations. State
bounds hold at the optimization time levels; Table S7 records violations on finer
forward replays. Sampled GPU allocations include allocator caches and all numerical
backends and are lower bounds on process peaks. Tables S18 and S19 use the earlier
advective transformer transport and have a different interpretation from the
current skew-symmetric comparisons.

The original [measurement definitions](../examples/manuscript_tables/sources/cht_si_methods.tex),
[timing procedures](../examples/manuscript_tables/sources/cht_protocol.tex), and
[earlier residual-correction description and Algorithm S1](../examples/manuscript_tables/sources/cht_refinement_diagnosis.tex)
are preserved as LaTeX source. The
[measured and guarded numerical procedures](mesh-showcases.md#choose-the-numerical-procedure)
give their executable reproduction routes. Earlier relocated tables remain in
[the preceding repository supplement](supplementary-tables.md).

## Table index

| Original table | Subject |
| --- | --- |
| [1](#table-1) | Timing procedures and reference settings |
| [2](#table-2) | Primary complete optimization costs |
| [3](#table-3) | Matched Cartesian transfer replays |
| [4](#table-4) | Steady scale comparisons |
| [5](#table-5) | Trajectory comparisons |
| [6](#table-6) | Operating sequences |
| [S1](#table-s1) | Transport energy stability |
| [S2](#table-s2) | Comparison inventory |
| [S3](#table-s3) | AmgX configuration |
| [S4](#table-s4) | Timing boundaries |
| [S5](#table-s5) | Independent body-fitted verification |
| [S6](#table-s6) | Independent Cartesian verification |
| [S7](#table-s7) | Time refinement and replay |
| [S8](#table-s8) | Complete timing ranges and GPU allocation |
| [S9](#table-s9) | Cartesian residual and KKT maxima |
| [S10](#table-s10) | Matched body-fitted transfer replays |
| [S11](#table-s11) | Deflated spectra of small replays |
| [S12](#table-s12) | Complete-sequence retention ablation |
| [S13](#table-s13) | Conditional conditioning certificates |
| [S14](#table-s14) | Supplementary sensitivity comparisons |
| [S15](#table-s15) | Single-repetition and solver-limit comparisons |
| [S16](#table-s16) | CPU direct and block-in-time comparisons |
| [S17](#table-s17) | Optimized fields under different bounds |
| [S18](#table-s18) | Earlier advective transformer comparisons |
| [S19](#table-s19) | Earlier residual-correction comparisons |

<a id="table-1"></a>

## Table 1: Timing procedures and reference settings

Timing procedures and their prescribed reference settings. Both procedures solve each inactive system with Algorithm 1, including its residual replacement, and restrict the reference on the GPU. Complete temporal coverage keeps every temporal factor of each retained spatial mode; truncated coverage keeps the lowest spatial–temporal pairs up to the total rank. Temporal coverage does not apply to steady cases.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-1.tex).

| Procedure | Comparisons | Inactive Hessian | Reference mesh and rank | Temporal coverage |
| --- | --- | --- | --- | --- |
| Per-update assembly | Table 2 | Sparse products at every update | Packaged mesh, total rank 100 | Truncated with four slabs; the steady cases have no temporal factors |
| Sliced Hessian | Table 4 (steady) | Principal submatrix of the preassembled Hessian | Closed form on Cartesian grids, rank 100; packaged mesh for the transformer and Bore 1, rank 100; first refinement for Bore 2 and Bore 3 and second refinement for Bore 4, rank 200 | No temporal factors |
| Sliced Hessian | Table 5 (trajectories) | Principal submatrix of the preassembled Hessian | Closed form on Cartesian grids; packaged mesh for the transformer, Bore 1 and Bore 2, total rank 100; first refinement for Bore 3, rank 200 | Complete on Cartesian grids ($S=12$ at eight slabs, $S=6$ at 16 and 32) and Bore 3 ($S=50$); truncated elsewhere |
| Sliced Hessian | Table 6 (steady and trajectory sequences) | Principal submatrix of the preassembled Hessian | Packaged mesh, total rank 100; first refinement for Bore 3, rank 200 | Truncated for the transformer day, overloads, drive cycle and four-slab shuffled sequences. The steady sequences have no temporal factors |

<a id="table-2"></a>

## Table 2: Primary complete optimization costs

Median complete-sequence times in seconds over five repetitions of 16-target optimization sequences; every method converges at every target in every repetition. The speedup is the ratio of the median times of the fastest alternative and RefDef; brackets give the envelope of ratios between extreme repetition times. Dimension counts the optimized spatial or space–time DOFs. RefDef and recycling use total rank 100, and the transformer uses the skew-symmetric transport 10.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-2.tex).

| Geometry | Form | Dimension | Jacobi-CG | RefDef | Recycling | AmgX | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | Steady | 10830 | 24.961 | 10.532 | 35.666 | 17.535 | 1.66 [1.62, 1.71] |
| Transformer | 4 slabs | 43320 | 38.564 | 22.543 | 73.053 | 51.116 | 1.71 [1.70, 1.72] |
| Bore 1 | Steady | 3958 | 6.358 | 3.805 | 5.698 | 8.740 | 1.50 [1.45, 1.51] |
| Bore 1 | 4 slabs | 15832 | 11.752 | 10.113 | 13.942 | 34.289 | 1.16 [1.13, 1.19] |
| Bore 2 | Steady | 36197 | 57.602 | 32.481 | 48.158 | 102.570 | 1.48 [1.42, 1.52] |
| Bore 2 | 4 slabs | 144788 | 162.978 | 142.336 | 176.113 | 527.429 | 1.15 [1.09, 1.28] |

<a id="table-3"></a>

## Table 3: Matched Cartesian transfer replays

Matched replays of direct restriction and zero-extension transfer of the full-domain reference on the same systems with zero initial guesses. Time and iterations sum per-kernel medians over three repetitions. Solves counts the replayed inactive systems, Rank gives the smallest and largest retained rank over these solves, and Energy fraction is the median of Equation 24.

[Numerical reproduction guide](reference-policy-study.md#frozen-complete-optimization-campaign). [Original table source](../examples/manuscript_tables/sources/table-3.tex).

| Grid and slabs | Solves | Transfer | Rank | Energy fraction | Iterations | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| $ 6^3$ | 32 | Direct restriction | 20–20 | 0.975 | 1002 | 0.048 |
| $ 6^3$ | 32 | Zero-extension transfer | 20–20 | 0.058 | 1625 | 0.072 |
| $ 4^3\times 4$ | 35 | Direct restriction | 20–20 | 0.953 | 1354 | 0.066 |
| $ 4^3\times 4$ | 35 | Zero-extension transfer | 20–20 | 0.057 | 1930 | 0.087 |
| $ 24^3$ | 86 | Direct restriction | 100–100 | 0.651 | 19400 | 6.034 |
| $ 24^3$ | 86 | Zero-extension transfer | 93–100 | 0.035 | 81533 | 25.404 |
| $ 12^3\times 8$ | 65 | Direct restriction | 100–100 | 0.403 | 17521 | 5.748 |
| $ 12^3\times 8$ | 65 | Zero-extension transfer | 100–100 | 0.025 | 35998 | 13.268 |
| $ 16^3\times 4$ | 74 | Direct restriction | 100–100 | 0.314 | 16749 | 6.166 |
| $ 16^3\times 4$ | 74 | Zero-extension transfer | 100–100 | 0.016 | 57025 | 19.127 |

<a id="table-4"></a>

## Table 4: Steady scale comparisons

Steady comparisons under Algorithm 2 with 16 targets. Times are median complete-sequence seconds over the repetitions, out of five, that converge at every target. The speedup is the ratio of the median times of the fastest alternative and RefDef, and brackets give the envelope of ratios between extreme repetition times; the iteration ratio divides the median inner iterations of Jacobi-CG by those of RefDef, and a dash marks a ratio without a converged Jacobi-CG repetition. The last column lists each method with a repetition that fails to converge, with its converged and attempted repetitions. The standard AmgX stopping factor is $10^{-1}$; AmgX ($10^{-2}$) and AmgX ($10^{-3}$) set it to $10^{-2}$ and $10^{-3}$. The $96^3$ and Bore 4 envelopes combine measurements from separate jobs. $^\dagger$Compared with a subset of the alternatives: Bore 4 with Jacobi-CG and AmgX, where PDAS may take up to 400 steps.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-4.tex).

| Problem | Size | Unknowns | RefDef | Fastest alternative | Speedup [range] | Iteration ratio | Not converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cartesian | $32^3$ | 32,768 | 13.5 | AmgX 25.2 | 1.87 [1.85, 1.91] | 4.34 | – |
| Cartesian | $48^3$ | 110,592 | 40.7 | AmgX 88.5 | 2.17 [2.12, 2.37] | 4.39 | – |
| Cartesian | $64^3$ | 262,144 | 100.4 | AmgX 177.5 | 1.77 [1.70, 1.83] | 4.42 | – |
| Cartesian | $96^3$ | 884,736 | 514.1 | AmgX ($10^{-2}$) 715.1 | 1.39 [1.37, 1.40] | 4.56 | AmgX 0/5 |
| Cartesian | $128^3$ | 2,097,152 | 1885.0 | Jacobi-CG 3644.8 | 1.93 [1.65, 2.67] | 6.32 | AmgX ($10^{-2}$) 0/5, AmgX ($10^{-3}$) 0/5, AmgX 0/5 |
| Transformer | packaged mesh | 10,830 | 10.6 | AmgX 17.4 | 1.65 [1.56, 1.70] | 4.19 | – |
| Bore-in-block | Bore 1 | 3,958 | 4.2 | recycling 6.0 | 1.42 [1.26, 1.59] | 4.95 | – |
| Bore-in-block | Bore 2 | 36,197 | 30.0 | recycling 42.3 | 1.41 [1.31, 1.56] | 7.55 | – |
| Bore-in-block | Bore 3 | 308,171 | 330.8 | recycling 560.6 | 1.69 [1.39, 1.94] | 7.03 | – |
| Bore-in-block$^\dagger$ | Bore 4 | 2,540,695 | 7065.2 | AmgX 9851.0 | 1.39 [1.36, 1.41] | – | Jacobi-CG 0/5 |

<a id="table-5"></a>

## Table 5: Trajectory comparisons

Trajectory comparisons optimize all slabs of the horizon at once. Columns follow Table 4, and unknowns count all slabs. The Cartesian references keep $S=12$ spatial modes, each with a temporal factor for every slab, at eight slabs and $S=6$ at 16 and 32 slabs, and the transformer uses the skew-symmetric transport. $^\dagger$Compared with Jacobi-CG alone.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-5.tex).

| Problem | Slabs | Unknowns | RefDef | Fastest alternative | Speedup [range] | Iteration ratio | Not converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cartesian | $12^3$, 8 slabs | 13,824 | 3.9 | Jacobi-CG 5.5 | 1.43 [1.39, 1.45] | 3.12 | – |
| Cartesian | $16^3$, 8 slabs | 32,768 | 7.1 | Jacobi-CG 11.1 | 1.56 [1.51, 1.65] | 3.16 | – |
| Cartesian | $24^3$, 8 slabs | 110,592 | 22.1 | Jacobi-CG 35.7 | 1.61 [1.59, 1.65] | 3.10 | – |
| Cartesian | $32^3$, 8 slabs | 262,144 | 65.9 | Jacobi-CG 95.4 | 1.45 [1.42, 1.47] | 3.08 | – |
| Cartesian | $48^3$, 8 slabs | 884,736 | 324.0 | Jacobi-CG 388.8 | 1.20 [1.19, 1.22] | 2.99 | – |
| Cartesian | $64^3$, 8 slabs | 2,097,152 | 1145.9 | Jacobi-CG 1279.7 | 1.12 [1.09, 1.12] | 3.03 | AmgX 0/5 |
| Cartesian | $16^3$, 16 slabs | 65,536 | 12.2 | Jacobi-CG 15.7 | 1.29 [1.22, 1.30] | 2.36 | – |
| Cartesian | $16^3$, 32 slabs | 131,072 | 20.6 | Jacobi-CG 22.3 | 1.08 [1.07, 1.12] | 2.52 | – |
| Transformer | 4 slabs | 43,320 | 21.6 | Jacobi-CG 37.9 | 1.75 [1.69, 1.76] | 2.91 | – |
| Transformer | 8 slabs | 86,640 | 27.4 | Jacobi-CG 44.3 | 1.62 [1.59, 1.64] | 2.69 | – |
| Bore 1 | 4 slabs | 15,832 | 6.9 | Jacobi-CG 8.8 | 1.27 [1.17, 1.33] | 2.75 | – |
| Bore 1 | 8 slabs | 31,664 | 10.2 | Jacobi-CG 11.0 | 1.07 [0.99, 1.65] | 1.97 | – |
| Bore 2 | 4 slabs | 144,788 | 72.0 | Jacobi-CG 91.2 | 1.27 [1.25, 1.30] | 2.72 | – |
| Bore 2 | 8 slabs | 289,576 | 140.3 | Jacobi-CG 143.5 | 1.02 [1.01, 1.05] | 1.96 | – |
| Bore 3$^\dagger$ | 4 slabs | 1,232,684 | 1474.6 | Jacobi-CG 1644.8 | 1.12 [1.10, 1.14] | 3.90 | – |

<a id="table-6"></a>

## Table 6: Operating sequences

Operating sequences change loads, flow rates, windows or query order; one reference built at nominal conditions is used for every query. Columns follow Table 4, and the transformer uses the skew-symmetric transport in every sequence.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-6.tex).

| Scenario | Problem | Unknowns | RefDef | Fastest alternative | Speedup [range] | Iteration ratio | Not converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer day | 24 hourly windows | 86,640 | 53.1 | Jacobi-CG 77.1 | 1.45 [1.16, 1.74] | 2.42 | – |
| Drive cycle | 16 windows | 144,788 | 85.5 | Jacobi-CG 107.5 | 1.26 [1.23, 1.28] | 2.74 | – |
| Off-design flow | transformer | 10,830 | no method converges at every query |  |  |  |  |
| Off-design flow | Bore 2 | 36,197 | 43.6 | recycling 60.8 | 1.39 [1.35, 1.41] | 4.86 | – |
| Off-design flow | Bore 3 | 308,171 | 529.0 | recycling 763.8 | 1.44 [1.41, 1.48] | 7.21 | – |
| Overloads | transformer, 8 slabs | 86,640 | 31.9 | Jacobi-CG 46.0 | 1.44 [1.20, 1.56] | 2.45 | – |
| Shuffled order | transformer | 10,830 | 15.6 | AmgX 17.5 | 1.12 [1.07, 1.16] | 2.72 | – |
| Shuffled order | Bore 2 | 36,197 | 27.8 | recycling 42.7 | 1.54 [1.52, 1.59] | 4.86 | – |
| Shuffled order | Bore 2, 4 slabs | 144,788 | 88.3 | Jacobi-CG 116.1 | 1.32 [1.29, 1.63] | 2.73 | – |

<a id="table-s1"></a>

## Table S1: Transport energy stability

Inertia of $\tfrac12(K_{FF}+K_{FF}^\top)$, the symmetric part of the assembled free-node matrix. Counts give negative, zero and positive eigenvalues. The transport-negative column counts the negative eigenvalues of the symmetric part $S$ of the free-node transport matrix alone, after adding $\sigma(C_\ell)_{FF}$ with $\sigma=10^{-9}\max\|S\|/\max(C_\ell)_{FF}$, which separates a semidefinite part from an indefinite one. The last column states whether the symmetric part of $K_{FF}$ is positive definite, which rules out growing modes; Table S7 shows the growth of the original transformer transport.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s1.tex).

| Model | Transport | Free nodes | Operator | Transport negative | Positive definite |
| --- | --- | --- | --- | --- | --- |
| Transformer | advective | 10,830 | 11/0/10819 | 4,753 | no |
| Transformer | skew | 10,830 | 0/0/10830 | 0 | yes |
| Transformer, refined once | skew | 42,180 | 0/0/42180 | 0 | yes |
| Transformer, refined twice | skew | 166,440 | 0/0/166440 | 0 | yes |
| Bore 1 | advective | 3,958 | 0/0/3958 | 660 | yes |
| Bore 2 | advective | 36,197 | 0/0/36197 | 4,442 | yes |
| Bore 3 | advective | 308,171 | 0/0/308171 | 32,505 | yes |

<a id="table-s2"></a>

## Table S2: Comparison inventory

Comparisons, problems and measurements. Every timed comparison uses 16 targets and five complete sequence repetitions per method unless stated otherwise. Ranks denote total spatial or space–time rank, and $S$ the number of spatial modes, each paired with a temporal factor for every slab.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s2.tex).

| Purpose | Problems and reference | Measurements |
| --- | --- | --- |
| Optimization verification | 12 body-fitted and nine Cartesian steady or transient verification problems | Independent bounded least squares, original inner residuals, KKT components, state and objective agreement, trajectory recovery |
| Body-fitted optimization | Axisymmetric transformer and three-dimensional bore-in-block meshes; coarse P1 references | Complete steady and transient costs, KKT and original-residual checks, constraint changes, rank effects, and process memory |
| Steady scale | Cartesian $32^3$–$128^3$; transformer; Bore 1–4 up to 2.5 million DOFs; ranks 100 and 200 | Complete time and inner iterations under main Algorithm 2, original residuals and KKT checks |
| Trajectories | Cartesian $12^3$–$64^3$ with 8–32 slabs, $S=12$ or 6; transformer and Bore 1–3 with 4 or 8 slabs | Complete trajectory optimization, original residuals, KKT and recovery checks |
| Operating conditions | Transformer day, drive cycle, off-design flow, overloads and query order; one nominal reference | Complete time over chained windows or reassembled operators; original residuals and KKT checks |
| Retention ablation | Six primary body-fitted sequences; the same reference by direct restriction or zero-extension transfer, with Jacobi-CG in the same job | Complete time and inner iterations |
| Transfer mechanism | Steady and transient PDAS traces; identical replay systems | Activation and release, retained rank, coarse energy removal, deflated spectrum on small systems, original-residual acceptance |
| Conditional analysis | Controlled $6^3$ CHT restrictions; exact rank-one scaled reference | Separation, actual angle, sufficient bound and effective conditioning |

<a id="table-s3"></a>

## Table S3: AmgX configuration

The complete steady and transient comparisons use the same AmgX hierarchy settings. Original inactive systems use the absolute target $10^{-11}\\|b\\|_2$, with $10^{-11}$ used when $b=0$. The separate correction protocol in Section S3.5.1 specifies error-equation targets and remaining iteration budgets.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s3.tex).

| Setting | Value |
| --- | --- |
| Deterministic mode | Enabled |
| Outer solver; arithmetic | Preconditioned conjugate gradient (PCG); double precision |
| Convergence; residual norm; iteration cap | Absolute; Euclidean; 10000 for Cartesian, 50000 for body-fitted |
| Preconditioner; construction | Algebraic multigrid (AMG); classical |
| Coarsening; interpolation | Parallel maximal independent set (PMIS); distance-two (D2) |
| Smoother; relaxation | Symmetric multicolor Gauss–Seidel; 1 |
| Pre-sweeps; post-sweeps | 1; 1 |
| Cycle; cycles per application; maximum levels | V; 1; 25 |
| Coarse solver | Dense lower–upper (LU) factorization |

<a id="table-s4"></a>

## Table S4: Timing boundaries

The timing intervals partition the measured complete optimization sequence. Their sum equals the complete time. Preparation costs use separate enclosing process measurements.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s4.tex).

| Interval | Included work |
| --- | --- |
| Assembly and target construction | Spatial or space–time operators, reduced matrix, desired states and right-hand sides |
| Reference construction | Full-domain spatial and temporal factors |
| Inactive-system and coarse-space setup | Inactive matrices and right-hand sides; reference restriction or recycling transfer and selection; rank-revealing orthogonalization, coarse products and factorization, or AMG hierarchy construction |
| Initialization and iteration | Initial correction and residual; iterative solver recurrence |
| Conversion and transfers | Matrix and vector conversion; uploads and solution downloads |
| Verification and active-set updates | Independently recomputed residuals, state and multiplier recovery, KKT components and constraint updates |
| Remaining work | Solver resource creation, synchronization, cleanup and other host work |

<a id="table-s5"></a>

## Table S5: Independent body-fitted verification

Independent bounded-variable least squares verifies the mesh PDAS solutions. State and objective errors are relative to the independent solution. KKT is the largest of the five stated KKT components of the PDAS solution, and Dimension counts the optimized DOFs. Uniform and unequal entries denote complete two-slab optimizations.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s5.tex).

| Geometry | Form | Target | Dimension | State error | Objective error | KKT |
| --- | --- | --- | --- | --- | --- | --- |
| Bore-in-block | Steady | 1 | 360 | 2.27e-15 | 0.00e+00 | 1.61e-14 |
| Bore-in-block | Steady | 2 | 360 | 7.31e-15 | 0.00e+00 | 1.13e-14 |
| Bore-in-block | Uniform | 1 | 720 | 2.09e-14 | 0.00e+00 | 2.54e-14 |
| Bore-in-block | Uniform | 2 | 720 | 2.67e-14 | 0.00e+00 | 1.40e-14 |
| Bore-in-block | Unequal | 1 | 720 | 4.23e-14 | 1.66e-16 | 1.77e-14 |
| Bore-in-block | Unequal | 2 | 720 | 5.39e-14 | 1.53e-16 | 1.81e-14 |
| Transformer subproblem | Steady | 1 | 51 | 6.67e-15 | 0.00e+00 | 1.71e-15 |
| Transformer subproblem | Steady | 2 | 51 | 3.27e-15 | 2.16e-16 | 8.67e-16 |
| Transformer subproblem | Uniform | 1 | 102 | 3.15e-15 | 0.00e+00 | 2.38e-15 |
| Transformer subproblem | Uniform | 2 | 102 | 3.05e-15 | 0.00e+00 | 2.03e-15 |
| Transformer subproblem | Unequal | 1 | 102 | 4.95e-15 | 0.00e+00 | 2.97e-15 |
| Transformer subproblem | Unequal | 2 | 102 | 4.39e-15 | 1.50e-16 | 3.46e-15 |

<a id="table-s6"></a>

## Table S6: Independent Cartesian verification

Independent bounded least squares corroborates the complete PDAS solutions. The last three columns give maxima across the three PDAS solvers for each problem. Query indices 0, 1 and 2 correspond to $\theta=0,1/2,1$. KKT entries are the largest of the five quantities defined in the main protocol.

[Numerical reproduction guide](reference-policy-study.md#frozen-complete-optimization-campaign). [Original table source](../examples/manuscript_tables/sources/table-s6.tex).

| Family | Query | Reference KKT | PDAS KKT | Relative state | Relative objective |
| --- | --- | --- | --- | --- | --- |
| Steady | 0 | 4.66e-14 | 2.58e-12 | 1.82e-12 | 0.00e+00 |
| Steady | 1 | 1.20e-13 | 5.55e-12 | 1.01e-13 | 0.00e+00 |
| Steady | 2 | 3.98e-14 | 3.99e-12 | 1.09e-12 | 0.00e+00 |
| Uniform trajectory | 0 | 3.02e-14 | 3.05e-12 | 6.21e-12 | 2.07e-16 |
| Uniform trajectory | 1 | 2.97e-14 | 3.81e-12 | 5.54e-13 | 1.82e-16 |
| Uniform trajectory | 2 | 2.98e-14 | 2.58e-12 | 8.16e-14 | 0.00e+00 |
| Unequal-step trajectory | 0 | 2.55e-14 | 3.23e-12 | 2.08e-13 | 0.00e+00 |
| Unequal-step trajectory | 1 | 2.33e-14 | 6.96e-12 | 4.10e-13 | 2.11e-16 |
| Unequal-step trajectory | 2 | 6.84e-15 | 2.80e-12 | 1.43e-13 | 0.00e+00 |

<a id="table-s7"></a>

## Table S7: Time refinement and replay

Time refinement and replay of the optimized trajectories. The verified column counts discrete optimizations, over all slab counts, that meet the residual and KKT criteria, agree with a direct-PDAS solution at the coarsest slab count, and whose forward replays on 128 and 256 steps are solved to the residual tolerance; it does not establish feasibility between time levels. The difference columns give the largest and root-mean-square temperature difference between the eight-slab and 64-slab optima. Replay change and exceedance are diagnostics of the eight-slab controls, and the last column of the 64-slab controls. A dash marks a quantity without a verified optimization or without a replay. The bore-in-block advective operator is energy-stable (Table S1). All temperatures are in kelvin.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s7.tex).

| Model | Horizon | Transport | Discrete verified | Difference | RMS | Replay change | Exceedance | 64-slab exceedance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | 600 s | skew | 10/10 | 5.86 | 0.47 | 0.37 | 2.87 | 0.40 |
| Transformer | 1 h | skew | 10/10 | 6.29 | 0.47 | 0.45 | 0.82 | – |
| Bore 1 | $T_f=0.1$ | advective | 10/10 | 159.09 | 15.56 | 11.44 | 59.87 | 36.95 |
| Transformer | 600 s | advective | 6/10 | – | – | $9.3\times10^{151}$ | $3.2\times10^{84}$ | – |
| Transformer | 1 h | advective | 0/10 | – | – | – | – | – |

<a id="table-s8"></a>

## Table S8: Complete timing ranges and GPU allocation

Complete time and sampled GPU allocation of the primary body-fitted comparisons. Each cell gives the median time in seconds, the observed range over the converged repetitions, and the maximum sampled GPU allocation in gibibytes (GiB), in that order. A count in parentheses gives converged and attempted repetitions when they differ. Rank is the requested reference or recycling rank.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s8.tex).

| Geometry | Form | Rank | Jacobi-CG | RefDef | Recycling | AmgX |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer | Steady | 100 | 24.961<br>[24.817, 25.328]<br>1.000 | 10.532<br>[10.494, 10.768]<br>1.041 | 35.666<br>[35.431, 36.059]<br>1.154 | 17.535<br>[17.460, 17.956]<br>1.656 |
| Transformer | 4 slabs | 100 | 38.564<br>[38.496, 38.699]<br>1.002 | 22.543<br>[22.504, 22.678]<br>1.137 | 73.053<br>[71.843, 74.714]<br>1.971 | 51.116<br>[50.708, 51.438]<br>1.656 |
| Bore 1 | Steady | 100 | 6.358<br>[6.275, 6.462]<br>1.000 | 3.805<br>[3.785, 3.899]<br>1.002 | 5.698<br>[5.642, 5.707]<br>1.045 | 8.740<br>[8.607, 8.980]<br>1.656 |
| Bore 1 | 4 slabs | 100 | 11.752<br>[11.604, 11.953]<br>1.000 | 10.113<br>[10.028, 10.279]<br>1.078 | 13.942<br>[13.691, 14.121]<br>1.471 | 34.289<br>[34.055, 35.202]<br>2.209 |
| Bore 2 | Steady | 100 | 57.602<br>[57.213, 57.827]<br>1.016 | 32.481<br>[32.156, 32.616]<br>1.385 | 48.158<br>[46.221, 48.746]<br>2.516 | 102.570<br>[102.163, 102.815]<br>2.209 |
| Bore 2 | 4 slabs | 100 | 162.978<br>[157.208, 174.287]<br>1.893 | 142.336<br>[136.253, 143.655]<br>4.760 | 176.113<br>[169.720, 189.251]<br>11.141 | 527.429<br>[520.974, 535.049]<br>4.484 |

<a id="table-s9"></a>

## Table S9: Cartesian residual and KKT maxima

The residual and KKT maxima cover the 300 complete Cartesian sequences of Section S3.2.3. The columns give the original relative linear residual and maxima of stationarity, primal feasibility, dual feasibility, complementarity and projected gradient. Primal feasibility is absolute; the other KKT quantities use the normalization in the main protocol.

[Numerical reproduction guide](reference-policy-study.md#frozen-complete-optimization-campaign). [Original table source](../examples/manuscript_tables/sources/table-s9.tex).

| Method | Sequences | Residual | Stationarity | Primal | Dual | Complementarity | Projected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jacobi-CG | 65 | 1.00e-11 | 6.88e-10 | 1.21e-09 | 0.00e+00 | 0.00e+00 | 1.21e-09 |
| RefDef | 85 | 1.00e-11 | 5.97e-10 | 1.21e-09 | 0.00e+00 | 0.00e+00 | 1.21e-09 |
| Recycling | 85 | 1.00e-11 | 7.14e-10 | 1.21e-09 | 0.00e+00 | 0.00e+00 | 1.21e-09 |
| AmgX | 65 | 1.36e-11 | 2.12e-09 | 1.21e-09 | 0.00e+00 | 0.00e+00 | 2.12e-09 |

<a id="table-s10"></a>

## Table S10: Matched body-fitted transfer replays

Matched mesh replays measure the information supplied by two specified transfers. Each trace replays the first repetition of the corresponding sequence of main Table 2; the transformer trace uses the skew-symmetric transport. Systems counts the inactive systems of each trace; activated and released count constraint changes after the first restriction, including those between queries. Energy fraction is the median initial energy-error fraction removed by the coarse correction. Iterations and time sum the per-kernel medians over the sampled systems, and Kernels counts these systems. Each kernel has three repetitions, all meeting the residual criterion.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s10.tex).

| Trace | Systems | Activated/released | Transfer | Kernels | Rank | Energy fraction | Iterations | Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bore 2, steady | 120 | 30091/42345 | Direct restriction | 8 | 100 | 0.322 | 3834 | 12.749 |
|  |  |  | Zero-extension transfer | 8 | 100 | 0.062 | 16415 | 49.626 |
| Bore 2, 4 slabs | 137 | 121018/168841 | Direct restriction | 8 | 100 | 0.248 | 7731 | 157.237 |
|  |  |  | Zero-extension transfer | 8 | 100 | 0.052 | 19189 | 378.354 |
| Transformer, 4 slabs | 45 | 769/932 | Direct restriction | 8 | 100 | 0.185 | 13142 | 36.028 |
|  |  |  | Zero-extension transfer | 8 | 100 | 0.158 | 13175 | 35.667 |

<a id="table-s11"></a>

## Table S11: Deflated spectra of small replays

Median condition numbers of the small matched replays in Jacobi-scaled coordinates: the original preconditioned matrix and the deflated operator after each transfer.

[Numerical reproduction guide](reference-policy-study.md#frozen-complete-optimization-campaign). [Original table source](../examples/manuscript_tables/sources/table-s11.tex).

| Trace | Original | Direct restriction | Zero-extension transfer |
| --- | --- | --- | --- |
| $6^3$ | 296 | 9.84 | 209 |
| $4^3\times4$ | 67.4 | 18.8 | 50.5 |

<a id="table-s12"></a>

## Table S12: Complete-sequence retention ablation

Complete-sequence retention ablation. Times are median complete-sequence seconds over five repetitions of 16 targets. The ratio divides the median time of zero-extension transfer by that of direct restriction; brackets give the envelope of ratios between extreme repetition times. The iteration ratio divides their median inner iterations.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s12.tex).

| Geometry | Form | Jacobi-CG | Direct restriction | Zero extension | Ratio [range] | Iteration ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer | Steady | 26.277 | 11.033 | 21.478 | 1.95 [1.89, 1.99] | 2.14 |
| Transformer | 4 slabs | 39.925 | 23.237 | 28.357 | 1.22 [0.92, 1.32] | 1.25 |
| Bore 1 | Steady | 6.581 | 3.890 | 9.285 | 2.39 [2.35, 2.47] | 4.63 |
| Bore 1 | 4 slabs | 11.625 | 9.944 | 15.326 | 1.54 [1.48, 1.57] | 2.60 |
| Bore 2 | Steady | 55.702 | 31.059 | 67.219 | 2.16 [2.11, 2.20] | 4.24 |
| Bore 2 | 4 slabs | 158.641 | 138.112 | 184.802 | 1.34 [1.27, 1.37] | 2.36 |

<a id="table-s13"></a>

## Table S13: Conditional conditioning certificates

Controlled $6^3$ CHT restrictions distinguish angle bounds from conditioning-improvement certificates. Every row retains rank one. The columns give removed nodes $q$, minimum restricted singular value $\sigma$, subspace residual, separation, actual largest-angle sine, sufficient angle bound $\epsilon$, original condition number, predicted upper bound $K$ and measured effective condition number.

[Numerical reproduction guide](reference-policy-study.md#frozen-complete-optimization-campaign). [Original table source](../examples/manuscript_tables/sources/table-s13.tex).

| $q$ | $\sigma$ | $\\|F\\|_F$ | $\eta$ | $\sin\theta$ | $\epsilon$ | $\kappa(B)$ | $K$ | $\kappa_{\rm eff}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.00000 | 0.00064 | 0.02466 | 0.0012 | 0.0259 | 350.94 | 102.16 | 92.60 |
| 8 | 0.99998 | 0.00264 | 0.02475 | 0.0060 | 0.1067 | 350.49 | 134.67 | 92.29 |
| 32 | 0.99981 | 0.00747 | 0.02502 | 0.0206 | 0.2988 | 347.69 | 228.52 | 91.12 |
| 48 | 0.99957 | 0.00893 | 0.02532 | 0.0261 | 0.3531 | 344.66 | 257.21 | 89.93 |
| 64 | 0.99907 | 0.01646 | 0.02606 | 0.0395 | 0.6326 | 336.53 | 437.47 | 86.70 |
| 100 | 0.99582 | 0.02714 | 0.02750 | 0.0617 | 0.9917 | 295.46 | 672.15 | 78.99 |
| 120 | 0.98895 | 0.05636 | 0.03060 | 0.1178 | 1.0000 | 211.15 | 502.17 | 62.59 |

<a id="table-s14"></a>

## Table S14: Supplementary sensitivity comparisons

Supplementary comparisons under the procedure and criteria of main Section 3.4. Columns follow main Table 4: times are median complete-sequence seconds over the converged repetitions, brackets give the envelope of ratios between extreme repetition times, and the last column gives converged and attempted repetitions of each method that fails to converge in some repetition. $^\dagger$Compared with Jacobi-CG alone, or on Bore 4 with Jacobi-CG and AmgX. Level 0 denotes the packaged mesh and level $\ell$ its $\ell$-th refinement. Unless the setting names a reference level or rank, the transformer and Bore 2 rows use the packaged-mesh reference with total rank 100, and the Bore 3 rows use the packaged-mesh reference with rank 200; the sorted rows and the four-method Bore 3 row give the matching nominal comparisons. The Bore 4 envelope combines measurements from separate jobs.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s14.tex).

| Problem | Setting | Unknowns | RefDef | Fastest alternative | Speedup [range] | Iteration ratio | Not converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sorted order | transformer | 10,830 | 10.6 | AmgX 17.4 | 1.65 [1.56, 1.70] | 4.19 | – |
| Sorted order | Bore 2 | 36,197 | 23.1 | recycling 39.0 | 1.69 [1.62, 1.72] | 4.81 | – |
| Sorted order | Bore 2, 4 slabs | 144,788 | 70.7 | Jacobi-CG 91.8 | 1.30 [1.21, 1.46] | 2.72 | – |
| Transformer steady | rtol $10^{-8}$ | 10,830 | 8.6 | AmgX 13.8 | 1.61 [1.52, 1.69] | 4.29 | – |
| Transformer steady | rtol $10^{-6}$ | 10,830 | 6.5 | AmgX 9.5 | 1.47 [1.43, 1.53] | 4.43 | – |
| Bore 2 steady | rtol $10^{-8}$ | 36,197 | 17.9 | recycling 29.0 | 1.63 [1.59, 1.66] | 4.77 | – |
| Bore 2 steady | rtol $10^{-6}$ | 36,197 | 14.3 | recycling 21.0 | 1.47 [1.44, 1.50] | 4.57 | – |
| Bore 3 steady | rtol $10^{-8}$ | 308,171 | 299.6 | Jacobi-CG 401.3 | 1.34 [1.23, 1.48] | 5.18 | – |
| Bore 3 steady | rtol $10^{-6}$ | 308,171 | 194.5 | Jacobi-CG 239.6 | 1.23 [1.21, 1.26] | 4.72 | – |
| Bore 2 steady | $\alpha=10^{-5}$ | 36,197 | 23.4 | recycling 39.8 | 1.70 [1.66, 1.73] | 4.68 | – |
| Bore 2 steady | $\alpha=10^{-4}$ | 36,197 | 22.5 | recycling 37.5 | 1.67 [1.62, 1.72] | 4.60 | – |
| Transformer steady | $\alpha=10^{-13}$ | 10,830 | no method converges at every query |  |  |  |  |
| Transformer steady | $\alpha=10^{-12}$ | 10,830 | no method converges at every query |  |  |  |  |
| Bore 2 steady | cold start | 36,197 | 99.7 | Jacobi-CG 199.3 | 2.00 [1.99, 2.03] | 3.73 | – |
| Bore 3 steady | cold start | 308,171 | 1605.0 | Jacobi-CG 2498.1 | 1.56 [1.54, 1.57] | 4.31 | – |
| Transformer steady | 64 targets | 10,830 | 35.5 | Jacobi-CG 77.2 | 2.17 [2.14, 2.21] | 3.56 | AmgX 0/5 |
| Transformer, 8 slabs | 64 targets | 86,640 | 83.4 | Jacobi-CG 136.5 | 1.64 [1.61, 1.66] | 2.67 | – |
| Bore 2 steady | 64 targets | 36,197 | 45.5 | recycling 64.4 | 1.41 [1.38, 1.46] | 5.11 | – |
| Bore 2, 4 slabs | 64 targets | 144,788 | 149.4 | recycling 187.6 | 1.26 [1.22, 1.32] | 2.78 | – |
| Transformer steady | 256 targets | 10,830 | 151.8 | Jacobi-CG 265.0 | 1.75 [1.63, 1.82] | 2.79 | AmgX 0/5, recycling 0/5 |
| Bore 2 steady | 256 targets | 36,197 | 97.2 | recycling 125.6 | 1.29 [1.24, 1.32] | 5.23 | – |
| Transformer steady | 4 loads $\times$ 4 bounds | 10,830 | 10.5 | AmgX 16.1 | 1.54 [1.47, 1.61] | 3.94 | – |
| Transformer, 8 slabs | 4 loads $\times$ 4 bounds | 86,640 | 26.5 | Jacobi-CG 38.1 | 1.44 [1.39, 1.51] | 2.42 | – |
| Bore 3 steady$^\dagger$ | level 0, rank 50 | 308,171 | 343.7 | Jacobi-CG 564.9 | 1.64 [1.38, 1.87] | 3.52 | – |
| Bore 3 steady$^\dagger$ | level 0, rank 100 | 308,171 | 327.0 | Jacobi-CG 566.4 | 1.73 [1.46, 1.98] | 4.58 | – |
| Bore 3 steady$^\dagger$ | level 0, rank 200 | 308,171 | 343.5 | Jacobi-CG 565.3 | 1.65 [1.40, 1.87] | 5.33 | – |
| Bore 3 steady$^\dagger$ | level 0, rank 300 | 308,171 | 372.7 | Jacobi-CG 560.3 | 1.50 [1.32, 1.71] | 5.75 | – |
| Bore 3 steady$^\dagger$ | level 1, rank 50 | 308,171 | 303.9 | Jacobi-CG 514.5 | 1.69 [1.53, 1.87] | 3.66 | – |
| Bore 3 steady$^\dagger$ | level 1, rank 100 | 308,171 | 283.3 | Jacobi-CG 511.8 | 1.81 [1.66, 2.02] | 4.98 | – |
| Bore 3 steady$^\dagger$ | level 1, rank 400 | 308,171 | 307.1 | Jacobi-CG 521.7 | 1.70 [1.69, 1.71] | 9.24 | – |
| Bore 3 steady$^\dagger$ | level 2, rank 200 | 308,171 | 378.6 | Jacobi-CG 523.2 | 1.38 [1.38, 1.41] | 7.68 | – |
| Bore 3 steady | level 0, four methods | 308,171 | 427.5 | Jacobi-CG 575.9 | 1.35 [1.30, 1.55] | 5.33 | – |
| Bore 4 steady$^\dagger$ | level 3, rank 200 | 2,540,695 | 15189.6 | AmgX 9851.0 | 0.65 [0.41, 0.66] | – | Jacobi-CG 0/5 |
| Cartesian | $12^3$, 8 slabs, global | 13,824 | 5.0 | Jacobi-CG 5.3 | 1.05 [1.04, 1.09] | 1.96 | – |
| Cartesian | $16^3$, 8 slabs, global | 32,768 | 9.6 | Jacobi-CG 11.3 | 1.18 [1.15, 1.21] | 2.06 | – |
| Cartesian | $24^3$, 8 slabs, global | 110,592 | 29.0 | Jacobi-CG 35.9 | 1.24 [1.20, 1.26] | 2.03 | – |
| Cartesian | $32^3$, 8 slabs, global | 262,144 | 87.2 | Jacobi-CG 95.4 | 1.09 [1.05, 1.12] | 2.01 | – |

<a id="table-s15"></a>

## Table S15: Single-repetition and solver-limit comparisons

Single-repetition comparisons at the longest horizons and the solver limits. Columns follow main Table 4; each row times one sequence per method, so the speedup is a single ratio without an envelope. The rows with $S=4$ and $S=6$ time RefDef in a separate job from the alternatives.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s15.tex).

| Problem | Setting | Unknowns | RefDef | Fastest alternative | Speedup | Iteration ratio | Not converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | 16 slabs, global | 173,280 | 15.0 | Jacobi-CG 21.0 | 1.40 | 2.72 | – |
| Transformer | 16 slabs, $S=4$ | 173,280 | 14.9 | Jacobi-CG 21.0 | 1.41 | 2.35 | – |
| Transformer | 32 slabs, global | 346,560 | 26.5 | Jacobi-CG 29.1 | 1.10 | 2.31 | – |
| Transformer | 32 slabs, $S=4$ | 346,560 | 26.1 | Jacobi-CG 29.1 | 1.11 | 2.43 | – |
| Cartesian | $16^3$, 64 slabs, global | 262,144 | 48.8 | Jacobi-CG 37.6 | 0.77 | 1.08 | – |
| Cartesian | $16^3$, 64 slabs, $S=6$ | 262,144 | 48.2 | Jacobi-CG 37.6 | 0.78 | 2.52 | – |
| Transformer, refined twice | steady | 166,440 | – | AmgX 76.6 | – | – | RefDef 0/1 |
| Transformer, refined twice | 4 slabs | 665,760 | no method converges at every query |  |  |  |  |
| Bore 4 | outer cap 100 | 2,540,695 | no method converges at every query |  |  |  |  |

<a id="table-s16"></a>

## Table S16: CPU direct and block-in-time comparisons

CPU factorization solvers beside the GPU comparisons of the same problems. Entries are median complete-sequence seconds over five 16-target repetitions. Block denotes the block-in-time CG of the main protocol, and a dash marks a steady problem.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s16.tex).

| Problem | Direct (CPU) | Block (CPU) | RefDef (GPU) | Fastest GPU alternative |
| --- | --- | --- | --- | --- |
| Transformer steady | 5.0 | – | 10.6 | AmgX 17.4 |
| Bore 1 steady | 18.1 | – | 4.2 | recycling 6.0 |
| Bore 2 steady | 3281.4 | – | 23.1 | recycling 39.0 |
| Transformer, 8 slabs | 343.3 | 44.3 | 27.4 | Jacobi-CG 44.3 |
| Bore 1, 8 slabs | 10561.9 | 196.9 | 10.2 | Jacobi-CG 11.0 |

<a id="table-s17"></a>

## Table S17: Optimized fields under different bounds

The optimized Bore 1 fields satisfy three upper temperature bounds. Extrema and active fractions cover all optimized state DOFs. The tracking error uses the spatial or space–time objective weights. Temperatures and errors are dimensionless; the active fraction is a percentage.

[Numerical reproduction guide](../examples/temperature_bounds/README.md). [Original table source](../examples/manuscript_tables/sources/table-s17.tex).

| Form | Bound | Minimum | Maximum | Tracking error | Active (%) |
| --- | --- | --- | --- | --- | --- |
| Steady | 0.20 | -0.0004 | 0.200 | 0.193 | 9.9 |
| Steady | 0.25 | -0.0004 | 0.250 | 0.185 | 7.6 |
| Steady | 0.30 | -0.0004 | 0.300 | 0.179 | 6.1 |
| Four slabs | 0.20 | -0.0004 | 0.200 | 0.133 | 6.9 |
| Four slabs | 0.25 | -0.0004 | 0.250 | 0.127 | 4.7 |
| Four slabs | 0.30 | -0.0004 | 0.300 | 0.123 | 3.3 |

<a id="table-s18"></a>

## Table S18: Earlier advective transformer comparisons

Transformer comparisons with the original advective transport, whose operator has growing modes. Columns follow main Table 4.

[Numerical reproduction guide](prescribed-speedup-v21.md#reproduce-the-tables). [Original table source](../examples/manuscript_tables/sources/table-s18.tex).

| Problem | Setting | Unknowns | RefDef | Fastest alternative | Speedup [range] | Iteration ratio | Not converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | steady | 10,830 | 10.8 | AmgX 17.5 | 1.62 [1.51, 1.65] | 4.19 | – |
| Transformer | 4 slabs | 43,320 | 22.1 | Jacobi-CG 38.9 | 1.76 [1.71, 1.78] | 2.94 | – |
| Transformer | 8 slabs | 86,640 | 27.8 | Jacobi-CG 45.2 | 1.63 [1.62, 1.66] | 2.74 | – |
| Transformer day | 24 hourly windows | 86,640 | 37.8 | Jacobi-CG 56.7 | 1.50 [1.47, 1.50] | 2.44 | – |
| Overloads | transformer, 8 slabs | 86,640 | 25.5 | Jacobi-CG 36.9 | 1.44 [1.43, 1.46] | 2.46 | – |
| Off-design flow | transformer | 10,830 | 21.6 | recycling 47.9 | 2.22 [2.16, 2.24] | – | AmgX ($10^{-2}$) 0/5, AmgX ($10^{-3}$) 0/5, AmgX 0/5, Jacobi-CG 0/5 |
| Shuffled order | transformer | 10,830 | 13.2 | AmgX 17.2 | 1.30 [1.28, 1.36] | 3.32 | – |

<a id="table-s19"></a>

## Table S19: Earlier residual-correction comparisons

Residual-correction comparisons preserve the finer-transformer four-target problems and final accuracy requirements. Complete sequences and target solves are counted separately. Times give the median and observed range in seconds over five complete repetitions. Residual and KKT columns give maxima across completed sequences; corrections count all error equations.

[Numerical reproduction guide](mesh-showcases.md#choose-the-numerical-procedure). [Original table source](../examples/manuscript_tables/sources/table-s19.tex).

| Form | Method | Sequences | Targets | Time [range] | Residual | KKT | Corrections |
| --- | --- | --- | --- | --- | --- | --- | --- |
| steady | Jacobi-CG | 5/5 | 20/20 | 39.300 [39.166, 40.001] | 9.47e-11 | 1.86e-09 | 50 |
| steady | RefDef | 5/5 | 20/20 | 29.251 [29.057, 29.578] | 9.56e-11 | 8.02e-09 | 20 |
| steady | Recycling | 5/5 | 20/20 | 64.140 [63.960, 64.638] | 9.09e-11 | 2.40e-09 | 45 |
| steady | AmgX | 5/5 | 20/20 | 23.109 [23.063, 23.172] | 7.06e-11 | 1.41e-09 | 0 |
| transient | Jacobi-CG | 5/5 | 20/20 | 65.404 [64.958, 65.741] | 4.82e-11 | 4.57e-10 | 110 |
| transient | RefDef | 5/5 | 20/20 | 64.986 [64.874, 65.931] | 9.69e-11 | 5.05e-09 | 65 |
| transient | Recycling | 5/5 | 20/20 | 110.645 [109.706, 114.649] | 9.75e-11 | 6.88e-10 | 132 |
| transient | AmgX | 5/5 | 20/20 | 93.350 [92.750, 93.869] | 9.71e-11 | 2.39e-09 | 14 |
