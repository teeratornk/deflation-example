# Skew-symmetric transformer comparisons and body-fitted baselines

This page documents the comparisons behind the body-fitted tables of the
prescribed-flow CHT study after the transformer transport was changed to its
skew-symmetric form. It extends the [scale, trajectory and operating-condition
comparisons](prescribed-speedup-v20.md): every transformer comparison of that
study was repeated with the skew-symmetric transport, and the six primary
body-fitted cases were timed again under one protocol.

RefDef denotes reference deflation, the full-domain reference restricted directly
to each inactive set; its method key in configurations and records remains
`reference`.

## Transport forms

The transformer carries a prescribed P2 Stokes velocity that is divergence-free
only weakly. The original advective form omits the divergence term; the
skew-symmetric form adds half of it,

```
a_skew(T, w) = ∫ 2πr c w (β_h · ∇T) + ½ ∫ 2πr c (∇ · β_h) T w,
```

so that its quadratic form equals the outflow heat flux and is nonnegative once
the inlet temperature is eliminated. [Transport forms](transport-forms.md)
derives the energy identity and lists the verification tests. An exact inertia
count of the symmetric part of every timed operator (symmetric factorization
without row interchanges; Sylvester's law of inertia) certifies the skew
transformer operator on the packaged mesh and both refinements and the
bore-in-block operators Bore 1 to Bore 3 as positive definite; the original
transformer operator has negative eigenvalues and grows under time refinement.

## Source

| Component | Branch | Commit |
| --- | --- | --- |
| Retention ablation (method `reference_zero`) | `study/prescribed-speedup-v21` | [`5ddc619`](https://github.com/teeratornk/deflation-example/tree/5ddc6197c060543c27255400247876ea67fd881b) |
| Stability certificates | `study/prescribed-speedup-v21` | [`6e22606`](https://github.com/teeratornk/deflation-example/tree/6e2260685b0f319bb682a0d1345c660b14cfaf73) |
| Timed comparisons, temporal studies, verification and transfer replays | `study/prescribed-speedup-v21` | [`297899f`](https://github.com/teeratornk/deflation-example/tree/297899fdba3a38b07d09db86114acdefbd4d5111) |
| Evidence digests and table rows | `study/prescribed-speedup-v21-evidence` | [`994d0c5`](https://github.com/teeratornk/deflation-example/tree/994d0c58a067536b15af578243d749fd6a3a96e2) |

The [records](../examples/prescribed_speedup_v21/README.md) contain the evidence
archive, the complete comparison listing and the checksums.

## Provenance of the tables and figures

Every timed comparison with the prefix `v21-` used commit `297899f`, which contains
the skew-symmetric transport (`transport_form=skew`); the retention ablation used
`5ddc619` and the stability certificates `6e22606`, whose transport code is
identical. The comparisons without the prefix are the Cartesian and bore-in-block
comparisons of the [v20 study](prescribed-speedup-v20.md#source); its transformer
comparisons used the original transport and appear here as diagnostics only. All
items in the table below regenerate from
[`evidence.tar.gz`](../examples/prescribed_speedup_v21/evidence.tar.gz) with the
export command of the last section; `rows.json` records the transport of every
tabulated comparison and the source digest of each replay and figure record.

| Manuscript item | Comparisons in the archive | Transport |
| --- | --- | --- |
| Table 2 and SI Table S8, all rows; Figure 5 | `v21-wave1/P-transformer-steady-skew`, `v21-wave1/P-transformer-x4-skew`, `v21-wave1/P-engine-L1-steady`, `v21-wave1/P-engine-L1-x4`, `v21-wave1/P-engine-L2-steady`, `v21-wave1/P-engine-L2-x4` | skew (transformer); advective, energy-stable (bore-in-block) |
| Figure 1(c) and 1(e) | record `reference-0` of `v21-wave1/P-transformer-x4-skew` and of `v21-wave1/P-engine-L2-x4` | skew; advective |
| Tables 4–6, transformer rows; Figure 7 | `v21-wave2/*-skew` (steady, 4 and 8 slabs, sorted and shuffled order, off-design flow), `wave15/O1s-*`, `wave15/O4s-*` | skew |
| Tables 4–6, Cartesian and bore-in-block rows; Figure 7, including the 256-query Bore 2 sequence | the comparisons without the `v21-` prefix in the [listing](../examples/prescribed_speedup_v21/populations.md) | Cartesian skew-symmetric differences; advective, energy-stable (bore-in-block) |
| SI Tables S14–S16, all rows | `v21-wave2/*-skew`, `v21-wave3/D-transformer-*-skew` and the comparisons without the `v21-` prefix | skew (transformer); Cartesian and bore-in-block as above |
| SI Table S10, all traces | `diagnostics/transfer/M-transformer-x4-skew-transfer.json`, `M-engine-L2-steady-transfer.json` and `M-engine-L2-x4-transfer.json`, replaying record `reference-0` of the corresponding `v21-wave1/P-*` comparison | skew (transformer); advective (Bore 2) |
| SI Table S5, transformer subproblem | `diagnostics/validation/V-validation-skew.json` | skew |
| SI Table S7, transformer rows | `temporal/T-transformer-600s-skew`, `temporal/hour-skew` (skew); `temporal/published-600s-advective`, `temporal/hour-advective` (diagnostics) | skew; advective |
| SI Table S12 (retention ablation), all rows; run commit `5ddc619` | `v21-wave6/X-transformer-*-skew-zero`, `v21-wave6/X-engine-L1-*-zero`, `v21-wave6/X-engine-L2-*-zero` | skew (transformer); advective, energy-stable (bore-in-block) |
| SI Table S1 | `diagnostics/certificates.json` | both forms |
| SI Table S18 (original-transport diagnostics) | `wave3/C-S4b-transformer-*`, `wave9/O*-transformer-*` | advective |

Three body-fitted items come from separate tagged records: Figure 4 and SI Table
S17 from the [temperature-bound records](https://github.com/teeratornk/deflation-example/tree/temperature-bounds-v1/examples/temperature_bounds),
and SI Table S19 from the
[finer-transformer refinement records](https://github.com/teeratornk/deflation-example/tree/mesh-cht-refinement-data-v1)
and the [guarded follow-up](https://github.com/teeratornk/deflation-example/tree/guarded-refinement-data-v1/examples/guarded_refinement);
these use the original transport.

## Definitions

The definitions of the [v20 page](prescribed-speedup-v20.md#definitions) apply.
In addition:

- **Speedup range:** the smallest and largest ratio of a converged time of the
  fastest alternative to a converged time of RefDef over the
  repetitions.
- **Not converged:** each method with a repetition that misses the criteria,
  with its converged and attempted repetitions.
- **Reduced coverage (†):** a comparison that times a subset of Jacobi-CG,
  recycling and AmgX.

## Comparisons

- **Primary body-fitted cases:** the transformer (skew transport), Bore 1 and
  Bore 2, steady and with four slabs, 16 targets, five repetitions of four methods.
- **Transformer twins:** every transformer comparison of the v20 study with the
  skew-symmetric transport — steady, 4 and 8 slabs, sorted and shuffled order,
  off-design flow, tolerances, regularization, 64 and 256 targets, loads and
  bounds, 16 and 32 slabs, the twice-refined mesh and the CPU direct and
  block-in-time solvers.
- **Temporal accuracy:** the transformer at 600 s and 1 h and Bore 1 optimized on
  4 to 64 slabs, with the piecewise-constant controls replayed on up to 4096 steps.
- **Support:** independent bounded least squares on 12 small mesh problems and
  matched transfer replays of the primary traces.
- **Retention ablation:** the six primary cases with Jacobi-CG, direct restriction
  and zero-extension transfer of the same reference in one job per case.

The [complete listing](../examples/prescribed_speedup_v21/populations.md)
reports every comparison with converged and attempted repetitions per method.
The comparisons with the original transport remain in the listing and in the
supporting information as diagnostics.

## Reproduce the tables

From a checkout of this revision, copy the archive and switch to the evidence commit:

```bash
cp examples/prescribed_speedup_v21/evidence.tar.gz /tmp/v21-evidence.tar.gz
git checkout --detach 994d0c58a067536b15af578243d749fd6a3a96e2
mkdir /tmp/v21 && tar -xzf /tmp/v21-evidence.tar.gz -C /tmp/v21
uv run --locked python -m deflation_example.v21_evidence export \
    --evidence /tmp/v21 --output /tmp/v21-out
```

`export` verifies every file of the archive against its manifest and writes the
table rows, the figure inputs, the summary and the listing to `/tmp/v21-out`.
The regenerated rows match the manuscript tables byte for byte. The export reads
the run digests and the archived diagnostics only and needs neither a GPU nor a
network connection.
