# Skew-symmetric transformer comparisons and body-fitted baselines

This page documents the comparisons behind the body-fitted tables of the
prescribed-flow CHT study after the transformer transport was changed to its
skew-symmetric form. It extends the [scale, trajectory and operating-condition
comparisons](prescribed-speedup-v20.md): every transformer comparison of that
study was repeated with the skew-symmetric transport, and the six primary
body-fitted cases were timed again under one protocol.

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
| Stability certificates | `study/prescribed-speedup-v21` | [`6e22606`](https://github.com/teeratornk/deflation-example/tree/6e2260685b0f319bb682a0d1345c660b14cfaf73) |
| Timed comparisons, temporal studies, verification and transfer replays | `study/prescribed-speedup-v21` | [`297899f`](https://github.com/teeratornk/deflation-example/tree/297899fdba3a38b07d09db86114acdefbd4d5111) |
| Evidence digests and table rows | `study/prescribed-speedup-v21-evidence` | [`713e6c3`](https://github.com/teeratornk/deflation-example/tree/713e6c3f3aa349630c7212962b2de46545bc3cec) |

The [records](../examples/prescribed_speedup_v21/README.md) contain the evidence
archive, the complete comparison listing and the checksums.

## Definitions

The definitions of the [v20 page](prescribed-speedup-v20.md#definitions) apply.
In addition:

- **Speedup range:** the smallest and largest ratio of a converged time of the
  fastest alternative to a converged time of reference deflation over the
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

The [complete listing](../examples/prescribed_speedup_v21/populations.md)
reports every comparison with converged and attempted repetitions per method.
The comparisons with the original transport remain in the listing and in the
supporting information as diagnostics.

## Reproduce the tables

From a checkout of this revision, copy the archive and switch to the evidence commit:

```bash
cp examples/prescribed_speedup_v21/evidence.tar.gz /tmp/v21-evidence.tar.gz
git checkout --detach 713e6c3f3aa349630c7212962b2de46545bc3cec
mkdir /tmp/v21 && tar -xzf /tmp/v21-evidence.tar.gz -C /tmp/v21
uv run --locked python -m deflation_example.v21_evidence export \
    --evidence /tmp/v21 --output /tmp/v21-out
```

`export` verifies every file of the archive against its manifest and writes the
table rows, the figure inputs, the summary and the listing to `/tmp/v21-out`.
The regenerated rows match the manuscript tables byte for byte. The export reads
the run digests and the archived diagnostics only and needs neither a GPU nor a
network connection.
