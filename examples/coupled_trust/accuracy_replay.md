# Inner accuracy and active-set decisions

This diagnostic replays the final inactive system of a completed
`quadratic_active_set_cap` attempt. It preserves the saved right-hand side,
initial guess, lower/upper bound assignments, temperature trajectory and secants.
The operator is reconstructed from checksum-verified fields. Independent flow,
thermal and balance checks, followed by a right-hand-side consistency check,
precede each solve. Every attempt receives a new output directory.

The declared first comparison uses residual targets `1e-4`, `1e-6`, `1e-8`
and `1e-10`, rank-zero frozen preconditioning and unchanged iteration caps.
All four start from the same recorded initial guess. These are diagnostic
inner targets; the complete-optimization criteria remain original residual
`1e-10` and KKT residual `1e-8`. No control, physical parameter or final
acceptance criterion changes.

Run on an allocated compute node using the source accompanying this example:

```sh
uv sync --frozen --extra study
uv run --no-sync python -m deflation_example.coupled_qp_accuracy \
  --record FAILED_ATTEMPT/record.json --baseline BASELINE \
  --rtol 1e-8 --output accuracy-1e-8
```

Repeat for every declared target in separate processes, with identical resource
limits. The diagnostic reports independently recomputed residuals, quadratic
KKT components, subsequent activations/releases, and differences from the
recorded solution. It saves every candidate and partition, including failed
solves. Reconstruction and preconditioner preparation are separate from the
inner solver's component timings. No optimized trajectory or complete-solve
speedup follows from a one-system replay.

Summarize the four paths in tolerance order, retaining missing paths and failures:

```sh
uv run --no-sync python -m deflation_example.coupled_qp_accuracy_report \
  --records accuracy-1e-4/record.json accuracy-1e-6/record.json \
    accuracy-1e-8/record.json accuracy-1e-10/record.json --output accuracy-summary
```

If tighter solves change the next partition substantially, a bounded replay of
the full failed quadratic is the next test. If the strict replay retains the
large activation/release changes, investigate active-set globalization on that
same quadratic. Neither outcome justifies increasing reference rank or launching
a nonlinear timing campaign before optimization converges.

The regression tests cover recorded-equation mismatches, both bound signs,
zero right-hand sides, converged initial guesses, input preservation, and failed
solver statuses:

```sh
uv run --no-sync pytest tests/test_coupled_qp_accuracy.py tests/test_coupled_qp_diagnostic.py
```

## Sparse preconditioner construction

The optional `restriction="submatrix"` factory uses the committed slicing
implementation from the neighboring prescribed-flow study (`4631927`). It
assembles the sparse frozen normal operator once and slices that matrix for
each inactive set. The true consistent-source and coupled operators remain
matrix-free. `restriction="product"` remains the default.

```sh
uv run --no-sync python -m deflation_example.coupled_preconditioner_setup \
  --trace FAILED_ATTEMPT/inactive-trace-00 --baseline BASELINE \
  --output frozen-setup
```

This comparison uses five evenly spaced saved masks from the final quadratic
and three alternating-order repetitions. Timings include first use. It charges
full-matrix construction as well as each restriction and factorization. The
same seeded residuals check preconditioner-action agreement. These component
measurements identify potential setup savings; they cannot establish a change
in complete optimization time. Tests cover changing masks, damping, consistent
source weighting, invalid indices and the timing sums.
