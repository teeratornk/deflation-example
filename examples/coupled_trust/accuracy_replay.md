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

The optional `restriction="submatrix"` factory assembles the sparse frozen
normal operator once and slices that matrix for
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

## Globalization of one recorded quadratic

If matched high-accuracy inner solves leave active-set instability, the separate
`coupled_qp_globalization` diagnostic compares PDAS with gradient projection and
reduced CG on the saved quadratic. It keeps its derivative, bounds, diagonal,
initial zero step, and recorded quadratic tolerance fixed. Both methods use
independent original-residual checks and fresh weighted KKT checks.

```sh
uv run --no-sync python -m deflation_example.coupled_qp_globalization \
  --trace INPUT/inactive-trace-00 --baseline BASELINE --quadratic 2 \
  --method projected --rtol 1e-8 --output OUTPUT/projected
uv run --no-sync python -m deflation_example.coupled_qp_globalization \
  --trace INPUT/inactive-trace-00 --baseline BASELINE --quadratic 2 \
  --method pdas --rtol 1e-8 --output OUTPUT/pdas
```

The projected method follows the two-stage searches in Section 3 of
[Benson, McInnes and Moré, *GPCG: A Case Study in the Performance and Scalability
of Optimization Algorithms*](https://ftp.mcs.anl.gov/pub/tech_reports/reports/P768.pdf).
It uses residual-based CG stopping rather than that paper's objective-decrease
test. This diagnostic is separate from the published optimization algorithm.
Its timers include the projected searches and independent checks. A completed
quadratic at an intermediate tolerance does not establish nonlinear optimality.

After both attempts finish, retain every outcome in a summary:

```sh
uv run --no-sync python -m deflation_example.coupled_qp_globalization_report \
  --pdas OUTPUT/pdas/record.json --projected OUTPUT/projected/record.json \
  --output OUTPUT/comparison
```

The summary checks that both records use the same input, source, hardware,
accuracy and budget. An unsuccessful comparison remains visible and supplies
no completed-solve speedup. A successful quadratic repair requires separate
nonlinear integration tests before another optimization campaign.

## Inexact reduced directions

If accurately solved reduced directions dominate the projected method's cost,
vary their residual target while keeping the quadratic and its final KKT
criterion fixed. A projected direction is an intermediate search direction;
the projected search still checks objective decrease, and the final state
still needs the independent weighted KKT test. This experiment changes neither
the complete nonlinear optimizer nor its final accuracy requirements.

The bounded comparison declares targets `1e-2` and `1e-4`, with the preceding
`1e-8` attempt as the strict control. Use the same numerical source, baseline,
trace, initial zero step, preconditioner and resource limits for all arms:

```sh
uv run --no-sync python -m deflation_example.coupled_qp_globalization \
  --trace INPUT/inactive-trace-00 --baseline BASELINE --quadratic 2 \
  --method projected --rtol 1e-2 --budget-seconds 7200 --output OUTPUT/projected-1e-2
uv run --no-sync python -m deflation_example.coupled_qp_globalization \
  --trace INPUT/inactive-trace-00 --baseline BASELINE --quadratic 2 \
  --method projected --rtol 1e-4 --budget-seconds 7200 --output OUTPUT/projected-1e-4
uv run --no-sync python -m deflation_example.coupled_qp_inexact_report \
  --records OUTPUT/projected-1e-2/record.json OUTPUT/projected-1e-4/record.json \
    OUTPUT/projected/record.json --tolerances 1e-2 1e-4 1e-8 \
  --output OUTPUT/inexact-summary
```

The internal CG trigger is one tenth of the declared direction target. Each
returned direction receives an independent residual check at its declared
target. The recorded intermediate quadratic KKT tolerance is unchanged.
The summary retains every declared arm, checks the common input and numerical
environment, and compares iteration totals and accumulated search decreases
against the final records. A budget-limited direction remains a failed direction,
even when preceding feasible updates decreased the objective. A single diagnostic
attempt supplies no repeated-timing or nonlinear-optimization speedup.
