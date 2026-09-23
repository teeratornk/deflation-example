# Coupled reference reuse: matched preconditioning

This example tests whether a fixed full-domain reference adds value to the
velocity-frozen preconditioner. The transformer geometry, physical coefficients,
64-slab optimization, 600 s horizon, smooth 60 s startup, bounds and final accuracy
requirements remain unchanged. These are development comparisons. No coupled
reference-reuse speedup is established by the new implementation alone.

## Evidence and termination

The existing retention campaign names directories after the dispatcher stage.
`heldout-summary` contains the *selection* summary used to choose the held-out
configuration. `assessment-summary` contains the actual held-out evaluation.
The following commands reconstruct the numerical entries and record their input
hashes, without modifying the original records:

```sh
uv sync --frozen --extra study --extra plot
uv run --no-sync python -m deflation_example.coupled_completion_audit retention \
  --run-root RUNS --input RETENTION_CAMPAIGN --output RETENTION_AUDIT
uv run --no-sync python -m deflation_example.coupled_completion_audit forward \
  --run-root RUNS --input TERMINATION_LEDGER.json --output FORWARD_AUDIT
```

The forward archive includes cancelled and unstarted PIMPLE repetitions. It
preserves saved partial costs, separates scheduler termination from the last
worker checkpoint, and reports no completed-solve ratio for an incomplete run.
The completed Newton–Anderson trajectories support forward convergence only.

## Preconditioner and reference

The exact coupled Gauss–Newton operator remains `H`. The approximate inverse `K`
uses three stationary block-Jacobi sweeps of the positive-definite velocity-frozen
normal operator. The slab blocks retain the original temporal coupling in the
residual; the independent block solves define the preconditioner. Three fixed
sweeps define a fixed linear map. The host solver retains its separately tested
Polak–Ribiere recurrence and original-residual verification.

The new reference starts from 400 thermal candidate directions `C` on the full
state domain. It augments their span with `K H C`, using the initial undamped
coupled operator before optimization. After rank-revealing QR and a small SVD,
the independent candidate matrix `V` defines the Ritz pencil

```text
(H V)^T K (H V) a = lambda (V^T H V) a.
```

`K H` is self-adjoint in the `H` inner product. We select alternating low and high
Ritz directions; prefixes give nested ranks 20, 50, 100 and 200. This is a
specified selection rule, not a claim that both ends always need deflation.
Rank loss produces no replacement directions. The resulting full-domain basis
remains fixed; each inactive system receives its own restricted basis and exact
coarse factorization. The construction retains dense selected directions, and
its memory is charged alongside all intermediate candidate storage.

```sh
uv run --no-sync python -m deflation_example.coupled_retention_replay bank \
  --baseline BASELINE --trace CAPTURE/linear-systems --device hybrid --width 20 \
  --bank-selection frozen --output FROZEN_REFERENCE_BANK
uv run --no-sync python -m deflation_example.coupled_retention_replay replay \
  --baseline BASELINE --trace CAPTURE/linear-systems --bank FROZEN_REFERENCE_BANK \
  --quadratic 0 --device hybrid --width 20 --policy preconditioned_coupled \
  --rank 20 --preconditioner frozen --sweeps 3 --repetitions 3 --output REPLAY
```

Run rank-zero controls with `--policy jacobi --rank 0`, once for each
`--preconditioner jacobi` and `--preconditioner frozen`. Run the thermal reference
with `--policy thermal`. The same original systems, recorded initial guesses,
precision and acceptance checks apply to every method. Factory construction,
mask-dependent factorization, applications and cleanup are timed. Component
measurements are contained in totals and must not be added a second time.

For complete initialized optimization, `coupled_retention run` accepts
`--policy preconditioned_coupled --rank 20 --preconditioner frozen --sweeps 3`
alongside the same baseline, optimization, initial-snapshot and
initial-assessment arguments documented in `coupled_retention_v8.md`.
The reference is built during that computation; a saved bank cannot silently
remove its construction cost. Add `--targets` followed by the frozen target
identifiers for a single-process complete sequence. The initial snapshot seeds
only the first target. Later targets use the preceding verified solution, and
the original full-domain reference persists throughout the sequence.

## Development and confirmation

Previously examined systems remain development evidence. Replays screen costs
before complete optimization; they cannot establish a complete-time advantage.
After three development repetitions, freeze the source, rank and reference before
five fresh repetitions of three distinct targets selected from target fields
alone. Compare Jacobi, improved recycling, the velocity-frozen preconditioner,
and that preconditioner with the fixed reference. Recycling retains deployed
coarse vectors alongside new directions and selects in the attached inverse
preconditioner's energy-metric Ritz pencil. Use identical starts and final
linear, nonlinear, conservation and KKT requirements. The publication criterion
is at least 10% lower median complete time than the matched no-reference solver
and the fastest tested alternative, with every confirmation run verified. Report
the timing ranges. The confirmation reporter conservatively withholds its
publication gate when the reference and fastest-alternative timing ranges
overlap. The ratio remains a descriptive measurement in that case.

```sh
uv run --no-sync python -m deflation_example.coupled_confirmation_targets \
  --baseline BASELINE --optimization VERIFIED_NOMINAL_RUN --output TARGET_SELECTION
uv run --no-sync python -m deflation_example.coupled_confirmation_report \
  --settings FROZEN_SETTINGS.json --records RUN_A/record.json RUN_B/record.json \
  --output CONFIRMATION_SUMMARY
```

The reporter requires all four arms, three distinct targets and five fresh
repetitions for confirmation. It checks complete timing sums, unchanged final
tolerances, source/deployment matching and agreement of saved states and
objectives. Missing, failed or unverified comparisons cannot produce a speedup.
It also audits inner iteration totals, independently evaluated residuals,
deployed ranks and fallback counts. A reference-space publication claim requires
complete inner histories and actual use of a nonzero space in every reference
sequence. A final verified solution can follow rejected inner attempts; the
summary retains those attempts and their costs.

Generate the performance figure from the same settings and records:

```sh
uv run --no-sync python -m deflation_example.coupled_confirmation_figures \
  --settings FROZEN_SETTINGS.json --records RUN_A/record.json RUN_B/record.json \
  --output COMPLETE_SEQUENCE_FIGURES
```

The figure retains every independently timed repetition. Its cumulative curves
end at the measured sequence totals. Stacked components belong to one actual
middle-ranked run, so they sum to that run's total. The memory panel shows sampled
whole-process allocations. Failed runs remain visible, and missing comparisons
withhold the speedup summary. The output records input hashes and all plotted
values. Use `coupled_figures` for saved temperatures, applied source fields and
fluid velocities; those plots use stored discrete time levels.

Use compute nodes for numerical work and request a GPU only for CUDA-enabled
tests and block products. Preserve every capped, cancelled, memory-limited or
failed attempt. Temporal resolution requires separate checks for each applied
control; solving the discrete equations does not establish it.

## Tests

```sh
uv run --no-sync pytest tests/test_coupled_completion_audit.py \
  tests/test_coupled_preconditioned_reference.py tests/test_frozen_preconditioner.py \
  tests/test_coupled_retention_replay.py tests/test_coupled_retention_report.py \
  tests/test_coupled_selected_reference.py -m 'not gpu'
```

GPU block tests use the `coupled-gpu` extra on an allocated GPU. Test records and
numerical records identify their exact source separately; older measurements are
not relabeled as results of this implementation.

## Independent spectral diagnostic

The thermal-candidate Ritz construction samples its candidate span. A separate
full-domain Krylov diagnostic checks directions outside that span. It starts
from a fixed random vector, scaled by the nominal diagonal, without consulting
target loads or inactive sets. Forty-eight applications of the stationary
inverse-preconditioned operator build an energy-orthogonal space. The diagnostic
reports projected Ritz values, independently evaluated energy-norm residuals,
construction cost and sampled memory. These values do not certify the extremal
eigenvalues of the complete operator.

```sh
uv run --no-sync python -m deflation_example.coupled_krylov_reference \
  --baseline BASELINE --trace CAPTURE/linear-systems --steps 48 --rank 8 \
  --seed 20260923 --output KRYLOV_DIAGNOSTIC
uv run --no-sync pytest tests/test_coupled_krylov_reference.py
```

This construction runs on CPUs because its operator applications have one
column. It does not alter the frozen replay campaign or establish an
optimization-time benefit. A separate diagnostic compares prefixes of ranks
1, 2, 4 and 8 against the rank-zero frozen preconditioner, on the first three
recorded quadratics with three repetitions each. All five policies use CPU
operator applications in this low-rank comparison:

```sh
uv run --no-sync python -m deflation_example.coupled_retention_replay replay \
  --baseline BASELINE --trace CAPTURE/linear-systems --bank KRYLOV_DIAGNOSTIC \
  --quadratic 0 --policy krylov_coupled --rank 2 --device cpu --width 20 \
  --preconditioner frozen --sweeps 3 --repetitions 3 --output KRYLOV_REPLAY
```

Repeat for quadratics 0--2 and each declared rank; use `--policy jacobi --rank 0`
for the matched control. Preserve the entire construction cost once per replay
sum and distinguish those sums from complete optimization. The sampled Ritz
values and their residuals help interpret these comparisons; measured solver
work determines whether the selected directions repay their use.
