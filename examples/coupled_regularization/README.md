# Bounded regularization study

This study varies the control penalty through four predeclared values,
`1e-14`, `1e-13`, `1e-12` and `1e-11`. Geometry, material properties, coupled flow
equations, desired temperatures, physical bounds, 64 time slabs, 600 s horizon,
60 s target startup and final accuracy remain fixed. Changing regularization
changes the optimization objective. Results at different values therefore remain
separate; each method comparison uses one common objective.

The first stage solves complete constrained Gauss--Newton subproblems at a common
initial temperature trajectory. It reconstructs the coupled state and control
derivative, then recomputes the gradient, Hessian, preconditioning diagonal and
active sets for each regularization value. Saved gradients, secants, inactive
masks and right-hand sides from the preceding study are not reused.

The three target identifiers are 7, 15 and 14. The screen compares velocity-frozen
preconditioning alone with reference ranks 4, 8 and 16. A seeded, 48-step
energy-metric Krylov construction supplies nested reference prefixes, separately
for each regularization value. Each space stays fixed across its target
quadratics. Three independently timed repetitions use cyclic method ordering.
Single-vector sparse tangent solves and these small reference spaces use CPUs.

## Run

Use a compute-node allocation with eight CPU threads. The input trace and baseline
are those described in [the fixed-setting study](../coupled_completion_v9.md).
The baseline and trajectory checksums are verified before use.

```sh
uv sync --frozen --extra study --extra plot
uv run --no-sync python -m deflation_example.coupled_regularization \
  --baseline BASELINE --trace CAPTURE/linear-systems --alpha 1e-13 \
  --output regularization-1e-13
uv run --no-sync pytest tests/test_coupled_regularization.py
```

Repeat for every value in `protocol.json`; retain every output directory.
Each alpha worker has a 12-hour scheduler limit. Interrupted workers retain their
last written record. No incomplete comparison yields a completed-solve speedup.

Verify the four output directories and generate the cost figure with:

```sh
uv run --no-sync python -m deflation_example.coupled_regularization_report \
  --runs regularization-1e-14 regularization-1e-13 regularization-1e-12 regularization-1e-11 \
  --output regularization-summary --plot
uv run --no-sync pytest tests/test_coupled_regularization_report.py
```

The summary checks every target and repetition, independently evaluated residuals
and quadratic KKT components, checksummed increment fields, objective agreement,
source and runtime consistency, and construction accounting. It keeps incomplete
and unsuccessful attempts visible and selects at most one complete-optimization
candidate. It never converts an incomplete or capped control into a speedup.

## Interpretation and continuation

These are constrained quadratic subproblems, not complete nonlinear optimization
queries. The original linear residual and independent quadratic optimality tests
must both pass. The screening criterion also requires solution agreement with
the rank-zero control and actual deployment of a nonzero reference.

Online intervals include all three quadratic solves, every active-set update,
solver resources, verification and cleanup. Common physical reconstruction is
reported separately. The setup-inclusive model charges the full reference
construction once per three-quadratic sequence. Whole-screen memory includes
the shared reference and is not a per-method memory comparison.

At most one alpha/rank proceeds to complete nonlinear optimization, under the
selection rule frozen in `protocol.json`. That comparison retains Jacobi CG,
velocity-frozen CG, matched-rank recycling and fixed-reference deflation. Complete
optimization must establish its own nonlinear KKT accuracy and timing benefit.
Screening improvements alone supply no complete-optimization speedup claim.

## Explicit continuation after a time limit

The separately documented `continuation.json` applies after authorization to
finish missing repetitions. Keep the original records unchanged and confirm that
their scheduler jobs have stopped. Run each missing three-target repetition in a
fresh eight-thread CPU allocation with a twelve-hour limit:

```sh
uv run --no-sync python -m deflation_example.coupled_regularization_resume \
  --original regularization-1e-11 --baseline BASELINE --trace CAPTURE/linear-systems \
  --rank 16 --repetition 0 --output continuation-r16-repeat0
```

The driver rejects replacement of a recorded result, source changes, altered
physical inputs, mismatched numerical runtimes and reference checksum changes.
Every original numerical module remains byte-identical. The original reference
construction cost stays in the cost model; restart preparation is also measured
and reported separately. Fresh-process repetitions remain explicitly identified.

Pass every continuation directory to the summary command using
`--continuations DIR1 DIR2 ...`. The summary combines records in memory without
rewriting the original attempts. It checks each repetition against its matching
rank-zero result, rejects duplicate replacements and retains unsuccessful
outcomes. A completed numerical population does not erase its earlier timeouts.

## Complete nonlinear comparison

After the completed summary selects a setting, freeze the four-arm development
comparison. The command refuses an incomplete population or a different choice:

```sh
uv run --no-sync python -m deflation_example.coupled_regularization_complete settings \
  --screen regularization-summary/summary.json --output complete-settings.json
uv run --no-sync python -m deflation_example.coupled_regularization_complete run \
  --screen regularization-summary/summary.json --settings complete-settings.json \
  --optimization NOMINAL_OPTIMIZATION --baseline BASELINE \
  --initial-snapshot INITIAL_SNAPSHOT --initial-assessment ASSESSMENT.json \
  --arm reference --repetition 0 --output complete-reference-repeat0
```

Run all four arms (`jacobi`, `frozen`, `recycling`, `reference`) for repetitions
0, 1 and 2 in fresh processes. Use eight CPU threads, a 48-hour limit per complete
sequence and at most four simultaneous sequences. Rotate arm order by repetition.
The prescribed three targets, physical inputs, 64 slabs and accuracy are unchanged.
Retain every output and scheduler status, including time-limited sequences.

All arms use the same assessed initial temperature. The explicit
`shared_temperature` policy allows the new regularization but still verifies the
physical configuration, target, bound, snapshot and derivative-assessment hashes.
The optimizer recomputes flows, recovered control and derivatives; no earlier
gradients, secants or recycling vectors are imported. Ordinary solver use retains
the default requirement of identical regularization.

Reference deflation constructs its fixed full-domain space from the initial
coupled trajectory once per complete sequence. Construction includes the extra
flow evaluation, frozen inverse and the same 48-step seeded energy-metric Krylov
procedure used by the screen. Every nonlinear update uses the current coupled
equations. Recycling retains deployed coarse vectors and new search directions,
then selects with the current frozen inverse in the operator energy metric. Both
nonzero-rank arms use the selected rank; both controls use rank zero.

Summarize the twelve complete records with:

```sh
uv run --no-sync python -m deflation_example.coupled_confirmation_report \
  --settings complete-settings.json --records RUN1/record.json RUN2/record.json \
  RUN3/record.json RUN4/record.json RUN5/record.json RUN6/record.json \
  RUN7/record.json RUN8/record.json RUN9/record.json RUN10/record.json \
  RUN11/record.json RUN12/record.json --output complete-development-summary
```

The summary checks nonlinear accuracy, all inner attempts, nonzero reference
deployment, field and objective agreement, timing partitions and matched source
and runtime identities. Five fresh confirmation repetitions require a verified
development population and lower reference median time than the fastest tested
alternative. Neither the screening result nor an unfinished baseline establishes
complete nonlinear acceleration.

The [matched linear space–time control](../linear_spacetime/README.md) uses
the same alpha/rank and thermal trajectory discretization with the verified
isothermal velocity held fixed. It establishes a separate within-physics
comparison and does not change the running nonlinear protocol.
