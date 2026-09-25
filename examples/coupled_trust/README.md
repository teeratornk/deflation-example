# Coupled temperature-step trust regions

This development example addresses expensive quadratic directions followed by
very short nonlinear steps. It is separate from the measured line-search
comparisons. The physical model, control regularization, targets and final
accuracy are unchanged. No coupled speedup is asserted by this example.

## Procedure

The quadratic increment lies within both the physical temperature bounds and
a temporary trust region. The initial radius is 0.25 K, the minimum is
1e-6 K, and the maximum is 2 K. The ratio of actual to predicted objective
reduction must reach 0.1. Rejected flow evaluations or insufficient reduction
halve the radius and cause a new quadratic solve. A ratio above 0.75 doubles
the radius when the step reaches 90% of its boundary. The existing independently
checked near-stationarity roundoff safeguard is retained. Final KKT tests use
only the physical bounds.

`strict` uses the original linear and quadratic tolerances. `adaptive` uses
weight-normalized quadratic KKT targets
`max(qp_tolerance, min(0.01, 0.1 * outer_kkt**1.5))` and intermediate original
linear-residual targets `max(inner_tolerance, min(1e-4, 0.1 * qp_target))`.
Both revert to strict targets at outer KKT 1e-4. A loose intermediate step
cannot supply final acceptance without a subsequent strict quadratic check.
The final original-residual, nonlinear KKT, coupled-equation and conservation
criteria remain 1e-10, 1e-8, 1e-12 and 1e-6, respectively.

The first diagnostic uses target 7, 64 slabs over 600 s, a smooth 60 s target
startup, alpha 1e-11, and the three-sweep frozen preconditioner. Each policy has
a cumulative 24-hour computational budget. Run strict accuracy first, then
adaptive accuracy. If both fail through flow evaluation, test the existing
residual-load continuation fallback with adaptive trust regions. Preserve every
outcome. Select the lowest-cost verified policy before comparing four solvers;
do not select different optimizer policies for different solvers.

## Run and recover

Use the checksum-verified inputs of the
[bounded regularization study](../coupled_regularization/README.md). On a CPU
compute allocation:

```sh
uv sync --frozen --extra study --extra plot
uv run --no-sync pytest tests/test_coupled_trust.py
uv run --no-sync python -m deflation_example.coupled_trust_run \
  --screen regularization-summary/summary.json --settings nonlinear-settings.json \
  --optimization NOMINAL_OPTIMIZATION --baseline BASELINE \
  --initial-snapshot INITIAL_SNAPSHOT --initial-assessment ASSESSMENT.json \
  --arm frozen --accuracy strict --output trust-strict-attempt1
```

Repeat with `--accuracy adaptive` and a new output directory. The fallback
adds `--continuation`; its converged flow must satisfy the original equations.
The optional `--complete-sequence` runs all three declared targets. It is used
only after the one-target diagnostic gate. The four arms are `jacobi`, `frozen`,
`recycling`, and `reference`; rank and reference construction come from the
unchanged regularization settings.

Checkpoints follow every completed active-set update and accepted nonlinear
step. Two atomic archive slots retain the current and previous generations;
the manifest identifies the complete archive by checksum. State, residual,
flow, secants, active-set partition, correction count, solver history and
elapsed work are saved together. Sparse factors are reconstructed and charged
on restart. The fixed full-domain reference is saved once per process and
restored by checksum, without repeating its construction.

To recover an interrupted attempt, use the same command and configuration,
add `--resume-from trust-strict-attempt1`, and choose a new `--output`. If the
old process was killed before it wrote its final elapsed time, also supply
`--prior-attempt-seconds SECONDS` using the scheduler's full measured runtime.
This includes work after the last checkpoint. Source or configuration changes,
corrupt archives and exhausted cumulative budgets prevent a restart. Original
attempts remain intact. Resumed intervals are development evidence and are
kept separate from uninterrupted performance repetitions.

## Inspect every outcome

```sh
uv run --no-sync python -m deflation_example.coupled_trust_report \
  --records trust-strict-attempt1/record.json trust-adaptive-attempt1/record.json \
  --plot --output trust-diagnostics
uv run --no-sync python -m deflation_example.coupled_trust_check \
  --record trust-strict-attempt1/record.json --output derivative-check.json
```

The plots show objective and physical KKT progress, temperature-step radii and
cumulative inner work. They retain incomplete and unsuccessful runs. CPU inner
timers contain coarse-space processing as well as CG; they do not provide a
separate coarse-setup measurement. Complete diagnostic cost also includes model
assembly, reference processing, checkpoint I/O and independent verification.
Common calibration is reported separately. Sampled memory is process allocation.
The derivative command checks the retained temperature with three central
difference steps, an independent adjoint assembly and a transpose dot product.
Run it on a compute node; its flow evaluations are separate diagnostic work.

### Iteration-level monitoring

The subsequent `cpu-cg-heartbeat-cooperative-deadline-v1` runtime adds CPU CG
heartbeats approximately every 30 seconds. Each heartbeat identifies the current
kernel, its iteration count, and whether the reported residual comes from the
recurrence or a fresh matrix application. During residual correction, the kernel
right-hand side is the error-equation load. Final acceptance still checks the
original inactive system. Heartbeat I/O is included in the measured solve time.

The deadline is checked before coarse construction and between CG iterations.
A stopped solve verifies its returned candidate; residual correction retains the
best verified state and performs no further error solves after the deadline.
Unfinished inner solves do not update the retained nonlinear temperature.
Native matrix operations, factorizations, and final verification remain
noninterruptible, so this is a cooperative limit with verification time in
addition to the numerical budget. The scheduler's hard limit remains separate.

Nonlinear progress now includes a `retained` entry with the updated objective,
physical KKT components, stationarity numerator and normalization scale. The
existing top-level objective and KKT describe the state before the step.
These changes have a separate numerical source. The strict/adaptive runs at
`bc212fed` keep their original runtime; their records are not attributed to this
later implementation. Source-bound checkpoints refuse cross-version restarts.

Verified single-target diagnostics permit the subsequent four-way, three-target
comparison. Three fresh development repetitions per method precede confirmation.
Five fresh confirmation repetitions additionally require solution agreement and
lower reference median time than the fastest tested alternative across the
retained development repetitions. Interrupted baselines cannot establish speedups.
