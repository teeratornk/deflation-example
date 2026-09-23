# Coupled optimization: checkpoint review

This example separates linear-solver accuracy, nonlinear stationarity, and flow
globalization. It preserves the original comparisons and evaluates saved states
without changing their physical problems or final tolerances. The computations
are diagnostic replays, not complete-optimization timing measurements.

## Input and numerical procedures

Use the configuration, baseline, and checkpoint produced by the
[corrected study](../corrected_study/README.md). A checkpoint contains a complete
temperature trajectory, velocity, pressure, secant pairs, and optimizer history.
The JSON history describes each pre-update state; the checkpoint arrays contain
the retained post-update state. The reviewer independently reevaluates that state.

The original comparisons use numerical source `466d547`. Review source
`fee2cab` corrects the independent adjoint verifier and adds checkpoint diagnostics.
It leaves the optimization Jacobian, quadratic subproblem, reference construction,
and line search unchanged. The verification now solves the independently assembled
source-adjoint equation before the momentum-adjoint equations. Newly evaluated
results require source and momentum residuals and weight-normalized gradient
agreement within `1e-8`; the original inner-residual and KKT thresholds remain
`1e-10` and `1e-8`. Earlier records retain their original verification procedure.

The first review tests exposed a missing source-transpose inverse in the old
independent verifier. The optimizer's Jacobian already applies that inverse.
This finding concerns verification and does not establish a cause of slow outer
convergence. The regression tests include deliberately corrupted source factors.

## Reproduction

Run the following commands on a compute node. Install the locked study dependencies
with `uv sync --locked --extra study --extra plot`. A hybrid replay additionally
requires `--extra coupled-gpu` and a compatible CUDA device. The review preserves
the recorded CPU or hybrid solver policy.

First capture an immutable, checksum-consistent checkpoint. An existing destination
is rejected. If the producer changes the arrays during copying, the command retries;
an unsuccessful capture has no `snapshot.json` and cannot be used by the reviewer.

```bash
uv run python -m deflation_example.coupled_review snapshot \
  --source runs/corrected-reference --output runs/review-snapshot

uv run python -m deflation_example.coupled_review review \
  --snapshot runs/review-snapshot --baseline runs/baseline \
  --mode inspect --output runs/review-inspection

uv run python -m deflation_example.coupled_review review \
  --snapshot runs/review-snapshot --baseline runs/baseline \
  --mode step --secants retained --output runs/review-retained

uv run python -m deflation_example.coupled_review review \
  --snapshot runs/review-snapshot --baseline runs/baseline \
  --mode step --secants plain --output runs/review-plain

uv run python -m deflation_example.coupled_review_derivatives \
  --snapshot runs/review-snapshot --baseline runs/baseline \
  --direction worst_stationarity --output runs/review-derivative-node

uv run python -m deflation_example.coupled_review_derivatives \
  --snapshot runs/review-snapshot --baseline runs/baseline \
  --direction random --output runs/review-derivative-random
```

The step replay computes one quadratic direction and evaluates fixed trial lengths
`1`, `1/2`, `1/4`, and `1/8`. These evaluations describe the direction; they do not
replace the optimizer's line-search policy. Add `--initial` to reconstruct the
original zero-state first subproblem, rather than continue the checkpoint. Add
`--continuation` to repeat failed flow evaluations using the existing residual-load
continuation, which must finish at the unchanged physical momentum equations.
Step replays support Jacobi-CG and full-reference retention. They reject recycling
and sequential-transfer checkpoints because their transfer histories are not
restored by this diagnostic. State inspection remains available for those records.

After a direction has been stored, its failed momentum systems can be captured
without repeating the quadratic solve:

```bash
uv run python -m deflation_example.coupled_review_flow \
  --snapshot runs/review-snapshot --review runs/review-retained \
  --baseline runs/baseline --steps 1 0.5 0.25 \
  --output runs/review-flow
```

The flow diagnostic saves each failed system's forcing, boundary conditions,
initial and returned fields, previous-time velocity, numerical settings, and
history. It checks the assembled Jacobian by central differences and compares
the quadratic Newton residual model against independently reassembled residuals.
Global mass imbalance and elementwise flux defects remain separate quantities.

The following inexpensive check locates proximity to the stabilization branches
using saved velocities. It does not perform an optimization or modify the
stabilization rule:

```bash
uv run python -m deflation_example.coupled_review_branches \
  --snapshot runs/review-snapshot --baseline runs/baseline \
  --output runs/review-branches
```

For a localized derivative, the preceding trajectory can be held fixed while
recomputing the exact remaining time equations. This retains the original weights,
time steps, and predecessor fields. Supply the zero-based slab and mesh identifiers
from the inspection report; the values below illustrate the reviewed checkpoint.

```bash
uv run python -m deflation_example.coupled_tail_review \
  --snapshot runs/review-snapshot --baseline runs/baseline \
  --slab 47 --mesh-node 1512 --cell 4158 --output runs/review-suffix
```

The default magnitudes range from `1e-3` to `1e-8`, with both signs evaluated.
`--steps` declares a different ordered set for a targeted follow-up, whose outputs
must remain separate. Flow failures and exact stabilization switches retain explicit
statuses. A failure in auxiliary momentum diagnostics preserves the original flow
outcome and its saved fields.

## Interpretation and tests

`review.json` contains the recomputed objective, KKT components, weight extrema,
largest stationarity contributors, equation checks, and independent adjoint checks.
Step reports additionally contain both quadratic residual normalizations, actual
rank, coarse conditioning, all inner histories, and every trial outcome. Derivative
reports retain each perturbation, including failed flow evaluations. Relative
directional-derivative error can be large when the true derivative is near zero;
the absolute error and perturbation-size trend must also be considered.

No replay timing includes earlier optimization, and sampled memory is specific to
the replay. A complete speedup requires fresh matched optimization runs, agreement
of their final solutions, identical final accuracy, and the full timing boundary.
Exploratory failures remain part of this example even when a later correction works.

```bash
uv run pytest tests/test_coupled_review.py tests/test_coupled_review_derivatives.py \
  tests/test_coupled_review_flow.py tests/test_coupled_review_branches.py \
  tests/test_coupled_source_adjoint.py tests/test_coupled_tail_review.py
```

The tests cover snapshot races and corruption, state/residual correspondence,
source and momentum adjoints, steady and transient derivatives, failed-flow
capture, and restoration of the instrumented solver. The full CPU suite and GPU
preflights remain required before a revised numerical procedure is used for timing.
# First-update reconstruction

For a checkpoint immediately after the first update, the saved secant displacement
and line-search length can recover its trial direction without another QP solve.
The diagnostic rejects missing displacements, mismatched initial states, and
extrapolations requiring clipping. It records the original QP outcome as such.
It does not restore recycling history or claim a newly verified QP solution.

```bash
python -m deflation_example.coupled_review_direction \
  --snapshot SNAPSHOT --baseline BASELINE --output RECONSTRUCTED
python -m deflation_example.coupled_review_flow \
  --snapshot SNAPSHOT --baseline BASELINE --review RECONSTRUCTED \
  --output FLOW_REPLAY --steps 1 0.5
```
