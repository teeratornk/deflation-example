# Complete coupled transient optimization

This is a development protocol for testing full-domain reference reuse in
complete nonlinear trajectory optimization. The physical problem and final
accuracy are fixed in [protocol.json](protocol.json). No coupled speedup is
established by the proposed experiments or their unit tests.

## Numerical gate

First finish the existing adaptive trust-region diagnostic. A verified solution
skips the repair stage. Otherwise compare two fresh, 24-hour diagnostics from
the same initial state: direction backtracking alone and direction backtracking
with residual-load continuation. Keep the earlier runs and their original gate.

The `backtrack` policy tries step lengths 1 through 1/64 from a single quadratic
direction. Each trial uses the actual projected increment in the model
reduction and independently evaluates the coupled equations. The acceptance
ratio stays at 0.1. An accepted shortened step sets the next radius to the
smaller of the old radius and twice the actual maximum temperature increment,
bounded below by the existing minimum radius. Seven rejected trials halve the
radius and require another quadratic solve. The final KKT tests use physical
temperature bounds alone. The historical `radius_rebuild` policy is the default.

Use the inputs documented in [the trust-region example](../coupled_trust/README.md):

```sh
uv sync --frozen --extra study --extra plot
uv run --no-sync pytest tests/test_coupled_backtrack.py tests/test_coupled_trial_capture.py tests/test_coupled_backtrack_gate.py
uv run --no-sync python -m deflation_example.coupled_trust_run \
  --screen regularization-summary/summary.json --settings nonlinear-settings.json \
  --optimization NOMINAL_OPTIMIZATION --baseline BASELINE \
  --initial-snapshot INITIAL_SNAPSHOT --initial-assessment ASSESSMENT.json \
  --arm frozen --accuracy adaptive --trial-policy backtrack --capture-trials \
  --output backtrack
```

Repeat with `--continuation --output backtrack-continuation`. Both final solutions
must satisfy the original equations. The continuation removes an artificial
residual load; it does not change viscosity, buoyancy, time steps or boundary data.
All trial capture and continuation work is included in the diagnostic interval.

```sh
uv run --no-sync python -m deflation_example.coupled_backtrack_gate \
  --adaptive-record original-adaptive/record.json \
  --repairs backtrack/record.json backtrack-continuation/record.json \
  --output repair-gate.json
```

If neither policy verifies a complete target, stop the timing campaign and
diagnose the remaining failure. Source-bound checkpoints cannot be used to
restart a historical run under the new algorithm.

## Reproduce a rejected trial

Capture writes the actual trial temperature and the starting temperature,
velocity and pressure before flow evaluation. A completed outcome records the
failed slab, residual history and final failed flow when available. Each archive
has a checksum. An interrupted evaluation keeps its started entry. The archives
are diagnostic inputs, not optimizer checkpoints.

Use the numerical source that generated the capture and run on a compute node:

```sh
uv run --no-sync python -m deflation_example.coupled_trial_replay \
  --run backtrack --position 0 --trial 0 --continuation off --output replay-off.json
uv run --no-sync python -m deflation_example.coupled_trial_replay \
  --run backtrack --position 0 --trial 0 --continuation on --output replay-on.json
```

The replay preserves the captured temperature and starting flow. It does not
reoptimize, clip the temperature, or count its cost as optimization work.

## Reference and complete-comparison gates

After a complete target converges, use its deterministic inactive-system sample
to screen the fixed-reference constructions in the protocol. Compare against
frozen preconditioning and rank-matched recycling with existing coarse vectors
retained. Measure coarse-error reduction and released-node information separately
from complete optimization timing. Matched GPU block processing is conditional
on CPU correctness and independent CPU/GPU operator checks.

Freeze the selected construction and one common optimizer policy before the
three-target development comparisons. Complete accuracy, solution agreement and
at least ten percent median reference savings over the fastest tested alternative
are required before five fresh confirmation repetitions. The confirmation uses
the eight-target sequence in the protocol, including five additional targets.
All methods retain the same targets, bounds, accuracy and warm-start access.

This protocol does not launch those later stages before the numerical gate.
Retain observed timing ranges and all unsuccessful outcomes. Report global
conservation and elementwise flux imbalance separately; weak incompressibility
does not establish exact elementwise conservation. Discrete trajectory
optimality does not establish a physically resolved temperature trajectory.
