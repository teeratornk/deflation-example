# Complete space–time speedup ablations

This study measures the cost of reference-space reuse for the prescribed-flow
problem. It is separate from the ongoing fully coupled optimization study.
The reference implementation preserves the coupled thermal time discretization,
consistent source and storage actions, and the fixed computed velocity.

The geometry, 64 time slabs, 600-second horizon, 60-second target startup,
regularization `1e-11`, temperature bounds, targets `[7, 15, 14]`, warm starts and
final accuracy remain unchanged. Each sequence must meet the original linear
residual threshold `1e-10`, weighted KKT threshold `1e-8`, thermal-equation
threshold `1e-12` and declared conservation threshold `1e-6`.

The 22 configurations are declared before execution:

- Rank-zero Jacobi and velocity-frozen preconditioning.
- Full reference restriction at ranks 4, 8, 16 and 32, with either lowest or
  alternating low/high Ritz selection from 96 seeded energy-Krylov steps.
- Sequential zero-extension transfer of each of those eight initial spaces.
- Adaptive recycling at the same four ranks, retaining existing coarse vectors
  and a window of new directions. Selection uses the current frozen inverse in
  the operator energy metric when that inverse is attached.

The reference construction is independent of target loads and active-set history.
Each worker reconstructs it once and includes its cost. Sequential transfer
removes entries as constraints activate; it learns no directions. Recycling
starts without history and learns from its own complete optimization sequence.
Requested and deployed ranks, numerical rank loss and fallbacks remain in the
records. The transfer ablation compares complete optimizers; a separate matched
replay is required to isolate the effect on identical inactive systems.

Each configuration has three fresh complete repetitions. The task order rotates
between repetitions, and the execution limit is two concurrent sequences.
All workers use the same CPU, thread count and numerical source. The current
sparse factorization path executes on CPUs; these runs make no GPU-speedup claim.
The measured interval includes construction, every active-set solve, restriction,
coarse/recycling processing, independent verification and cleanup. Common
calibration and process preparation are recorded separately and in an inclusive
total. Sampled process RSS measures memory using the same boundary for every arm.

## Reproduction

First obtain the inputs described in [the space–time example](README.md),
including the verified nominal optimization and the complete regularization
screen. Run on a compute node:

```sh
uv sync --frozen --extra study --extra plot
uv run --no-sync python -m deflation_example.linear_spacetime_ablation settings \
  --screen regularization-summary/summary.json --output ablation-settings.json
uv run --no-sync python -m deflation_example.linear_spacetime_ablation run \
  --screen regularization-summary/summary.json --settings ablation-settings.json \
  --optimization NOMINAL_OPTIMIZATION --baseline BASELINE \
  --initial-snapshot INITIAL_SNAPSHOT --initial-assessment ASSESSMENT.json \
  --task 0 --output ablation-00
```

Repeat the worker command with task indices 0–65 and distinct output directories.
The frozen settings map each index to a method and repetition. Output directories
are never reused. Preserve nonzero worker exits and scheduler timeouts alongside
their numerical records. For a completed population:

```sh
uv run --no-sync python -m deflation_example.linear_spacetime_ablation_report \
  --screen regularization-summary/summary.json --settings ablation-settings.json \
  --records ablation-*/record.json --output ablation-summary --plot
```

The summary checks timing sums, settings, numerical source, actual prescribed
velocity, final accuracy, inner histories and saved-state/objective agreement.
Missing or failed repetitions have no complete-population median or speedup.
Every full-reference setting is compared with fresh Jacobi, frozen and
rank-matched recycling controls. The time–rank and sampled-memory plot retains
the observed repetition ranges and identifies incomplete configurations. These
are development comparisons; favorable settings need new confirmation runs.

The tests exercise both selection policies, restriction and release, complete
small trajectories, timing sums, missing comparators, mismatched sources and
accuracy, and disagreement of saved solutions:

```sh
uv run --no-sync pytest tests/test_linear_spacetime_ablation.py \
  tests/test_coupled_krylov_reference.py tests/test_coupled_nominal_krylov.py
```
