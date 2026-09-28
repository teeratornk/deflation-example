# Projected quadratic steps and device profiling

This example keeps the coupled equations, physical bounds and final accuracy
requirements fixed. It separates three tests: verification of the temperature
used to construct the reference, performance of individual operator actions,
and convergence of complete nonlinear optimization. An operator timing or a
verified intermediate quadratic does not establish nonlinear convergence.

## Verification and profiling

The following small verification is self-contained and runs from an installed
wheel without study data. It compares projected quadratic steps with the
existing PDAS procedure and checks the coupled equations and final optimality:

```sh
uv run --locked --extra study python -m deflation_example.coupled_projected_example \
  --device cpu --output OUTPUT/small-verification
```

The annular test mesh and nondimensional coefficients provide an implementation
check. They are separate from the body-fitted application comparisons.

Install the locked environment on a compute node:

```sh
uv sync --locked --extra study --extra coupled-gpu
uv run --no-sync pytest tests/test_box_projected_cg.py tests/test_coupled_trust.py \
  tests/test_coupled_reference_verification.py tests/test_coupled_cuda_frozen.py \
  tests/test_coupled_hybrid_solver.py -q
```

Use the optimization record's declared initial snapshot and assessment to
compare reference initialization policies. `OPTIMIZATION` is the directory of
that record; `BASELINE` contains the matching physical baseline. These inputs
must be obtained from the study data, not synthesized from later iterates.

```sh
uv run --no-sync python -m deflation_example.coupled_reference_verification \
  --optimization OPTIMIZATION --baseline BASELINE --output OUTPUT/initial-checks
uv run --no-sync python -m deflation_example.coupled_device_profile \
  --trace OPTIMIZATION/inactive-trace-00 --baseline BASELINE --quadratic 2 \
  --output OUTPUT/device-profile
```

The verification output preserves every time slab and any failed check. It
includes absolute thermal defects and normalization terms. The profiler uses
identical vectors, two warmups and five repetitions. Device-event and wall-clock
times describe the same interval and must not be added. The grouped frozen
preconditioner uses the original independently factored slab blocks and their
permutations. Coupled forward and adjoint temporal dependencies are unchanged.

## Fixed quadratic comparison

For the updated reference initialization, append these arguments to the fixed
quadratic commands in [CUDA quadratic assessment](cuda_quadratic.md):

```sh
--reference-initial optimizer --initial-snapshot SNAPSHOT \
--initial-assessment ASSESSMENT
```

The default `clipped_zero` remains available for reproducing the earlier
attempts. A failure stops reference construction and writes
`reference-equations.json`. No threshold is relaxed automatically.

`--device cpu`, `--device cuda`, and `--device hybrid` select execution.
For CUDA, `--frozen-layout block_diagonal` groups independent preconditioner
solves; the default `serial` preserves the preceding procedure. The hybrid
executes sparse products and solves on CPU, retains coarse-space operations on
GPU, and charges all transfers. This differs from the earlier hybrid policy
that also offloaded sparse block products.

## Complete nonlinear test

The projected procedure is opt-in:

```sh
uv run --no-sync python -m deflation_example.coupled_trust_run \
  --screen SCREEN.json --settings SETTINGS.json --optimization OPTIMIZATION \
  --baseline BASELINE --initial-snapshot SNAPSHOT --initial-assessment ASSESSMENT \
  --arm frozen --qp-solver projected --accuracy adaptive_projected \
  --trial-policy backtrack --device cpu --budget-seconds 28800 \
  --output OUTPUT/nonlinear-frozen
```

The screen and settings identify the predeclared physical case. This command
tests the first target; `--complete-sequence` includes all declared targets.
Reference comparisons use `--arm reference --rank 16` or `--rank 32` with the
same remaining arguments. The reference is constructed from the initial
temperature supplied to every optimizer, before solving the sequence.

The initial intermediate quadratic target and inner relative target can both
be 0.01. The CG trigger is ten times tighter. Targets decrease with nonlinear
optimality; the strict switch remains at a normalized KKT error of 0.0001.
Final original residual and nonlinear KKT requirements remain 1e-10 and 1e-8.
Every returned quadratic state receives a fresh optimality check. Nonlinear
steps still pass the existing actual-versus-predicted decrease test.

Checkpoints bind the source, configuration, procedure and retained state.
`--resume-from` resumes only a matching computation. The cost of restored
factors and preceding attempts remains identifiable. A saved state from an
older procedure may be used only as a declared fresh initial condition, with
the same access given to competing methods.

Keep every capped or failed attempt. Compare complete time only when both
methods meet all final criteria; report intermediate progress separately.
