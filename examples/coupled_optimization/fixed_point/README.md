# Residual-equation momentum iteration

## Same-mesh reference setup correction

The numerical solver files are unchanged from `16dabaf`. The earlier launcher
passed the optimization baseline as a separate coarse baseline. That option
requires an immediate nested parent mesh, so the reference runs stopped before
optimization. The corrected launcher sets `reference_baseline_directory=null`
and constructs the reference on the already loaded optimization mesh. Rank,
target, time levels, precision, stopping tests and solver settings are unchanged.
The two original setup failures remain in their original directories. They are
execution errors and do not establish numerical convergence or a solver ranking.

Reproduce the error and check the corrected rank-200 construction on a compute
node before starting the corrected reference runs:

```bash
uv run python examples/coupled_optimization/fixed_point/reference_setup_check.py \
  --data-root "$DATA_ROOT" --output "$SETUP_CHECK"
```

Matching completed selection and derivative checks can be reused because the
numerical source is unchanged. Run the corrected reference tasks (2, 3, 8, 9,
14, 15) into a new output root, keeping the original records:

```bash
uv run python examples/coupled_optimization/fixed_point/workflow.py optimize \
  --task 2 --data-root "$DATA_ROOT" --gate-root "$ORIGINAL_STUDY_OUTPUT" \
  --output "$CORRECTED_REFERENCE_OUTPUT"
```

Each attempt records its launcher hash and the exact gate-record hashes. The
unchanged Jacobi and recycling computations continue in the original root.
Report this configuration correction explicitly; preserve setup-error costs
alongside the corrected complete computations. A setup check supplies no
optimization timing, and no failed attempt enters a completed-solve speedup.

This study changes only the nonlinear momentum solver used inside coupled
thermal optimization. The physical model, controls, bounds, exact tangent and
adjoint operators remain unchanged. The forward screen and twelve fixed-control
trajectories retain source `330766b` and its separate version-2 protocol.

Anderson acceleration for Boussinesq fixed-point solves has been studied by
Pollock, Rebholz and Xiao, *Acceleration of nonlinear solvers for natural
convection problems*, https://arxiv.org/abs/2004.06471. Their analysis does not
establish convergence for these particular meshes, parameters and safeguards.

## Installation and verification

Use the study source accompanying the records. Release v0.6.2 and earlier timing
sources do not contain this addition. Run tests and experiments on compute nodes.

```bash
uv sync --frozen --extra study --extra plot
uv run pytest tests/test_fixed_point.py tests/test_fixed_point_study.py \
  tests/test_coupled_audit_regressions.py
```

The screen uses CPU sparse factorizations. Install `--extra coupled-gpu` for the
subsequent hybrid optimization comparison.

## Inputs and commands

`protocol.json` identifies the saved baselines, control and difficult steps.
Supply their common parent as `DATA_ROOT`. Data must be provided separately when
institutional release restrictions apply. Manufactured tests need no archive.

```bash
uv run python examples/coupled_optimization/fixed_point/workflow.py screen \
  --task 0 --data-root "$DATA_ROOT" --output "$STUDY_OUTPUT"
```

Run tasks 0–29 in separate processes with identical resources. These cover five
saved inputs and six momentum policies. Task 0 is ordinary-step Newton. Every
task has a distinct output and attempt record; existing outputs cannot be reused.
The saved steps use backward Euler or second-order backward differentiation
(BDF2). Subsequent optimization uses backward Euler.

After all thirty tasks terminate, freeze selection:

```bash
uv run python examples/coupled_optimization/fixed_point/workflow.py select \
  --data-root "$DATA_ROOT" --output "$STUDY_OUTPUT"
```

Selection ranks fixed-point policies by converged case count, then median cost
on commonly converged cases. Newton remains the comparator. Missing or
mixed-source screens cannot be selected. Caps and failures remain visible.

Run `derivatives --task 0` and `--task 1`, then `optimize --task 0` through
`--task 17`. Optimization compares Jacobi, reference and recycling solvers, two
momentum procedures and three repetitions on the nominal 64-slab problem.
Both derivative gates must pass. Every task command supports `--dry-run`.

Use source `330766b` for the twelve forward trajectory tasks. This version refuses
that phase. Original version-2 momentum records remain available and are never
replaced by version-3 results.

## Numerical definition and interpretation

Relaxation factors are 0.25, 0.5 and 1. Anderson uses depth three or five and
damping 0.5. Pivoted QR has relative rank threshold `1e-12`. A coefficient
one-norm above 10 or an increasing original residual rejects the accelerated
proposal. Velocity mass weights determine the coefficients; the same affine
combination acts on pressure. Boundary values and the pressure gauge are retained.

The screen requires original momentum and continuity residuals at most `1e-12`.
Optimization retains inner original residual `1e-10` and KKT threshold `1e-8`.
Returned states and reported residuals correspond. An already-converged initial
guess bypasses the nonlinear solve.

The version-2 ordinary momentum screen reached a residual floor above `1e-12`
with state-form Oseen solves. A matched diagnostic compared this route with the
algebraically equivalent equation `A(v) delta = f - A(v) x`. The residual form
met the same nonlinear criterion in four updates. Version 3 uses this equation
to construct every Picard/Anderson map. Final accuracy, damping, history safeguards
and iteration caps remain unchanged. The full screen and derivative gates must
finish before the procedure supports complete optimization comparisons.

Report local momentum time and complete optimization time separately. Preserve
tracking, bound violations, sampled memory and every termination status. Faster
nonlinear convergence does not establish spatially or temporally resolved
temperatures. Existing manuscript timings retain their original implementations.

Generate local residual figures in a fresh output directory:

```bash
uv run python examples/coupled_optimization/fixed_point/summarize.py \
  --root "$STUDY_OUTPUT" --output "$STUDY_FIGURES"
```
