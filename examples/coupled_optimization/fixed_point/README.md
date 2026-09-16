# Fixed-point iteration for coupled thermal control

This example compares nonlinear solution procedures for the same equations.
The forward screen alternates momentum and thermal solves with an unchanged
saved heat-source control. The momentum screen changes only the nonlinear flow
solver used inside optimization. Exact tangent and adjoint operators remain in
use. Neither comparison changes the physical model or temperature bounds.

Anderson acceleration for Boussinesq fixed-point solves has been studied by
Pollock, Rebholz and Xiao, *Acceleration of nonlinear solvers for natural
convection problems*, https://arxiv.org/abs/2004.06471. The safeguards here define
a separate measured policy; that analysis does not establish convergence for
these particular meshes and parameters.

## Install and verify

Use the study source identifier accompanying the records. Release v0.6.2 and
the earlier timing sources do not contain this addition. From that checkout:

```bash
uv sync --frozen --extra study --extra plot
uv run pytest tests/test_fixed_point.py tests/test_fixed_point_study.py \
  tests/test_coupled_audit_regressions.py
```

Run numerical tests and experiments on a compute node. The forward and momentum
screens use CPU sparse factorizations. GPU time is used for the later hybrid
optimization comparisons; install `--extra coupled-gpu` for those tasks.

## Inputs and execution

`protocol.json` identifies the existing baseline, nested-mesh baseline, optimized
control and saved difficult steps by directory and checksum. Supply their common
parent as `DATA_ROOT`; these files are evidence inputs, not generated meshes or
replacement controls. Input data must be supplied separately when institutional
release restrictions apply. The manufactured verification tests need no archive.

```bash
uv run python examples/coupled_optimization/fixed_point/workflow.py screen \
  --task 0 --data-root "$DATA_ROOT" --output "$STUDY_OUTPUT"
```

Run screen tasks 0–59 in separate processes with identical resources. There are
five saved-step cases, six numerical policies and two equation families. Task 0
is the ordinary step with Newton. A screen repetition measures one selected
step; it is not a complete optimization or a resolution study. Every task has a
separate output and attempt record. Reusing an output directory is prohibited.
The saved steps come from backward Euler and second-order backward
differentiation formula (BDF2) replays. Complete trajectories in this example
use backward Euler.

After all screen tasks have terminated:

```bash
uv run python examples/coupled_optimization/fixed_point/workflow.py select \
  --data-root "$DATA_ROOT" --output "$STUDY_OUTPUT"
```

Selection considers every declared outcome and freezes the best fixed-point
policy separately for each family. Newton remains the comparator. Coverage is
primary; elapsed time on commonly converged cases breaks ties. Selection refuses
incomplete or mixed-source screens. A policy with no converged case cannot
proceed. Caps and failed solves never contribute a completed-solve speedup.

Use the same command with `trajectory --task 0` through `--task 11` for both
meshes, both policies and three repetitions. Each trajectory starts from the
original initial fields and retains immutable step archives. Use the underlying
`fixed_point_study trajectory --resume` command only after interruption, with
the same arguments and source. Resume checks every committed step and input
identity. Numerical failures are terminal. All unfinished process costs remain
unknown until scheduler accounting supplies them; they are never treated as zero.

Run `derivatives --task 0` and `--task 1` before `optimize --task 0` through
`--task 17`. The latter compares three linear solvers, two momentum procedures
and three independent repetitions on the nominal 64-slab optimization. The
derivative gates check the full coupled control derivatives and must both pass.
Every command supports `--dry-run` to inspect its exact arguments.

## Numerical policies and interpretation

For the version-3 momentum follow-up, the same-mesh reference launcher correction
has its own output root. The numerical solver files are unchanged. Audit it with
the explicit amendment shipped with `study/fixed-point-launcher-v3`:

```bash
uv run python examples/coupled_optimization/fixed_point/audit.py \
  --protocol "$MOMENTUM_PROTOCOL" --root "$MOMENTUM_OUTPUT" \
  --reference-repair-root "$CORRECTED_REFERENCE_OUTPUT" \
  --reference-repair-declaration "$REFERENCE_SETUP_AMENDMENT" \
  --output "$COMBINED_AUDIT"
```

The audit retains both original setup errors, checks their hashes, and refuses
to replace any completed or failed numerical solve as a setup error. It checks
the corrected configuration against the original settings and verifies the gate
hashes. Summary groups report corrected sequence intervals and, separately, those
intervals plus the original setup-error work. Missing or failed repetitions have
no complete-comparison median. The four unstarted duplicate setup tasks remain
identified in the amendment.

The relaxed policies use factors 0.25, 0.5 and 1. Anderson histories contain at
most three or five differences, with damping 0.5. Pivoted QR uses relative rank
threshold `1e-12`; difference-coefficient one-norm above 10 triggers the relaxed
step. A freshly evaluated equation residual that increases also rejects the
accelerated proposal. History resets at each physical step or rejected proposal.
Temperature acceleration uses lumped spatial mass weights. Momentum acceleration
uses velocity mass weights and applies the same affine combination to pressure.

Forward acceptance requires momentum, continuity and thermal residuals at most
`1e-12`, and mass/energy defects at most `1e-6`. Optimization retains original
inner residual `1e-10` and KKT threshold `1e-8`. Returned states and residuals
always correspond. An already-converged initial guess needs no nonlinear solve.

Version 2 also uses `1e-12` for the forward momentum subsolve and checks its
initial flow before iterating. In the retained version-1 ordinary-step diagnostic,
Newton reached approximately `8e-13` in one update but continued toward an
unnecessary `1e-13` internal margin. All ten diagnostic updates satisfied the
outer momentum threshold. The correction leaves final coupled acceptance
unchanged. Version-1 smoke outputs remain separate; its unstarted screen tasks
do not contribute numerical outcomes or timing comparisons.

Report step time, complete-trajectory time, complete optimization time, sampled
memory, tracking and temperature-bound violation separately. Preserve the
original spatial/temporal resolution requirements. Faster nonlinear convergence
does not establish resolved physical temperatures or feasibility between stored
time levels. Existing manuscript timings remain attached to their original
implementation.

Generate tables and residual/temperature figures with:

```bash
uv run python examples/coupled_optimization/fixed_point/summarize.py \
  --root "$STUDY_OUTPUT" --output "$STUDY_FIGURES"
```

The summary retains failures and distinguishes local step costs from trajectory
costs. Generate a new output directory for each summary; incomplete records are
visible and do not establish completed-solve speedups.

Audit the declared populations, paired local roots and complete timing figures:

```bash
uv run python examples/coupled_optimization/fixed_point/audit.py \
  --root "$STUDY_OUTPUT" --output "$STUDY_AUDIT"
```

The audit lists missing, failed and verified outcomes separately. It verifies
field checksums for both sides of each local-root comparison and checks the
optimizer timer partition. It also checks the recorded equation residuals,
conservation defects, derivative diagnostics and final KKT components against
their declared thresholds. These checks inspect the independent evaluations
stored by each solver; they do not rerun the differential equations.
Complete-time medians require all three declared
repetitions to meet the criteria. The figures retain each verified repetition
and show the number verified out of the declared population. For interrupted
optimization attempts, the stage interval alone does not supply the all-attempt
cost; the audit leaves that comparison unavailable pending complete accounting.
The forward-process interval includes input preparation and serialization,
whereas the optimization-sequence interval excludes common calibration and
process preparation. They appear in separate figures.
Use `--root-policy anderson3` to compare verified local solutions when Newton
has no verified root. This option changes only the reference for field
differences; it leaves the numerical policy selection and timing populations
unchanged. Every supplied root must meet the original equation checks and use
the same source, control and saved initial fields.

For a complete verified forward trajectory, plot the desired temperature,
computed temperature, unchanged signed heat source and fluid speed:

```bash
uv run python examples/coupled_optimization/fixed_point/trajectory_fields.py \
  --trajectory "$TRAJECTORY_OUTPUT" --data-root "$DATA_ROOT" \
  --output "$FIELD_FIGURES"
```

`--time-indices` selects declared stored time levels for the figure. The default
shows the first, middle and last levels. Temperatures use kelvin, source density
uses megawatts per cubic metre, and velocity uses metres per second. Radial and
axial plotting scales are unequal. Each field archive is checked before plotting;
incomplete trajectories produce no complete-trajectory field figure. The residual
plots use crosses at the last recorded iterate of failed local solves. When the
solver returns an earlier, better state, a separate hollow diamond shows its
verified residual at termination. The iteration curve excludes that return
verification, so restoring the best state does not appear as a new nonlinear
update. Detailed termination labels and returned-state residuals remain in
`outcomes.csv`.

The following diagnostic compares a state solve and a correction solve for the
same frozen Oseen matrix at the ordinary screening input. It then applies five
linear residual corrections with that factorization and evaluates the original
nonlinear momentum equations after every update:

```bash
uv run python examples/coupled_optimization/fixed_point/momentum_precision.py \
  --data-root "$DATA_ROOT" --output "$PRECISION_DIAGNOSTIC"
```

The diagnostic does not change the screening policies or their final accuracy
criteria. It also compares at most ten successive Oseen updates using the two
algebraically equivalent maps, starting from the same initial flow. A small
frozen-matrix residual alone does not establish a nonlinear momentum solution.

For the separate residual-equation momentum study, run `audit.py` with
`--protocol` pointing to the exact version-3 protocol and `--root` pointing to
its own outputs. This audits thirty momentum cases, two derivative gates and
eighteen optimization tasks. The version-2 forward study retains its own source
and records; the audit refuses mixed numerical sources in either population.

The version-2 derivative and optimizer follow-ups were replaced before execution.
Their replacement uses the residual-equation momentum map in version 3. All
sixty version-2 screen outcomes remain available, including the precision-floor
failures. To identify the unstarted follow-ups explicitly in the version-2 audit:

```bash
uv run python examples/coupled_optimization/fixed_point/audit.py \
  --root "$V2_OUTPUT" --output "$V2_AUDIT" \
  --disposition examples/coupled_optimization/fixed_point/followup-disposition.json
```

The declaration is bound to the original protocol and numerical source. It
retains the unstarted comparisons in the audit as `withdrawn_before_execution`,
without displaying them as measured comparisons. It cannot exclude an existing
record. Audit the replacement population separately with its version-3 protocol;
the two sources do not share a timing population.

## Agreement across complete trajectories

Compare the temperature and velocity fields at every stored time level after
the forward trajectories finish. The baseline and candidates must use the same
saved source, mesh, time levels, physical inputs, accuracy criteria and numerical
source. Each trajectory must meet its reported equation and conservation tests.

```bash
uv run python examples/coupled_optimization/fixed_point/trajectory_agreement.py \
  --baseline "$V2_OUTPUT/trajectory/original/newton/rep-0" \
  --candidate "$V2_OUTPUT/trajectory/original/newton/rep-1" \
    "$V2_OUTPUT/trajectory/original/newton/rep-2" \
    "$V2_OUTPUT/trajectory/original/anderson5/rep-0" \
    "$V2_OUTPUT/trajectory/original/anderson5/rep-1" \
    "$V2_OUTPUT/trajectory/original/anderson5/rep-2" \
  --output "$TRAJECTORY_AGREEMENT"
```

The example verifies file hashes and computes maximum nodal temperature
differences in kelvin and maximum velocity-vector differences in metres per
second. The JSON output identifies the time and array index of each maximum.
It compares stored fields without re-solving the equations. Agreement between
solvers at the same discretization establishes neither uniqueness nor spatial
or temporal resolution. Failed and incomplete trajectories remain in the main
outcome audit and cannot enter this complete-trajectory comparison.
