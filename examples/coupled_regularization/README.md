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
