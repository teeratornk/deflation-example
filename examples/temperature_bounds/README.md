# Optimized temperatures under different upper bounds

This example fixes the bore-in-block geometry, material coefficients,
prescribed velocity, desired temperature and regularization. It changes only
the upper bound within each steady or transient comparison. The primary
16-target timing sequences retain their original fixed bounds and records.

The declared comparison uses Bore 1 (3958 spatial state degrees of freedom),
target 8 of the original 16-target definition, and bounds 0.1, 0.2 and 0.3.
The transient problem has four backward-Euler steps over horizon 0.1;
all four states satisfy the same bound. Temperatures, source control and
time use the dimensionless bore-in-block conventions. The initial and
outer-boundary temperatures are zero, and regularization is `alpha=1e-6`.

Each form constructs one rank-100 coarse-mesh reference in Jacobi-scaled
coordinates. The same reference is restricted for every inactive solve at
all three bounds. CPU reference deflation uses `residual_policy=refine`.
Each optimization starts independently with all state variables active;
inner solves use the current state as their initial guess. Sparse-direct
PDAS provides a separate linear-solver check with the same initial mask.
This comparison verifies fields and supplies no solver timing claim.

## Reproduce the solutions

From a checkout containing this example:

```bash
uv sync --locked --extra plot
uv run --locked --extra plot python -m deflation_example.temperature_bounds \
  output=runs/temperature-bounds
uv run --locked --extra plot python -m deflation_example.temperature_bounds \
  action=plot data=runs/temperature-bounds output=runs/temperature-bounds-figure
uv run --locked --extra plot pytest tests/test_temperature_bounds.py
```

The configuration is `src/deflation_example/conf/temperature_bounds.yaml`.
The solver source and numerical-library versions are recorded in `report.json`.
Both commands require new output directories. A small verification run uses
`level=0 rank=20 slabs=2`; it has a separate scope from the published illustration.

The original inactive-system relative residual must be at most `1e-10`,
and all five KKT components must meet `1e-8`. Primal feasibility is absolute;
the other components use the existing load normalization. The reference
state must agree with direct PDAS to relative error `1e-7`. Independent
forward substitution of the recovered control must reproduce the optimized
state to relative error `1e-7`. Every outcome and inner termination is
retained. The plotting command requires all declared cases to meet these
checks and re-evaluates optimality, control recovery and the numerical
summaries from the stored fields. It checks input and output hashes.

## Read the figure

The figure contains six optimized temperature fields: three steady solutions
and the final state of each four-slab optimization. Each panel intersects the
tetrahedral mesh at `x3/L=0.5` and evaluates the piecewise-linear temperature
on that plane. Every optimized-temperature panel uses the same color scale.
White contours indicate level 0.5 of the interpolated nodal active indicator.
They describe the displayed section; feasibility checks cover every state
degree of freedom and every positive time level. Saved temperatures remain
unchanged; the plot applies no clipping to enforce the bound.

The files retain the desired trajectory, optimized temperatures, recovered
source, adjoint, multiplier, active set and all solver histories. Weighted
tracking errors and objectives use the same spatial and temporal weights as
the optimization. These discrete comparisons do not establish time- or
mesh-converged physical predictions.
