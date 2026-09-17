# Fully coupled validation

This development example implements the full-validation protocol in
[`protocol.json`](protocol.json). It does not change the prescribed-flow
manuscript or the numerical sources used for earlier timing records.

The subsequent [completion protocol](completion_protocol.json) adds bounded
Newton diagnostics, local time comparisons, checkpointed forward runs and an
explicit resolution-based design margin. Its [reproduction instructions](completion.md)
use a pinned numerical source. These steps remain separate from the earlier
timing populations.

## Formulation audit

The experimental residual-weighted thermal assembly now includes the complete
cell-interior advective residual. Its test weight is cell constant and uses the
velocity at the element centroid. Its trial residual uses the integrated
quadratic velocity, consistent with the Galerkin transport term. For a
piecewise-linear temperature and cell-constant conductivity in cylindrical
coordinates, the strong diffusion includes

\[
\nabla\cdot(K\nabla T)=\frac{(K\nabla T)_r}{r}.
\]

That term survives even though the Cartesian second derivatives vanish.
Multiplication by the axisymmetric volume weight cancels the denominator.
The implementation includes its velocity derivative through the streamline
test weight. Source, background load, and transient storage use the same
weight. The non-residual-weighted default remains unchanged. Weighted skew
transport is rejected until its additional reaction residual is implemented.

Residual-based stabilization requires weighting the complete strong residual;
see the formulation in [On the convergence order of the finite element error
in the kinetic energy for high Reynolds number incompressible flows](https://doi.org/10.1016/j.cma.2021.114032).
The cylindrical expression above follows directly from the divergence formula
and is also tested by independent element quadrature.

The matched coarse construction preserves the corrected stiffness and uses
lumped storage and source for the reference approximation. This does not change
the operator used for final residual checks. Earlier consistent-assembly
timings belong to their earlier source versions; they do not measure this
correction.

Run the focused checks from the repository root on a compute node:

```sh
uv sync --frozen --group dev
uv run pytest tests/test_axisymmetric_residual_weighting.py \
  tests/test_thermal_stabilization_consistency.py \
  tests/test_consistent_control_jacobian.py tests/test_coupled_step_spectrum.py \
  tests/test_matched_reference_stabilization.py
```

The element tests exercise radial temperature gradients, anisotropic
conductivity, constant and quadratic velocities, and planar and axisymmetric
volume weights. The coupled tests check control tangents, transpose identities,
current and previous time blocks, conservation, and coarse-space construction.
Passing these tests verifies the stated discrete formulas. It does not establish
physical resolution or a speedup for the application.

The forward runner uses fixed-scaled Newton and complete-field Anderson from
the retained fields if Newton stops before meeting the original equations:

```sh
uv run python -m deflation_example.coupled_newton_replay \
  --baseline runs/baseline --optimization runs/optimization \
  --method reference --target-position 0 --subdivision 1 \
  --consistent-stabilization --forward-policy newton_anderson \
  --line-search fixed_scaled --cap 100 --tolerance 1e-12 \
  --output runs/full-validation-64
```

Repeat with subdivisions 2 and 4 and distinct output directories. The saved
optimization must contain 64 slabs for these names to correspond to 64, 128
and 256 assessment steps. Each run begins at the physical initial fields.
Newton uses at most 100 iterations per step. The fallback uses depth five,
relaxation 0.5, at most 300 coupling iterations and 100 momentum iterations.
Both attempts retain their histories and elapsed times. The final check uses
the equations at the returned fields, irrespective of a solver's status label.

For the declared zero initial dimensionless temperature, compare all refined
time levels with:

```sh
uv run python -m deflation_example.coupled_time_resolution_report \
  --replays runs/full-validation-64 runs/full-validation-128 runs/full-validation-256 \
  --temperature-scale 20 --initial-value 0 --plot \
  --output runs/full-validation-temporal-summary
```

Use the actual physical temperature scale and initial field of the chosen
baseline. The example above requires the declared uniform initial field; a
nonuniform initial field requires extending the comparison input explicitly.
Missing runs and unsuccessful time steps prevent a passing assessment. A
single passing refinement comparison is insufficient.

## Resolution and comparison gates

Each assessment applies the saved source without reoptimization or clipping.
Two successive spatial comparisons and two successive temporal comparisons
must pass the full-horizon temperature and tracking criteria. A product of
sampled local eigenvalue moduli is not a bound on the time-varying propagator
and cannot replace this check. If two comparisons fail to improve, inspect the
discretization and nonlinear solves before further refinement.

If the old optimized source violates the resolved temperature bound, optimize
on the verified discretization and assess the new frozen source independently.
Any changed source gets its own identity and records. Final performance
comparisons begin only after the formulation, derivative and resolution gates.
All outcomes remain available, including iteration caps and unsuccessful
attempts. A positive result requires lower measured complete cost at the same
accuracy; it is not an inclusion criterion.
