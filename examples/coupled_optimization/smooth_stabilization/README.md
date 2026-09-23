# Smooth bounded streamline stabilization

This example evaluates a separate thermal discretization for the coupled
optimization study. Geometry, material coefficients, momentum equations, desired
temperature, physical bounds and final accuracy requirements are retained.
The original `hard_min` rule remains the default for earlier configurations.

The new `smooth_p8` rule uses

\[
\tau=(\tau_d^{-8}+\tau_a^{-8}+\tau_r^{-8})^{-1/8}.
\]

Here the limits are diffusive, advective and the existing nodal row bound.
The row term is omitted when row limiting is disabled. At rest the velocity
dependent inverse limits vanish. For three limits,
\(3^{-1/8}\min_i\tau_i\leq\tau\leq\min_i\tau_i\).
The exponent is fixed at eight. Assembly and velocity derivatives use the same
coefficient in the thermal, storage, source and control terms. Inverse limits
are rescaled before taking powers.

## Verification

Run numerical commands on a compute node with the locked study environment:

```bash
uv sync --locked --group dev --extra study --extra plot --extra coupled-gpu
uv run pytest tests/test_smooth_streamline.py tests/test_consistent_control_jacobian.py tests/test_coupled_source_adjoint.py
```

These tests cover limit crossings and zero velocity, analytic derivatives,
source and storage conservation, recovered-control derivatives, independent
adjoints, and small steady and transient optimization comparisons with SLSQP.
The full CPU regression suite remains a separate gate.

## Checkpoint derivative assessment

Use a checksum-verified snapshot from the numerical-review example. The following
command evaluates the diagnosed node in the saved transformer trajectory:

```bash
uv run python -m deflation_example.coupled_tail_review \
  --snapshot runs/review-snapshot --baseline runs/baseline \
  --slab 47 --mesh-node 1512 --cell 4158 \
  --streamline-rule smooth_p8 --centered --output runs/smooth-derivatives
```

The first 47 time slabs remain fixed. The final 17 coupled time equations retain
their original time steps and objective weights. The source and its derivatives
are recovered with the selected formulation. The output identifies both source
and evaluated rules and retains every finite-difference outcome. Distances to
the original hard limits describe the original switch location; the smooth
coefficient has no switch there. This is a derivative assessment at an initial
state, with no convergence or optimization speedup claim.

## New optimization runs

The `coupled_smooth` Hydra configuration inherits the physical inputs and final
criteria from `coupled_corrected`, and selects the smooth rule explicitly:

```bash
uv run python -m deflation_example.coupled_sequence --config-name coupled_smooth --cfg job
```

Fresh comparisons require new outputs and reference construction. Initializing
from an older temperature field requires reevaluated controls and gradients and
empty secant history. Each measured result must identify its numerical source
and formulation. Earlier timing records retain their original source.

### Matched initialization from the assessed checkpoint

The next comparison uses the complete 64-slab, 600-second target-7 problem,
with its smooth 60-second target startup, physical bounds 337.3--357.3 K and
regularization coefficient `alpha=1e-14`. Both methods receive the same saved
temperature and flow guesses. The optimizer reevaluates all 64 time slabs,
recovers the source and its gradient, and begins with empty secant and recycling
histories and zero damping. The derivative assessment identifies the snapshot,
baseline and smooth formulation. The original snapshot remains unchanged.

From a clean, frozen checkout, run each method in its own process on a compute
node with eight CPU cores and one GPU:

```bash
uv run --extra coupled-gpu python examples/coupled_optimization/smooth_stabilization/run_initialized.py \
  --method reference --source FULL_COMMIT_ID \
  --baseline runs/baseline --snapshot runs/review-snapshot \
  --assessment runs/smooth-derivatives/report.json --output runs/smooth-reference
uv run --extra coupled-gpu python examples/coupled_optimization/smooth_stabilization/run_initialized.py \
  --method jacobi --source FULL_COMMIT_ID \
  --baseline runs/baseline --snapshot runs/review-snapshot \
  --assessment runs/smooth-derivatives/report.json --output runs/smooth-jacobi
```

Replace `FULL_COMMIT_ID` with the full commit hash of that checkout. The driver
requires a clean numerical source and uses the versioned `coupled_smooth`
configuration. Reference rank is 200; Jacobi-CG uses rank zero. Both use the hybrid
backend, final inner relative residual `1e-10`, nonlinear KKT threshold `1e-8`,
equation threshold `1e-12`, and conservation threshold `1e-6`. The nonlinear cap
is 100, with at most 50,000 iterations per inner solve. Configuration, source,
every completed optimizer update and all final outcomes are retained.

## Audit and figures

The initialized comparisons at numerical source
`6ccc2dfda0381370776723703a37af17a1b3c059` retain their original implementation.
The subsequent assessment source adds the following checks and plots:

```bash
uv run python -m deflation_example.coupled_smooth_report \
  --jacobi runs/smooth-jacobi --reference runs/smooth-reference \
  --source 6ccc2dfda0381370776723703a37af17a1b3c059 \
  --output runs/smooth-summary
uv run python -m deflation_example.coupled_figures \
  --baseline runs/baseline --optimization runs/smooth-reference \
  --method reference --target-position 0 --time-indices 5 31 63 \
  --format png --output runs/smooth-fields
```

The summary checks the recorded equations, both adjoints, optimality, all inner
residuals, timing sums, source identities, initialization and saved field
agreement. Its cost plot subtracts the nested coarse interval from the enclosing
inner timer. It retains ratios below one. Each method has one timed optimization
of a complete trajectory from a common assessed initial field. This comparison
does not measure repeated sequences or include the earlier optimization that
provided that initial field. The field figure displays the stored discrete
trajectory; temporal resolution needs a separate assessment.

## Fixed-source forward assessment

Use this assessment source for smooth-model replay. Earlier replay routines
did not propagate `smooth_p8` through thermal assembly and step derivatives.
That omission did not affect the initialized optimization, which uses its own
matched assembly and derivatives. The current replay also inherits the saved
full-residual weighting by default; an explicit override changes the model.

```bash
uv run python -m deflation_example.coupled_newton_replay \
  --baseline runs/baseline --optimization runs/smooth-reference \
  --method reference --target-position 0 --subdivision 1 \
  --consistent-stabilization --forward-policy newton --time-scheme backward_euler \
  --tolerance 1e-12 --cap 30 --threads 8 --output runs/smooth-forward-64
```

Run subdivision 1 first, then 2 and 4 with distinct output directories for a
temporal-resolution study. Every replay starts at the declared initial condition,
copies the saved source unchanged over each original time interval, and retains
unsuccessful steps. Equation tolerance is `1e-12`; mass and energy tolerances
are `1e-6`. Temperature is neither clipped nor reoptimized. Coupled convergence,
temperature-bound satisfaction and refinement differences remain separate
results. The flow discretization enforces continuous pressure tests and global
balance; it does not impose elementwise mass conservation.

Before an application replay, run:

```bash
uv run pytest tests/test_coupled_resolution.py tests/test_coupled_step_spectrum.py \
  tests/test_coupled_newton_replay.py tests/test_coupled_smooth_report.py
```

These tests include matched storage and source assembly, complete current/history
Jacobians, unchanged-control trajectory recovery for both stabilization rules,
saved-formulation inheritance, timer nesting and unsuccessful comparisons.

After a forward run finishes, independently recompute its equations from the
saved temperature, velocity, pressure and unchanged source:

```bash
uv run pytest tests/test_coupled_forward_verify.py
uv run python -m deflation_example.coupled_forward_verify \
  --baseline runs/baseline --optimization runs/smooth-reference \
  --forward runs/smooth-forward-64 --method reference --target-position 0 \
  --threads 8 --output runs/smooth-forward-64-equations.json
```

This check writes a new, checksum-linked record and leaves the forward output
unchanged. It verifies the temporal history, physical time grid, velocity boundary
conditions and pressure gauge as well as the coupled equations. It checks every
saved step, including an unsuccessful final candidate; a verified prefix does
not qualify an incomplete trajectory. This route also supports pure Newton
records, whose iteration histories omit the hybrid solver's separate
`returned_state_verification` label. The existing strict refinement-report
reader expects that hybrid label and must not be used to infer verification of
pure Newton records. Use the independent equation record alongside refinement
differences until a combined reader is available.

## Local propagation diagnostic

The following command evaluates the saved optimized states, rather than
reoptimizing or altering the source:

```bash
uv run python -m deflation_example.coupled_step_spectrum \
  --baseline runs/baseline --optimization runs/smooth-reference \
  --method reference --target-position 0 --slabs 0 15 21 31 47 63 \
  --modes 3 --threads 8 --transport-form advective \
  --output runs/smooth-propagation
```

The current diagnostic includes the saved control and preceding temperature in
the derivatives of the streamline-weighted storage and source. Earlier versions
of this command omitted those inputs, although the Newton forward solver
supplied them. Those earlier diagnostic records do not represent the complete
consistent Jacobian. Each computed mode retains its eigenproblem residual.
The coupled, frozen-velocity thermal and frozen-temperature momentum pencils
describe local linearized propagation. Their eigenvalues do not certify
stability of a product of changing time-step maps or physical time resolution.

The timed interval includes input checking, model assembly, fresh reference
construction, all new optimization work and independent verification. Common
isothermal calibration and process preparation are reported separately. The
earlier optimization that produced the initial temperature is excluded. These
measurements assess optimization from the declared initial state; they do not
represent optimization from zero or a repeated complete-target population.

On interruption, `stage.resume` may resume a checkpoint from this new run only
under its identical configuration. `initial_state_snapshot` remains part of the
configuration; it is superseded by the new run's checkpoint on resume. An old
hard-rule optimization checkpoint cannot resume a smooth-model run.
