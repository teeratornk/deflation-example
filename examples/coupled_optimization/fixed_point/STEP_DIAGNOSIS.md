# Saved-step conservation and convergence checks

This example investigates the first failed finer-mesh Anderson trajectory.
The original control, physical problem, residual targets and saved runs remain
unchanged. It is a development diagnostic, separate from manuscript evidence.

Preselected systems: the initial step, the last verified step (271.875 s), and
the failed step (276.5625 s), from finer-mesh Anderson repetition 0. Inspect all
three. Preserve every diagnostic and subsequent solver outcome.

```bash
uv sync --frozen --extra study --extra plot
uv run python examples/coupled_optimization/fixed_point/step_diagnostics.py \
  --baseline /path/to/original-baseline \
  --fine-baseline /path/to/refined-baseline \
  --optimization /path/to/saved-optimization \
  --trajectory /path/to/trajectory/fine/anderson5/rep-0 \
  --slab 58 --newton-check --output /path/to/new-diagnostic
```

Repeat with `--slab 0` and `--slab 57`; the expensive Newton check is optional
for those control steps. Input hashes, archive dimensions, time ordering and
the verified preceding trajectory are checked. Outputs include equation
residuals, elementwise cylindrical divergence, integrated cell mass flux,
sampled cell Péclet numbers, field plots, and an optional original-Jacobian
finite-difference/Newton-direction check.

The sum of absolute cell fluxes divided by boundary throughput measures local
imbalance relative to throughflow. It is neither a percentage of mass lost nor
a reliable relative scale when throughput vanishes; dimensional fluxes and
divergence are also reported. Global balance can hide cancellation. Field
roughness, pressure gradients and positive thermal off-diagonal entries alone
do not prove spurious oscillations. Signed sources can produce interior extrema.

After diagnosis, first test a correction of the *same discrete equations* on
the saved failed state under the unchanged 1e-12 equation and 1e-6 balance
criteria. A local success does not certify a complete trajectory, physical
resolution, or optimization acceleration. Changes to transport or time steps
require separately identified comparisons and cannot inherit frozen timings.
