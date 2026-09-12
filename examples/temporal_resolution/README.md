# Temporal resolution of prescribed-flow thermal optimization

This example tests time-step sensitivity on the existing transformer mesh.
It preserves prescribed velocity, material properties, distributed-source
control, the initial temperature, the upper bound, and the 600-second horizon
of the manuscript's prescribed-flow model. It is separate from the nonlinear
flow–temperature optimization examples and supplies no solver timing comparison.

Before optimization, `prepare` fixes nominal target 7 of 16 and selects a second
target with the largest weighted squared desired-temperature exceedance above
the bound, excluding target 7. It evaluates this criterion on 64 time slabs;
the selected control or solver performance never enters selection. Every target
uses 4, 8, 16, 32 and 64 slabs. The physical target function is unchanged.

Each optimization uses rank-100 reference deflation with guarded residual
correction. The independently recomputed inner residual must meet `1e-10`;
all five KKT components must meet `1e-8`. Sparse-direct PDAS also checks each
four-slab solution. These computations have their own source identifier and
do not replace the frozen primary or finer-transformer timings.

## Reproduce on a compute node

Use the `prescribed-temporal-source-v1` source tag and an unchanged checkout.

```bash
git clone https://github.com/teeratornk/deflation-example.git
cd deflation-example
git checkout prescribed-temporal-source-v1
uv sync --frozen --extra study
uv run --frozen pytest tests/test_temporal_resolution.py
uv run --frozen python -m deflation_example.temporal_resolution action=prepare
uv run --frozen python -m deflation_example.temporal_resolution action=run case=0
```

Run indices 0–9, each once in a fresh process. `design.json` gives their exact
target and time grid. Cases can run independently after preparation. The driver
checks the source, protocol and thermal-input hashes. Existing outputs are
preserved; use a new `root` for another experiment.

```bash
uv run --frozen python -m deflation_example.temporal_resolution action=report
```

The summary retains all ten cases, including missing and unsuccessful solves.
Its default output is `runs/temporal-resolution-summary/summary.json`.

## Interpret the measurements

Optimized states use piecewise-linear temporal reconstruction through the initial
value and the optimized time nodes. Controls remain constant on each original
interval. Comparisons use the 64-slab optimization as a numerical reference,
axisymmetric lumped-mass weights, and the same physical objective quadrature.
Temperature differences are in kelvin; control differences are relative weighted
root-mean-square differences. The finest optimization is a comparison reference,
not an exact solution.

Every saved source is also applied unchanged in forward thermal solves on 128
and 256 steps. Both grids align with all original source jumps. These replays
preserve initial conditions, interface sources and boundary conditions and check
every new step's linear residual. They measure bound violation at all replay
levels without clipping the temperature. The 128-to-256 change estimates temporal
sensitivity; it is not a rigorous error bound or evidence of spatial resolution.
The predeclared temperature-change scale is 0.05 K. A violation remains reported
alongside this change, including its location and time, regardless of its sign
relative to that scale.

The full fields, controls, verification records and source identifiers are saved
under each case directory. Smaller timesteps can change the discrete optimizer
and the fixed-source response; these are reported separately.

## Separate energy-consistent transport pilot

The original `prescribed-temporal-source-v1` study uses advective thermal
transport. Its finer fixed-source replays reveal growing modes in the discrete
thermal operator. Small equation residuals establish algebraic accuracy and do
not establish temporal resolution. Both the original attempts and this diagnosis
are retained.

The `prescribed-temporal-skew-v1` pilot keeps the mesh, prescribed quadratic
velocity, properties, targets, bound and 600-second horizon. It adds
`0.5*c*div(v)*T` to the cellwise thermal transport term. In axisymmetric
coordinates, `div(v) = d_r(v_r) + v_r/r + d_z(v_z)`. Exact polynomial quadrature
gives the corresponding discrete energy identity: transport contributes half
the boundary flux and any interelement capacity-flux jumps. With continuous
normal capacity flux and homogeneous inflow temperature, its energy contribution
is the outflow term. This change corrects the energy term caused by discrete
velocity divergence; it does not project the supplied velocity to an exactly
divergence-free field.

The pilot changes the thermal discretization. Its results do not replace or
relabel the primary optimization timing records. The default public model and
all frozen measured sources keep their original transport form.

```bash
git checkout prescribed-temporal-skew-v1
uv sync --frozen --extra study
uv run --frozen pytest tests/test_thermal_transport.py tests/test_temporal_resolution.py
uv run --frozen python -m deflation_example.temporal_resolution action=stability output=runs/transport-stability
uv run --frozen python -m deflation_example.temporal_resolution action=prepare transport_form=skew root=runs/temporal-skew
uv run --frozen python -m deflation_example.temporal_resolution action=run transport_form=skew root=runs/temporal-skew case=0
```

Run cases 0–9 separately and summarize with `action=report transport_form=skew
root=runs/temporal-skew output=runs/temporal-skew-summary`. The paired stability
diagnostic reports both formulations at each declared time step, the computed
amplification factor, the spatial eigenpair residual and diffusion, transport
and stabilization energy contributions. A verified growing eigenmode establishes
instability. Finding only decaying modes in the sparse calculation is not an
exhaustive stability certificate.

The corrected pilot uses the original target-selection rule before optimizing.
The original and corrected designs must select the same nominal and demanding
targets. Every replay keeps its optimized source unchanged and records the
temperature-change scale separately from the equation and KKT checks. Neither
pilot measures fully coupled flow–temperature optimization or a deflation speedup.

## Finer fixed-source assessment

The separate `prescribed-replay-source-v1` driver evaluates both preselected
targets at the coarsest and finest optimization grids (cases 0, 4, 5 and 9).
It runs 256, 512, 1024, 2048 and 4096 forward steps for each saved source.
Selection uses the declared target and time grids, not the outcomes. The source
remains piecewise constant on its original intervals. Input, numerical-source
and optimized-field checksums prevent changes during transfer.

```bash
git checkout prescribed-replay-source-v1
uv sync --frozen --extra study
uv run --frozen pytest tests/test_temporal_resolution.py tests/test_thermal_transport.py
uv run --frozen python -m deflation_example.temporal_replay \
  --root runs/temporal-skew --case 0 --output runs/replay-case-00
```

Run all four selected indices with separate outputs. The root must contain
the completed `prescribed-temporal-skew-v1` optimizations. The driver retains
unverified optimizations as such instead of interpreting their controls.
It keeps every forward grid and all bound violations. Both final consecutive
maximum temperature changes must be at most 0.05 K, and every equation residual
must meet `1e-10`, to meet the declared time-step sensitivity criterion. A
larger change at the final grid is recorded as `resolution_cap`. Meeting this
criterion does not imply temperature-bound satisfaction or spatial resolution.
Saved time histories and spatial peak maps accompany the numerical summary.
