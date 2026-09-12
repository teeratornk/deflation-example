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
