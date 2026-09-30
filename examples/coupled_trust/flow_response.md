# Local momentum-response diagnostic

This example investigates a terminated trajectory without changing its
temperature fields, physical equations, final criteria, or numerical records.
It holds the preceding velocity fixed and perturbs one time slab along the
temperature difference from the preceding optimization target. The perturbation
has unit maximum amplitude in kelvin and vanishes on the thermal boundary.

```bash
uv run --extra study python -m deflation_example.coupled_flow_response \
  --record RESULTS/record.json --position 1 --slab 8 \
  --budget-seconds 180 --output flow-response-slab8
```

Use a terminated transient record with both temperature and flow fields saved
for the selected and preceding target. Slab indices start at zero. The source
checks that the momentum implementation and physical inputs match the record;
the output identifies both numerical and diagnostic sources. An existing output
directory is never overwritten.

The declared comparisons are three initial guesses at the unchanged temperature
(the retained flow, preceding time level, and preceding target), signed
perturbations of 1e-4, 1e-6, and 1e-8 K from the retained flow, and the existing
residual-load continuation at signed 1e-6 K. Each case has its own time limit.
The tangent equation uses the analytic momentum Jacobian and a fresh residual
check. Every terminal flow, residual, status, and nonlinear history is retained.
Derivative differences are reported only for independently verified roots.

These single-slab calculations diagnose local response and initialization
sensitivity. They neither optimize a trajectory nor establish a complete-solve
speedup. A failed perturbation does not establish that no solution exists.
Perturbed temperatures are diagnostic inputs; they are not clipped to the
optimization bounds. A change in flow branch must be distinguished from a
derivative error before using these results to modify the optimizer.
