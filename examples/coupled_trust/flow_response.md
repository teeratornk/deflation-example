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

Summarize the three selected slabs and check every saved-field checksum:

```bash
uv run --extra study --extra plot python -m deflation_example.coupled_flow_response_report \
  --records flow-response-slab7/record.json flow-response-slab8/record.json \
  flow-response-slab9/record.json --plot --output flow-response-summary
```

The summary retains all initial-guess and continuation outcomes. The figure
compares the signed direct perturbations with the analytic tangent. Its
velocity norm is a Euclidean norm of nodal values; it is not a spatially
weighted norm. Unsuccessful solves appear as explicit status annotations,
without an inferred sensitivity value.

## Matched trust-radius restart

After diagnosing a restricted local response range, this bounded comparison
tests the trust-radius floor from the same retained unsuccessful trajectory:

```bash
uv run --extra study python -m deflation_example.coupled_radius_restart \
  --record RESULTS/record.json --position 1 --minimum-radius-K 1e-6 \
  --budget-seconds 1800 --output radius-original
uv run --extra study python -m deflation_example.coupled_radius_restart \
  --record RESULTS/record.json --position 1 --minimum-radius-K 1e-10 \
  --budget-seconds 1800 --output radius-smaller
```

Both arms start with a 1e-6 K radius, discard the old secant and recycling
history, and use rank-zero, velocity-frozen preconditioned CG on the CPU.
The maximum number of outer steps is 40. Each arm retains the physical
equations, temperature bounds, and final residual and KKT criteria from its
source record. The default optimization procedure still uses the original
radius policy. Explicit radius settings are local to a solver call and bind
its checkpoint identity.

The output preserves terminal fields, trial histories, independently evaluated
equations and adjoints, and the final KKT components. A numerical convergence
result requires subsequent independent derivative checks and complete
comparisons before it can support a revised optimization claim. Diagnostic
restart times are separate from the cost of solving the original sequence.
