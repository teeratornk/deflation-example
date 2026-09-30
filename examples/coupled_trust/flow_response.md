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

## Curvature-selected flow initial guesses

For a strongly sensitive slab, this additional check projects the quadratic
momentum residual onto its velocity response. The resulting scalar approximation
selects three Newton initial guesses at the unchanged temperature and preceding
velocity. Its predicted turning point is diagnostic; every proposed root must
satisfy the full original equations independently.

```bash
uv run --extra study python -m deflation_example.coupled_flow_branch \
  --record RESULTS/record.json --response flow-response-slab8/record.json \
  --output flow-branch-check
```

The seed multipliers are -1, 1, and 2, each with a 180 s limit. The output
retains every result and its field checksum. Convergence of two distinct roots
would establish local numerical nonuniqueness at this time step; it would
neither establish a physical bifurcation nor choose the physical trajectory.

Add `--trajectory-check` to evaluate both the retained and the unit-multiplier
flow seeds over the complete saved temperature trajectory. Each evaluation has
a 300 s flow-solve budget. It keeps the temperature fixed and recovers the
corresponding source control from the thermal equations. This is an evaluation
of the temperature-elimination objective, not a fixed-control forward assessment
or a new optimization. Both equation checks, adjoint checks and KKT components
are retained, even when the alternative seed provides no improvement.

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

## Matched flow-branch restart

After the complete trajectory check, select either assessed flow explicitly:

```bash
uv run --extra study python -m deflation_example.coupled_radius_restart \
  --record RESULTS/record.json --position 1 \
  --branch-assessment flow-branch-check/record.json --initial-branch retained \
  --minimum-radius-K 1e-10 --outer-cap 100 --budget-seconds 7200 \
  --output branch-restart-retained
uv run --extra study python -m deflation_example.coupled_radius_restart \
  --record RESULTS/record.json --position 1 \
  --branch-assessment flow-branch-check/record.json --initial-branch alternate_seed \
  --minimum-radius-K 1e-10 --outer-cap 100 --budget-seconds 7200 \
  --output branch-restart-alternative
```

Both commands use the identical initial temperature and unchanged physical
inputs. They load the complete assessed flow trajectory and recover the control
through the thermal equations. The selected flow must survive fresh evaluation
and optimizer initialization: its nodal velocity distance from the selected
seed must remain below one tenth of the assessed branch separation plus
1e-12 m/s. This measures initialization consistency; it does not select a
physical branch or constrain later iterates to a prescribed velocity.

The two-hour budget covers optimization. Initial verification and final checks
have separate intervals within the total diagnostic time. Complete checkpoints
include the retained temperature, flows, objective, KKT components, secant
history, quadratic state, and remaining-radius policy, bound to the source and
input checksums. Initial branches, previous attempts, and output directories
remain separate.

The output recomputes the objective, gradient, and KKT components at the returned
state. A converged restart additionally runs the existing three-step-size
directional and adjoint verification. `gpu_gate_passed` requires all final
checks. Budget exhaustion, branch loss during initialization, and verification
failures remain explicit outcomes. GPU comparisons proceed only after this
gate; a decrease in objective alone cannot establish optimization convergence.

Generate the paired summary and figure after both arms terminate:

```bash
uv run --extra study --extra plot python -m deflation_example.coupled_branch_restart_report \
  --records branch-restart-retained/record.json branch-restart-alternative/record.json \
  --plot --output branch-restart-summary
```

The summary checks the common physical inputs, source and restart policy,
verifies returned-field checksums, and retains both termination statuses.
The figure reports retained-state histories without treating unsuccessful
restarts as completed optimization timings.

## Momentum time refinement and complete discretization study

The next comparison separates the local momentum response from complete
optimization and fixed-control time refinement. It leaves the archived
trajectories unchanged. Run these commands on a compute node.

For the nominal target-7 trajectory and the two target-8 branch restarts,
integrate the ninth original interval using one, two or four substeps. The
temperature varies linearly between its saved endpoints; perturb its final
value by -1e-6, 0 or 1e-6 K along the same normalized target-difference
direction. The velocity at the interval start stays fixed. Subsequent
substeps use their computed predecessor, including in the sensitivity equation.

```bash
uv run --extra study python -m deflation_example.coupled_interval_refinement \
  --record branch-restart-retained/record.json --position 1 \
  --direction-record RESULTS/record.json --slab 8 --subdivision 2 \
  --perturbation-K=1e-6 --output interval-retained-2-plus
```

Use `RESULTS/record.json --position 0` for the nominal trajectory and the
alternative restart record for the third trajectory. Retain all 27
combinations. Each interval has a 900 s limit. A local root or a smaller
sensitivity alone does not establish convergence of the thermal optimizer.

The complete comparison uses both thermal transport forms and 16, 32 and
64 time slabs at the original 600 s horizon. All six sequences retain the
60 s target startup, targets 7/8/9, physical properties, momentum equations,
boundary data, control regularization and temperature bounds. The first
target starts from the physical initial temperature and baseline flow;
subsequent targets use their verified predecessor.

```bash
for form in advective skew; do
  for slabs in 16 32 64; do
    uv run --extra study python -m deflation_example.coupled_discretization \
      --source-record RESULTS/record.json --transport "$form" --slabs "$slabs" \
      --output "sequence-$form-$slabs"
  done
done
```

Each sequence has an eight-hour budget and 200 outer iterations per target.
The initial trust radius is 0.25 K, its minimum is 1e-10 K, and rank-zero CG
uses the three-sweep velocity-frozen preconditioner. The final inner residual,
KKT and momentum tolerances remain 1e-10, 1e-8 and 1e-12. Global mass and
energy checks retain their 1e-6 threshold. Local mass imbalances are recorded
separately: Taylor--Hood weak continuity does not imply exact cellwise balance.

Intermediate projected solves use adaptive targets up to 1e-2. The summary
checks their independently recomputed residuals against those targets and
requires a final strict phase at 1e-10, together with the final nonlinear
and coupled-equation checks. It uses the projected-solver audit; the earlier
PDAS audit has a different intermediate tolerance policy. Updating this
reporting check does not change the frozen numerical runs.

The skew thermal form includes half the discrete velocity divergence times
temperature. Its streamline test weights that same term, along with
advection, diffusion, storage and sources. The implementation differentiates
both the Galerkin and weighted divergence contributions. Directional and
transpose tests cover each form; completed sequences receive independent
derivative checks at every returned target. These are new discretization
comparisons, separate from the earlier advective timing records.

For each sequence that passes optimization and independent verification,
replay all three saved controls with subdivisions two and four. Use a new
directory for every target and subdivision:

```bash
uv run --extra study python -m deflation_example.coupled_newton_replay \
  --baseline BASELINE --optimization sequence-advective-32 --method jacobi \
  --target-position 0 --subdivision 2 --tolerance 1e-12 --cap 30 \
  --time-scheme backward_euler --line-search equation_max --backtrack-cap 21 \
  --output replays/advective-32-target-0-x2
uv run --extra study python -m deflation_example.coupled_discretization_replay \
  --optimization sequence-advective-32 \
  --forward replays/advective-32-target-0-x2 --position 0 \
  --output replays/advective-32-target-0-x2/verification.json
```

The saved signed source is constant on each original interval and is copied
unchanged to its subdivisions. Temperature is neither clipped nor reoptimized.
The check recomputes the coupled equations from saved fields. It compares
every refined temperature with linear interpolation of the optimized
trajectory, including the initial condition, and reports maximum and
mass-weighted RMS differences together with bound violations. A difference
below 0.05 K is a discrete sensitivity criterion, not a physical error bound.

Summarize all six sequences and 27 local cases with
`python -m deflation_example.coupled_discretization_report --help`.
Supply sequences in advective-16/32/64 then skew-16/32/64 order; supply local
cases in nominal/retained/alternative, subdivision-1/2/4, perturbation-minus/zero/plus
order. The summary waits for the full declared population and selects the
coarsest configuration meeting every optimization, derivative and replay
criterion. It prefers the advective form on a grid tie and never uses elapsed
time for this selection. Missing or unsuccessful comparisons stay visible.
If none passes, the bounded study ends without a GPU speedup claim.

Add `--plot` to write `interval_refinement.pdf` and its PNG counterpart beside
the summary. The annotations retain the number of successful signed
perturbations, including failed original-step cases.

An initial-flow failure can be reproduced separately without rerunning the
optimizer or changing its criterion:

```bash
uv run --extra study python -m deflation_example.coupled_initial_flow \
  --record sequence-advective-16/record.json --output initial-flow-check
```

This diagnostic saves the returned flow and every Newton residual. It also
evaluates that same field using extended accumulation and a separated storage
term. Those comparisons investigate cancellation; they do not replace the
original final residual or change a failed status.

Only a completed six-case summary with action `gpu_feasibility` enables the
GPU comparison. Allocate one GPU per run and use all five declared arms:

```bash
for arm in rank0 reference8 reference16 recycling8 recycling16; do
  uv run --extra study --extra coupled-gpu python -m deflation_example.coupled_discretization_gpu \
    --summary SUMMARY.json --cpu-sequence SELECTED_SEQUENCE --replay-root replays \
    --arm "$arm" --repetition 0 --output "gpu-feasibility/$arm"
done
```

The hybrid backend keeps momentum factors and individual sparse solves on the
CPU and coarse processing on the GPU. The five arms share the selected
physical problem, initial states, three-sweep preconditioner, stopping rules
and complete timing boundary. Recycling retains a 48-direction window and
the existing coarse information. The fixed references retain 8 or 16
directions from the declared 48-step initial-trajectory construction. All
construction and update work is charged. These small-rank comparisons are
distinct from a wholly device-resident flow solver.

After all five feasibility runs pass, use repetitions 1 through 5, supply
`--feasibility-root gpu-feasibility`, and choose a new output directory for
each run. The driver checks the numerical source and all feasibility records
before permitting repeated timings. Every arm keeps the eight-hour budget;
unsuccessful runs remain outcomes rather than completed-solve speedups.
