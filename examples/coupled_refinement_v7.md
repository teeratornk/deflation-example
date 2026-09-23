# Fixed-source coupled refinement

This study preserves the smooth-stabilization optimization and its saved signed
source. It tests forward solution procedures and temporal resolution before any
new optimization or performance claim. Earlier unsuccessful attempts remain
separate records.

Create the CPU environment with `uv sync --frozen --extra plot --extra study`.
On an allocated GPU, add `--extra coupled-gpu`. The commands below use
`--no-sync` to retain those explicitly installed optional dependencies. Run
the numerical examples on compute nodes. Input names such as `BASELINE` and
`OPTIMIZATION` denote the selected saved directories; output directories must
be new.

## Failed-step comparison

Run the tests, then compare four predeclared procedures at the first unsuccessful
step. Each receives the same source and verified preceding physical states.
The two starting guesses are the saved failed candidate and the preceding
physical state. The comparison changes neither the time step nor the equations.

```sh
uv run --no-sync pytest tests/test_coupled_repair_protocol.py tests/test_coupled_newton_replay.py tests/test_coupled_forward_verify.py tests/test_coupled_step_spectrum.py
uv run --no-sync python -m deflation_example.coupled_newton_repair \
  --baseline BASELINE --optimization OPTIMIZATION --replay FAILED_REPLAY \
  --method reference --target-position 0 --protocol refinement_v7 \
  --cap 100 --threads 8 --output REPAIR
```

The procedures are equation-maximum Newton, fixed-scaled-merit Newton,
fixed-scaled Newton with a 0.05 update limit, and fixed-scaled Newton followed by
Anderson iteration. Newton allows 100 iterations. The first procedure allows
21 backtracks; the other three allow 40. The fallback allows 300 Anderson
iterations and 100 iterations per flow solve. Every returned field is checked
against the original equations with a `1e-12` equation target and `1e-6`
conservation limits. A complete-equation directional derivative check precedes
the comparison. A failure of that check prevents solver selection.

If the four procedures fail to produce a matching verified root, the separate
`--protocol continuation_v7` fallback varies an auxiliary storage time step
while keeping the original source and previous physical fields fixed. It starts
at one sixteenth of the original step, increases the continuation increment
after successful stages, and halves it after a failure. It permits 64 stages
with minimum increment `1/1024`. Every returned candidate is verified at the
original physical step. Intermediate auxiliary roots are neither physical
substeps nor successful solutions of the original equation. This procedure
does not modify the declared temporal discretization.

The first procedure in the declared order that converges from both starts and
agrees within `1e-4 K` in temperature and `1e-10 + 1e-5*scale` in each flow field
is eligible for a complete replay. This agreement is local evidence, not proof
of uniqueness. All procedures and outcomes are retained. The complete replay
must independently meet the same criteria at every time level.

## Resolution and subsequent optimization

The saved source is piecewise constant on the original optimization intervals.
Temporal refinement subdivides those intervals without changing the source.
The new predeclared refinement cohort is 256, 512 and 1024 slabs, with extension
to 2048 and 4096 if the thresholds require it. The original 64/128/256 comparison
and all failed-step repairs remain separate, linked diagnostics; no coarse
failure is replaced by a successful fine solve.
Resolution requires two successive nested comparisons with maximum temperature
differences at all refined times at most `0.05 K`, relative tracking changes at
most `0.01`, and nonincreasing changes. Bound excess is reported separately.
Solver convergence alone establishes neither temporal resolution nor physical
bound satisfaction.

Pure Newton records can be assessed without changing their history format by
supplying the separately recomputed, checksum-bound equation audits:

```sh
uv run --no-sync python -m deflation_example.coupled_forward_verify \
  --baseline BASELINE --optimization OPTIMIZATION --forward REPLAY \
  --method reference --target-position 0 --threads 8 --output AUDIT.json
uv run --no-sync python -m deflation_example.coupled_time_resolution_report \
  --replays REPLAY64 REPLAY128 REPLAY256 \
  --equation-audits AUDIT64.json AUDIT128.json AUDIT256.json \
  --verify-equations --temperature-scale 20 --initial-value 0 \
  --plot --output RESOLUTION
```

Use one audit for each replay in the same order. The report requires matching
source, input, record and field identifiers, time grids and numerical checks.
Failed or missing declared refinements remain in the population and prevent a
passing resolution assessment. The temperature scale and initial value above
are specific to the declared transformer model.

Reoptimization, any declared inward optimization margin, and performance
comparisons follow the resolution checks. The physical bound remains unchanged.
No failed solve is converted to a completed-solve speedup.

For a subsequent nested-grid optimization, `coupled_sequence` supports
`initial_trajectory_directory=OPTIMIZATION`,
`initial_trajectory_method=reference`, and `initial_trajectory_position=0`.
Declare one target stage, an integer temporal refinement, the same physical
model, horizon, target, regularization and bounds, and no competing snapshot or
restored warm start. The initializer interpolates temperature and flow guesses
in physical time, including the initial condition. The optimizer reevaluates
the full trajectory and recovers new controls and derivatives. No secant pairs,
recycling history or old controls are imported. Bound-violating guesses are
rejected without clipping. This initializer does not itself establish temporal
accuracy or authorize a change in the optimization margin.

After the resolution and backend memory checks pass, a prepared finer-grid
configuration can be used as follows:

```sh
uv run --no-sync python -m deflation_example.coupled_sequence --config-name coupled_refinement \
  baseline_directory=BASELINE initial_trajectory_directory=OPTIMIZATION \
  method=reference rank=20 recycle_window=20 output=REFINED_OPTIMIZATION
```

This is a configuration for subsequent work, not a reported result. It keeps
the 600-second horizon, 60-second target startup, target 7, physical temperature
bounds, regularization, smooth consistent stabilization and final criteria.
It changes the temporal grid to 256 slabs and limits GPU blocks to five columns.
The smaller block width leaves more room for coarse spaces after factor and
triangular-plan storage; complete-process memory still requires a runtime check.
All methods must share the initialization and final criteria; rank-zero uses
`method=jacobi rank=0`. Reoptimization requires a new fixed-source forward
assessment before any physical-feasibility claim.

## Memory preflight

The memory screen separates named arrays and sampled process allocation.
On an allocated GPU, the optional block-width study measures the device
triangular plans for one saved momentum factor and independently checks the
forward and transpose actions against the CPU matrix:

```sh
uv run --no-sync python -m deflation_example.coupled_memory_screen \
  --baseline BASELINE --optimization OPTIMIZATION --method reference \
  --target-position 0 --sample-steps 15 63 --slabs 128 256 \
  --ranks 0 20 100 200 --gpu-widths 20 50 100 --threads 8 --output MEMORY
```

Per-trajectory values extrapolate named factor and plan arrays. For consistent
stabilization, the screen also reports the thermal-source factors and combined
momentum/source estimates. The estimates exclude other operators, coarse
spaces and opaque workspace.
They are screening estimates, not complete optimization peaks or allocation
guarantees. CPU-only screening omits `--gpu-widths` and can add
`--checkpoint-comparison` to compare retained and recomputed LU actions.
