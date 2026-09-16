# Coupled verification and complete-cost study

The coupled example must establish discrete optimality, thermal resolution,
and the complete cost of reference reuse separately. The declared criteria and
comparison population are in `publication_protocol.json`. Numerical failures
remain part of the record; they cannot supply a completed-solve speedup.

## Corrected forward verification

`fresh_trajectory.py` runs fixed-scaled Newton from the physical initial fields.
It preserves the saved source, equations, time grid, final residual criteria,
and temperature bounds. Each step is independently checked. This driver differs
from the mixed-procedure continuation, which imports an earlier verified prefix.
The initial checks use a previously saved control and establish solver behavior.
Final resolution checks must use a completed optimization's control.

Run on a compute node from the source accompanying the results:

```bash
uv sync --frozen --extra study --extra plot
uv run pytest tests/test_fresh_corrected_trajectory.py tests/test_repaired_continuation.py tests/test_coupled_newton_replay.py
uv run python examples/coupled_optimization/fixed_point/fresh_trajectory.py \
  --baseline "$BASELINE" --optimization "$OPTIMIZATION" \
  --subdivision 1 --repetition 0 --threads 8 --output "$NEW_OUTPUT"
```

For the nested fine mesh add `--fine-baseline "$FINE_BASELINE"`; use subdivision
two for 128 slabs when the saved control has 64 intervals. The signed source is
constant within each original time interval and prolonged on the nested mesh.
This driver also supports the existing `--target-position` selection for saved
sequences. It never selects a control using its forward outcome.

An interrupted run may use `--resume` with otherwise identical arguments.
Every committed field is hash checked and its original equations recomputed.
Terminal failures cannot resume. Missing costs of killed attempts remain unknown;
such attempts cannot become uninterrupted timing repetitions. Process time,
step time, and nested nonlinear time are distinct and must not be added together.

## Evidence required before manuscript inclusion

1. Complete optimization satisfies independently evaluated residual and KKT
   criteria. The coupled problem supports a stationary-point claim, without a
   global-optimality conclusion.
2. Final fixed-control responses meet the declared spatial and temporal
   refinement criteria. Report violations and numerical uncertainty explicitly.
3. Matched complete comparisons include Jacobi, reference and recycling, all
   repetitions, all termination statuses, construction and update costs, and
   comparable sampled process memory. GPU acceleration is identified only for
   the operations executed on the GPU.
4. Figures and tables derive from these immutable records. Local solver repairs,
   transfer replays, and formulation checks retain their narrower interpretations.

The optimization commands and Hydra configurations remain those identified by
their own frozen studies. This forward driver does not modify those sources or
relabel their performance. A formulation change requires renewed numerical and
derivative verification and new timings for all methods.
