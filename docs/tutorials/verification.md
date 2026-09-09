# Accuracy and stopping — tutorial version 1

Run the small CPU checks and summarize all three recorded CHT stopping protocols:

```sh
uv sync --locked
uv run --locked python -m deflation_example.tutorials verification --output runs/tutorial-verification
```

The [protocol](../../src/deflation_example/conf/tutorial_verification.json) is written before calculation. `results.json` records the source, environment, input hashes and numerical diagnostics; `stopping.csv` reports each grid, start policy and method. [Saved outputs](../../examples/tutorials/verification/results.json) identify their generating source. [tutorial_verification.py](../../src/deflation_example/tutorial_verification.py) supplies the examples.

## Accuracy checks

1. **Active-set release.** For `H=[[2,-1],[-1,2]]`, `f=[0.2,2.9]` and upper bound one, initialize both indices as active. PDAS releases the first index and gives `[0.6,1]`. The full update history includes the signed-multiplier and KKT checks.
2. **Objective and stationarity.** L-BFGS-B independently solves the same quadratic. The report gives objective agreement and separately computed KKT components. A second controlled quadratic, `H=diag(1,1e6)`, has exact minimizer zero. The trial `[0,1e-9]` has objective error `5e-13` and gradient infinity norm `1e-3`. Objective agreement alone does not establish stationarity.
3. **Original and recurrence residuals.** A two-dimensional float32 teaching recurrence produces a zero recurrence residual with a nonzero double-precision original residual. Every iteration is recorded. A separate call to the production float64 solver meets the original-residual criterion. The float32 illustration is deliberately separate from the double-precision manuscript measurements.
4. **Corrected initial guess.** For `B=[[2,1],[1,2]]`, coarse vector `e1` and `b=B e1`, the projected right-hand side is zero. Accepting a zero state from that projected equation leaves original relative residual one. The production solver applies the coarse correction and returns `e1` with the prescribed original residual.

Primal feasibility is an absolute component. Stationarity, dual feasibility, complementarity and projected gradient use the documented normalization by `max(1, norm(f, inf))`. The final CHT linear tolerance is the independently recomputed original relative residual `1e-10`; the outer KKT limit is `1e-8`.

## Recorded stopping protocols

| Protocol | Deflation trigger | AmgX native target | Recorded outcome |
|---|---|---|---|
| 1 | Relative `1e-10` | `1e-10` with normalized RHS/guess | Some native-success AmgX solves fail independent residual verification. |
| 2 | Relative `1e-11` | Absolute `1e-11 * norm(b)` with original RHS/guess | Deflation breakdowns occur at grid 32; all AmgX sequences pass. |
| 3 | Relative `1e-10`, with fresh checks during iteration | Absolute `1e-11 * norm(b)` | All 60 final-protocol sequences pass. |

The final independent acceptance rules are matched across methods and protocols. Each protocol retains 60 independently measured sequences: two grids, three start policies, two methods and five repetitions. The tutorial checks every raw sequence hash and all acceptance/timing records. Accepted-time medians exclude failed sequences; their elapsed attempt times and failure types remain in the JSON output. A missing accepted population has `null` timing.

Full outer/inner traces remain under [cht_v1](../../examples/benchmarks/extended/cht_v1), [cht_v2](../../examples/benchmarks/extended/cht_v2) and [cht_v3](../../examples/benchmarks/extended/cht_v3). Their `protocol.json` files contain the exact AmgX configurations and numerical source identifiers. The source revisions are `235221a`, `9939c11` and `1e1551c`, respectively. Re-running an earlier GPU protocol requires its corresponding source checkout; current code uses protocol 3. [GPU benchmark instructions](../gpu-benchmark.md#extended-complete-cht-protocol) give the native-library installation and execution commands. This tutorial reads the GPU results without requiring CUDA.
