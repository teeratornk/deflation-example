# Reference-space reuse study

This study tests a fixed full-domain reference policy in complete steady and
transient state-constrained conjugate heat-transfer (CHT) optimization.
It compares four complete solvers and isolates reference transfer on matched
optimization traces. Earlier benchmark records retain their original scope.

## Claims and measurements

| Claim | Comparison or verification | Required quantities |
|---|---|---|
| Full-domain storage supplies reference values at released constraints | Replay identical optimization traces with direct restriction and sequential zero-extension transfer | Newly active and newly inactive nodes, actual rank, transferred-space difference, coarse energy-error reduction, accepted iterations and total kernel cost |
| One reference supports complete constrained optimization | Complete sequences with Jacobi-CG, reference deflation, scaled recycling and persistent-resource AmgX | Original residuals, all KKT components, accepted targets and sequences, complete costs and all failures |
| A space–time construction retains temporal coupling | Independent small trajectory optimization and mode-dependent versus tensor-product references | State/adjoint identities, initial and final blocks, objective weights, recovered control, total rank and restricted storage |
| Reuse has a useful time–memory range | Spatial, temporal and rank studies with matched starts and declared budgets | Complete time, cumulative cost, reference/candidate storage, comparable sampled process memory, caps and memory limits |
| Conditional analysis describes the measured coarse correction | Small CHT trace systems and a controlled parameter study | Remaining deflated spectrum, initial energy-error reduction, positive separation and certified conditioning improvement when available |

Residual recomputation, KKT verification, Cholesky factorization and GPU execution
support these tests. They are implementation and verification choices. The
algorithmic policy defines correction information over the full domain before
optimization and restricts it at every inactive-set update. The transfer ablation
compares two specified transfers; conclusions about recycling concern the
implemented comparator.

## Study components

1. Implement complete-sequence Jacobi-CG and recycling that retains existing
   coarse vectors with new search directions. Use Jacobi-scaled Ritz selection,
   report effective ranks, and include capture, selection and transfer costs.
2. Implement complete backward-Euler CHT optimization. Verify the full trajectory
   against an independent small constrained optimizer before timing. Compare two
   space–time reference constructions in a pilot and then freeze one policy.
3. Replay steady and transient optimization traces for the matched transfer
   ablation. Keep matrices, right-hand sides, starting vectors and tolerances
   identical. Separate activation from release counts.
4. Freeze final configurations after pilots. Run independently repeated complete
   sequences, spatial and temporal refinements, and a small rank sweep. Preserve
   every declared attempt and report accepted timings separately from failures.
5. Inspect spectral diagnostics and decide whether the longer conditional bounds
   belong in the SI. Rewrite the paper from the supported claims and numerical
   records; write the abstract last.

The primary start policy carries accepted active sets across queries and uses
the last full state as the next inner initial guess. Cold starts are a supporting
control. Recycling begins with empty history and updates between PDAS inner
solves. Reference construction is charged from the first target. Requested and
effective ranks, retained vectors, candidate pools, reference factors, restricted
bases and cached operator products are recorded separately.

The frozen steady progression is 24, 32, 48 and 64 interior nodes per axis.
Transient studies
refine space at fixed time discretization and time at fixed physical horizon.
A change in horizon forms a separate comparison. Pilots determine feasible
transient sizes and rank budgets before the final configuration is frozen.

## Physical and temporal conventions

The solid–fluid geometry, conductivity interface, prescribed transport and
distributed source control follow the existing CHT preset. The temperature bound
is a fixed physical scalar within each declared refinement comparison. Scenario
indices identify optimization queries; time levels identify states within one
trajectory. Targets move and pulse in physical time and vary across queries.

For thermal-capacity matrix C and steady state matrix A, backward Euler uses
`C (y_n - y_(n-1))/dt + A y_n = u_n`. The initial temperature is prescribed.
The objective uses right-endpoint time quadrature for both temperature tracking
and the L2 control penalty, with the same spatial cell-volume factor.
The records specify capacity values, time steps, horizon, initial data and target
parameters. Full block coupling enters the operator, transpose,
reduced Hessian, control recovery and KKT checks.

## Reproducibility

Pilots and final studies have separate output directories and protocol identifiers.
Final protocols record source identifiers, environment, order, warmup, numerical
thresholds, budgets and failure rules before timed execution. Complete sequence
costs include assembly, construction, every active-set update and inner solve,
transfers, verification and cleanup. Process initialization and warmup are
reported separately and included in preparation-inclusive totals. Memory
measurements use the same boundary and sampling method for all competitors.

The manuscript is titled *Reference-Space Reuse for Steady and Transient
State-Constrained Conjugate Heat Transfer*. It presents the formulation,
reference construction, transfer mechanism, complete optimization comparisons,
scaling and one limitations subsection. Auxiliary benchmark branches remain
available in the reproducible repository.

## Pilot reproduction

The small CPU pilot runs three complete trajectory optimizers and needs the
optional process-memory dependencies:

```bash
uv run --locked --extra study python -m deflation_example.benchmark_cht device=cpu 'methods=[jacobi,reference,recycling]' problem=transient n=4 calibration_grid=4 slabs=4 targets=3 rank=12 window=12 output=runs/transient-cpu-pilot
```

The default configuration is a pilot, with four methods and a small steady grid.
After installing the GPU, study and plotting extras, install AmgX and its binding
as described in [the GPU guide](gpu-benchmark.md). The new study uses PyAMGX source
commit `6229ff008ee5a264cfc1799eeb2f83d96da0aadc`; its source identifier and binary
hash enter each GPU worker's record. Use `--no-sync` after this native installation.
Every GPU worker initializes and warms both numerical libraries, including when
running one method for a construction comparison.

```bash
uv run --no-sync python -m deflation_example.benchmark_cht --cfg job
uv run --no-sync python -m deflation_example.benchmark_cht output=runs/steady-gpu-pilot
uv run --no-sync python -m deflation_example.benchmark_cht problem=transient n=6 calibration_grid=6 slabs=4 rank=24 window=24 output=runs/transient-gpu-pilot
```

`declared_protocol.json` precedes calibration. `protocol.json` additionally fixes
the calibrated physical bound before comparison. Each sequence runs in a fresh
process and has a separate JSON record; `results.json` contains their hashes and
acceptance flags. Compressed masks retain the inactive systems for replay. The
complete sequence and preparation-inclusive process times are separate fields.
These pilot commands establish correctness and feasibility; final performance
claims require the frozen final configurations and repeated measurements.

The transient verification tests include nonzero initial temperature, unequal time
steps, independent forward and transpose actions, the final adjoint block,
objective quadrature and a bounded-variable least-squares reference. Mode-dependent
reference vectors satisfy the homogeneous space–time eigenproblem. The tensor
alternative uses shared temporal vectors and reports their reference Rayleigh
values. Both constructions specify the total space–time rank.

## Transfer and conditioning diagnostics

Replay an accepted source sequence with direct full-reference restriction and
sequential zero-extension transfer:

```bash
uv run --locked python -m deflation_example.benchmark_transfer --records-root runs/transient-cpu-pilot --output runs/transfer --repeats 3
uv run --locked python -m deflation_example.benchmark_cht_conditioning --output runs/cht-conditioning
uv run --locked python -m deflation_example.benchmark_cht_validation --output runs/independent-validation
uv run --locked python -m deflation_example.benchmark_cht_report --records-roots runs/transient-cpu-pilot --output runs/trajectory-summary
```

Replay reconstructs each inactive matrix and right-hand side and checks the
source hashes. Both transfers start from zero and receive the same original
reference on their first solve. The sequential control subsequently carries only
its surviving entries. It receives no replacement directions. The report records
newly inactive and newly active nodes, numerical rank, fallback, coarse energy
removal and independently verified residuals. Small systems also include the
remaining nonzero spectrum in Jacobi coordinates. Reconstruction, independent
direct solutions and spectral diagnostics are separate from kernel timings.

The controlled CHT example uses an exact eigenvector of the full Jacobi-scaled
heterogeneous operator. Its prescribed restrictions progress from certified
conditioning improvement to an uninformative bound. This reference construction
differs from the analytical spaces in the timing study. The regression checks
evaluate the stated formula, including the numerical reference residual, and
compare its condition-number bound with the measured deflated spectrum.

The sequence summary verifies timing sums, ranks, numerical thresholds, masks,
trajectory recovery and KKT acceptance before generating JSON and CSV tables.
It reports failed attempts separately from accepted sequence timings. Component
medians describe their own distributions; their sum can differ from the median
complete time.

## Timing and initialization pilots

The configuration exposes the initial active set (`initial_active=empty` or
`all`) and residual-refresh interval. Every method receives the same declared
active-set initialization. Accepted histories take precedence under the matched
warm-start policy. These settings support documented stopping and initialization
pilots; they remain fixed within every final comparison.

Final memory measurements use a separate monitoring process. This avoids gaps
caused by native solver calls holding the Python interpreter lock. The monitor
samples the solver process's complete host RSS and GPU allocation, including
all numerical libraries and allocator caches. Its own storage is excluded.
The recorded maximum sampling gap quantifies the resolution of the sampled
peak. Monitor initialization and finalization lie outside the complete solve
interval and enter the preparation-inclusive process time.

`cg_factor` and `amgx_factor` specify the internal stopping margins relative to
the shared independently checked tolerance. A CG margin can prevent a small
difference between CPU and GPU residual evaluation from rejecting a solve at
the threshold. The report verifies both internal targets against the protocol.

A refinement run can read `calibration_report` from an earlier study's
`results.json`. It checks the physical bound and model conventions, records the
input hashes, and charges the measured calibration once in each
preparation-inclusive total. The filesystem location stays out of the report.
Changing the number of time slabs preserves the horizon and physical bound.

Optional `host_memory_budget_bytes` and `gpu_memory_budget_bytes` specify common
sampled-peak feasibility limits. A numerically completed sequence that exceeds
either limit retains its solution checks and timing but receives
`memory_budget_exceeded` status. These limits screen measured storage; they do
not enforce hard allocator caps. Rank and memory-budget comparisons are reported
as separate experiments.

The complete-study adapters cache `BZ` for both reference deflation and recycling.
The projection then uses `(BZ).T @ z`, which avoids an additional sparse matrix
application in each preconditioning step. Construction and retained storage enter
the same timing and memory boundary. The rank-zero Jacobi path allocates no such
product. The underlying CPU and GPU kernels expose this as an explicit option;
their legacy default remains uncached. Tests compare both implementations with
independent solutions and verify rank-zero and conditioning-fallback behavior.

The independent validation report contains nine distinct small steady and
transient problems. A bounded-variable least-squares optimizer supplies a separate
solution and KKT check for each problem. Jacobi-CG, reference deflation and
recycling then solve the complete PDAS problems. Uniform and unequal time steps
use nonzero initial temperature. The report records forward and adjoint action
checks, recovered trajectories, objective agreement, state differences and every
inner residual. These cases validate the discretization and optimizer; their
execution times do not support performance claims.

## Frozen complete-optimization campaign

`benchmark_cht_campaign` declares the final populations before execution. Each
primary comparison uses 16 distinct targets, five independently timed sequences,
all four solvers and fully matched warm starts. Steady grids have 24, 32, 48 and
64 interior nodes per axis. Transient grids have 12, 16, 24 and 32 nodes per axis
with eight time slabs. Temporal refinement uses 4, 8, 16 and 32 slabs at fixed
grid 16 and horizon 0.1. The physical bound stays fixed within each family.

The rank study uses 20, 100 and 200 directions at steady grid 32 and transient
grid 24. The Jacobi method supplies the rank-zero comparison. Cold-start controls
use those same two grids. Both CG and AmgX use an internal margin of 0.1 relative
to the independent final tolerance of `1e-10`; PDAS requires every KKT component
to be at most `1e-8`. Every first target starts with the full active set. The
residual-refresh interval is 10000 iterations. These choices follow the recorded
stopping and initialization pilots, whose failed attempts remain available.

The mode-dependent temporal reference is fixed for the final study. It satisfies
the declared homogeneous reference eigenproblem. Its measured construction pilot
costs were similar to the tensor alternative. Both candidate constructions and
their outcomes remain part of the study records.

The campaign also declares common GPU memory screens of 8, 16 and 64 GiB and a
16 GiB host screen. Numerical acceptance and memory feasibility remain separate.
Reports first compare equal requested ranks, then show the measured time of
feasible rank choices under each common screen. Those comparisons account for
the full process allocation, including numerical-library caches. Sampling can
miss shorter peaks, so these screens do not certify a hard allocation limit.

The versioned data provide the original calibration reports. To regenerate
the physical bounds, run the following CPU commands. Calibration uses a
separate direct-PDAS solve at query 0.5 and is independent of the two
subsequent comparison targets:

```bash
uv run --locked --extra study python -m deflation_example.benchmark_cht device=cpu 'methods=[reference]' problem=steady n=24 calibration_grid=24 targets=2 rank=100 window=100 threads=4 output=runs/steady-calibration
uv run --locked --extra study python -m deflation_example.benchmark_cht device=cpu 'methods=[reference]' problem=transient n=12 calibration_grid=12 slabs=8 targets=2 rank=100 window=100 threads=4 output=runs/transient-calibration
```

The recorded bounds are `0.00019498904452558432` for steady problems and
`0.00026847890969378894` for trajectories. Calibration time varies with the
deployment. Supplied recorded calibration times reproduce the published
preparation-inclusive accounting; newly measured calibration times describe
the new deployment.

After obtaining the calibration reports, run any subset from a clean checkout:

```bash
uv run --no-sync python -m deflation_example.benchmark_cht_campaign --selected steady24 steady32 --steady-calibration runs/steady-calibration/results.json --transient-calibration runs/transient-calibration/results.json --output runs/final-steady-small
```

The command's help lists all declared populations. Separate invocations can run
disjoint populations concurrently on separate GPUs. Each output directory retains
the complete campaign declaration, source identifier, calibrated protocols and
every sequence outcome. A failed solve does not prevent the remaining declared
comparisons from running.

## Regenerate the submitted figures and tables

The original numerical workers used clean source commit
`cc611292f66025f4d0f96a406c6aeb67799b7d34`. Version 0.4.0 adds the checked
publication summaries and preserves the numerical protocol. The frozen evidence
is a separate data-only tag in this repository. It keeps the installed package
small and retains all final populations, pilot attempts and diagnostic records.

```bash
git clone --branch v0.4.0 --depth 1 https://github.com/teeratornk/deflation-example.git deflation-code
git clone --branch reference-policy-data-v1 --depth 1 https://github.com/teeratornk/deflation-example.git deflation-evidence
cd deflation-code
uv sync --locked --extra plot --extra study
uv run --locked --extra plot --extra study python -m deflation_example.benchmark_evidence --records ../deflation-evidence --output runs/submission
```

This CPU command verifies every file against the data manifest, checks the
complete optimization records, reconstructs the controlled CHT matrices, and
generates the submission tables and figures. It requires no GPU and performs no
new timing comparison. Omitting the output option performs byte-hash verification
alone. Each generated report records its input hashes and generator version.

The data categories are `final`, `pilot`, `diagnostic` and `summary`. The final
category contains every frozen campaign population. The pilot category includes
stopping and construction comparisons, calibration records and larger feasibility
tests. The diagnostic category contains independent optimization checks,
conditioning examples, source traces and matched transfer replays. The summary
category freezes the publication reports. GPU reruns use the campaign command
above and the calibration records in the data checkout. Source identifiers,
configuration, numerical acceptance, hardware, library versions and each timed
repetition remain in the individual JSON records.

The manuscript importer consumes the generated `presentation`, `transfer` and
`support` directories. PDF metadata can differ between builds; the input hashes,
numerical summaries and plotted values provide the reproducibility checks.
