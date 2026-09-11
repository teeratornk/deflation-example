# Coupled optimization ablations

The [declared design](protocol.json) contains 31 single-target computations.
Each transient target is a complete 600-second trajectory with a smooth
60-second startup. This design extends the operating-point pilots; it does not
replace the prescribed-flow results or the earlier coupled attempts.

| Factor | Comparison | Solvers |
| --- | --- | --- |
| Total space–time rank | 20, 100, 200, with rank-zero control | Reference, recycling, Jacobi-CG |
| Upper temperature bound | 353.3, 357.3, 361.3 K | All three |
| Desired trajectory | Target indices 0, 7, 15 | All three |
| Temporal discretization | 32 versus 64 slabs, fixed horizon | All three |
| Nonlinear model | Zero versus ten damped BFGS secants | All three |
| Line search | Halving versus safeguarded quadratic backtracking | All three |
| Temporal reference | Mode-dependent versus shared factors | Reference |
| Spatial reference | Scaled-Schur versus diffusion modes | Reference |
| Reference transfer | Full-domain restriction versus sequential zero extension | Reference |

The baseline appears once. Its unchanged solver controls also support the
reference-only construction comparisons. The rank-zero control constructs no
coarse space. Recycling combines existing coarse vectors with new directions
and selects in Jacobi coordinates. Rank comparisons also change the recycling
direction-window budget to the same declared rank; this is a matched retained-rank
comparison, not an equal-memory comparison.
The reference rule also uses the requested rank as its spatial candidate-pool
budget. The rank sweep therefore evaluates that declared construction policy;
it does not hold a larger candidate pool fixed while truncating only its output.

## Run on a compute node

Use the `coupled-ablations-source-v1` source tag and keep that checkout fixed
for the whole design.
The source and baseline checksums are recorded automatically. A verified computed
flow baseline is required; its reproduction is described in the
[parent example](../README.md).

```bash
git clone https://github.com/teeratornk/deflation-example.git
cd deflation-example
git checkout coupled-ablations-source-v1
uv sync --frozen --extra study --extra plot
uv run python -m deflation_example.coupled_ablations list
uv run python -m deflation_example.coupled_ablations gate --baseline runs/stabilized-baseline --slabs 64 --output runs/ablation-gate64
uv run python -m deflation_example.coupled_ablations gate --baseline runs/stabilized-baseline --slabs 32 --output runs/ablation-gate32
uv run --extra study python -m deflation_example.coupled_ablations run --baseline runs/stabilized-baseline --gate runs/ablation-gate64 --case 0 --root runs/ablations-cpu
```

Run every index printed by `list` in a separate process. Indices 19–21 are the
32-slab group and require `--gate runs/ablation-gate32`; all other cases use the
64-slab gate. The driver verifies the gate's grid, physical parameters, baseline,
source version and numerical checks before starting. The three target functions
share the same state/source derivative; their tracking term has the same Hessian.

The CUDA comparison uses a separate output root and the seven predeclared
rank/solver cases, with the same physical and numerical settings:

```bash
uv sync --frozen --extra study --extra plot --extra coupled-gpu
uv run --extra coupled-gpu pytest tests/test_coupled_cuda.py -q
uv run --extra study --extra coupled-gpu python -m deflation_example.coupled_ablations run --baseline runs/stabilized-baseline --gate runs/ablation-gate64 --device cuda --case 0 --root runs/ablations-gpu
```

Indices 0–6 form the initial rank/solver comparison on each device; the remaining
24 cases belong to the CPU screen. CPU and CUDA
records must remain separate. Numerical verification is required for the backend
before its application runs. The first screen uses one run per configuration;
repeat a frozen subset with isolated resources before publishing performance
ratios or timing variability.

## Quantities and interpretation

Each computation records complete time, reference construction, optimizer and
inner iterations, requested/deployed rank, fallback status, independent residuals,
all five optimality components, activation/release counts and sampled process
memory. Calibration and process preparation have separate totals. The common
progress output is enabled for this diagnostic design and its cost is included.
Controls, desired temperatures, resulting temperatures and velocities are saved
for verified field comparisons and independent resolution checks.

Sequential transfer starts with the same reference at the first inactive set.
Thereafter it stores only the preceding restricted entries. It transfers at every
requested inactive system, including a system solved by the initial-residual
guard. An entry removed by an earlier restriction stays zero after release.
Numerically dependent columns are handled by the ordinary rank and conditioning
checks; the control adds no replacement or learned vectors. The transfer cost
is included in basis processing and complete time. Its full optimizer can follow
a different nonlinear path. This comparison is distinct from a replay of
identical systems and from adaptive recycling.

The 32-slab runs test sensitivity to the discrete temporal problem. They do not
establish a physically resolved solution. Fixed-source spatial and temporal
replays remain necessary for that claim. Effects from this one-factor design
also need not add independently when several settings change together.

## Summaries and figures

```bash
uv run --extra plot python -m deflation_example.coupled_ablations report --root runs/ablations-cpu --output runs/ablation-summary-cpu --plot
uv run --extra plot python -m deflation_example.coupled_ablations report --device cuda --root runs/ablations-gpu --output runs/ablation-summary-gpu --plot
```

The JSON and CSV retain every declared case. Missing, running, failed and capped
computations remain visible. Figures use circles for verified solutions and
crosses for elapsed work in unsuccessful attempts; they never turn failed solves
into speedups. Empty positions have no completed measurement. The report checks
the declared settings, source/deployment consistency and the final accuracy and
timing records. Use a new output directory for every summary snapshot.
