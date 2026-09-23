# Construction and retention ablations

The [six-case design](ablations.json) extends the four corrected target-7
pilots without changing their equations, controls, bounds or accuracy criteria.
At each regularization value, it adds:

1. Recycling with retained coarse vectors and new directions, selected in
   Jacobi-scaled coordinates.
2. Sequential zero-extension transfer of the initial reference, without learning
   or replacement directions.
3. Shared tensor-product temporal factors instead of mode-dependent factors.

Every added case uses requested rank 200, 64 time slabs, the 600-second horizon
and smooth 60-second startup. These are single-run optimization ablations.
They do not replace the resolution tests, rank selection or final repeated
campaign. Numerical dependence, rank-zero fallbacks, capped runs and failures
remain in the records. The sequential optimizer can follow a different nonlinear
path; only a subsequent replay can compare identical inactive systems.

## Reproduce on a compute node

Use separate clean numerical and reporting checkouts. The design fixes the
numerical checkout to `466d547a933214a5e15219b02b3707e7b7c99c3d`, the source of
the original four pilots. The reporting checkout contains this design and driver.
The driver verifies the source, baseline checksum and dimensions, refuses an
existing output directory, and launches the numerical module from that checkout.
It does not substitute the reporting checkout's solver.

From the reporting checkout, install the locked dependencies and run each case
0 through 5 in a separate process with one GPU and eight CPU threads:

```sh
uv sync --frozen --extra study --extra coupled-gpu
uv run --frozen --extra study --extra coupled-gpu python -m deflation_example.coupled_corrected_ablations \
  --design examples/coupled_optimization/corrected_study/ablations.json run \
  --case 0 --numerical-checkout ../numerical-checkout \
  --baseline runs/baseline --output runs/ablation-0
```

Add `--dry-run` to inspect the command without solving. The hybrid deployment
uses CPU vector iterations and independent residual checks, with GPU coarse
processing. The recycling and reference methods receive the same GPU policy.
Equal retained rank does not imply equal peak memory; retain the measured
allocations and candidate-pool costs.

Summarize all ten slots, including any missing or unsuccessful runs:

```sh
uv run --frozen python -m deflation_example.coupled_corrected_ablations \
  --design examples/coupled_optimization/corrected_study/ablations.json report \
  --baselines runs/pilot-0 runs/pilot-1 runs/pilot-2 runs/pilot-3 \
  --ablations runs/ablation-0 runs/ablation-1 runs/ablation-2 \
    runs/ablation-3 runs/ablation-4 runs/ablation-5 \
  --output runs/ablation-summary
```

The summary checks final equations, optimality, original inner residuals, timer
sums, configurations and source versions. A ratio requires matched hardware,
timing boundaries and solutions. It compares saved desired fields exactly,
objectives within 1e-6 relative and temperatures within 0.001 K. Ratios on either
side of one remain visible. Missing files cannot establish solver failure;
retain scheduler exit status alongside the numerical records.

The frozen numerical source and new design must be published before these
commands form an external reproduction package. Local tests do not establish
public availability or a successful remote CI run.
