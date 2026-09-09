# Reuse costs — tutorial version 1

This tutorial regenerates cost summaries from all recorded methods and attempts. It requires CPU Python only:

```sh
uv sync --locked
uv run --locked python -m deflation_example.tutorials costs --output runs/tutorial-costs
```

The [protocol](../../src/deflation_example/conf/tutorial_costs.json) is written before processing. [tutorial_costs.py](../../src/deflation_example/tutorial_costs.py) checks timing partitions, paired systems and original residuals. `results.json` includes configurations, original measurement environments, executing source and raw-input hashes. Four CSV files summarize resources, preparation, rank/memory and recycling. The [saved output](../../examples/tutorials/costs/results.json) is generated from the same records.

## Resource persistence and hierarchy rebuilding

The [Laplacian records](../../examples/benchmarks/gpu.json) contain all six methods: GPU-QR deflation, CPU-SVD/GPU deflation, fresh-resource AmgX, persistent-resource AmgX, GPU Jacobi-CG and CPU deflation. All four repeated methods retain five repetitions per target; the two single-run controls remain labeled separately. `resources.csv` includes every method and termination status.

Persistent AmgX retains Config and Resources. Each inactive solve creates matrix/vector/solver handles and a new hierarchy. Both policies therefore rebuild hierarchies. Kernel totals include CPU conversion, handles, transfers, basis or hierarchy setup, iteration, independent residual verification and cleanup. Common reference, restriction, warmup and persistent-session costs enter the separately reported cumulative curves. These curves sum per-instance medians. The exact AmgX configuration, source `b6f2e97`, environment and every component are in the raw record.

## Complete sequence and preparation-inclusive costs

The [final CHT records](../../examples/benchmarks/extended/cht_v3/results.json) contain 60 independently timed complete sequences. `preparation.csv` preserves all grids, methods and start policies. It adds bound calibration and the separately repeated [process startup/shutdown measurement](../../examples/benchmarks/extended/startup.json), counting GPU initialization once. For fully warm-started grid 32, sequence medians are 10.805 and 22.955 s; preparation-inclusive totals are 234.580 and 246.731 s. The latter add a common measured cost to each median and have a different timing boundary. Numerical sources are `1e1551c` for CHT and `0c72aab` for startup.

## Rank and memory

The [matrix-free records](../../examples/benchmarks/extended/scale.json) compare ranks 0, 20 and 200 on four grids. `rank_memory.csv` retains all twelve attempts, requested/deployed ranks, both original residuals, timing, memory and capped status. The systems have manufactured solutions and prescribed inactive sets. Rank zero reaches the 10,000-iteration cap on grids 102 and 128. Rank 200 converges on all four grids; at grid 128 it uses 15.07 GiB of peak Torch allocation. Runs proceed in ascending grid/rank order, with minimal runtime initialization and no dedicated dense-library warmup. First-use and size-dependent costs are combined in these single-run measurements. The source is `8d785f1`.

## Last-window recycling

The [history records](../../examples/benchmarks/extended/controls.json) retain three independently repeated seven-target kernel sequences per family. The candidate pool contains only the last 100 new CG directions, excludes existing coarse vectors and uses unscaled Ritz selection. Zero extension and restriction transfer the selected directions. Both methods start at rank zero, then match the available retained rank up to 20. Reference construction is charged once; history transfer and selection are charged at every target. `recycling.csv` preserves every sequence, method and termination status. The source is `8d785f1`.

The small span example in `results.json` uses `B=diag(1,2,3)`, coarse vector `e1` and new directions `e2,e3`. These directions satisfy `Z.T B p=0`. Selecting rank one from the new pool gives `e2` and discards `e1`. This explains the scope of the specified last-window control; broader recycling policies can retain both old coarse and new directions.

## Constant-cost amortization

The report also evaluates an illustrative additional setup cost of 1.2 s with per-query savings of 0.1, 0 and -0.1 s. Positive savings give break-even count 12. Zero and negative savings give `null`: that constant-cost model has no finite break-even count. These are synthetic cost examples, separate from measured GPU or CPU timings.

## Re-running measurements

The [GPU instructions](../gpu-benchmark.md) specify native dependencies, controls and accuracy criteria. Use the pinned source identified in the relevant record to repeat its exact protocol. The commands are:

```sh
uv run --no-sync python -m deflation_example.benchmark_gpu output=runs/gpu-rerun
uv run --no-sync python -m deflation_example.benchmark_extended output=runs/cht-rerun
uv run --no-sync python -m deflation_example.benchmark_startup --output runs/startup-rerun
uv run --no-sync python -m deflation_example.benchmark_matrix_free --output runs/rank-rerun
uv run --locked python -m deflation_example.benchmark_controls --output runs/history-rerun
```

GPU commands require the installed CUDA/AmgX environment and sufficient device memory. The tutorial command above only summarizes existing results; it submits no GPU job. All original outputs remain unchanged.
