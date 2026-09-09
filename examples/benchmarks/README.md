# Recorded benchmark results

These complete reports were generated from clean numerical source commit
`b6f2e97` (version 0.2.0). Each JSON file includes the full source commit,
per-module hashes, software environment and numerical controls.

| File | Measurement population |
|---|---|
| `gpu.json` | Six verified Laplacian inactive systems, four GPU implementations, five timed repetitions each: 120 runs, plus warmups and single-run controls |
| `cht-sequences.json` | Five complete three-target CHT PDAS sequences per method; every outer step and inner solve is recorded |
| `structure.json` | Eight controlled restriction/perturbation cases and 45 paired CPU Ritz/reference timing runs |

The GPU comparison tests fresh and persistent AmgX **resources**, rebuilding
the hierarchy every time. The cumulative plot is a sum of instance medians.
Only the CHT sequence report contains independently repeated complete
optimization measurements. It also reports shared calibration costs, which
must not be hidden when comparing full benchmark-generation costs.

Regenerate the GPU figure and the CSV containing every timed component:

```bash
uv run --locked --no-dev --extra plot python -m deflation_example.benchmark_report examples/benchmarks/gpu.json --output runs/recorded-figures --plot
```

This command needs no GPU or AmgX installation. The raw reports are numerical
evidence, not expected wall-time targets for another machine. See
[the benchmark protocol](../../docs/gpu-benchmark.md) for rerunning the solves.
