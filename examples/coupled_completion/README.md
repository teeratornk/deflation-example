# Fixed-setting coupled reference comparisons

This archive contains measurement extracts from all 117 solves in two matched
replay studies. They use three recorded inactive systems from the same coupled
transformer optimization trace. The physical model, 64 time slabs, 600 s horizon,
smooth 60 s startup, regularization and temperature bounds remain fixed.
These are linear-system replays, not complete optimization sequences.

## Reproduce the summaries and figures

From a clean checkout containing this example:

```sh
uv sync --frozen --extra study --extra plot
uv run --no-sync python examples/coupled_completion/reproduce.py \
  --output replay-summary
```

The command needs no GPU or external data. It reconstructs the timing sums,
coverage checks and plots from `measurements.json`. Every individual timing
repetition appears in the figures. The output directory must be new.

The extract retains source-file hashes, numerical source versions, hardware and
library metadata, every termination status, original residual, requested and
deployed rank, solve cost, reference-construction cost, resource costs and sampled
memory. The original record digest identifies each source record. Nested timer
histories and the captured matrices, fields and reference arrays are omitted.
This example reproduces the reported measurements; it does not rerun the PDE
solves or independently re-evaluate their residuals. The numerical execution
commands and required inputs are described in [the study example](../coupled_completion_v9.md).

## Results

The thermal and enriched thermal references retain 20--200 directions. They
require 53 iterations across the three systems, as does the velocity-frozen
preconditioner without a reference. Their measured online and construction costs
are greater. The numerical source is `17cf4bb`.

The independent energy-Krylov construction selects prefixes of 1, 2, 4 and 8
directions. Its rank-eight prefix reduces the iteration sum from 53 to 47.
Median online replay time rises from 204.6 s to 225.4 s. Construction adds
431.3 s once per three-system replay sum. These measurements establish an
iteration reduction without a time saving. The numerical replay source is
`b5d720c`; the reference was constructed with `cd21830`.

All 117 solves meet the independently evaluated relative residual threshold of
`1e-10`. The comparisons retain the requested ranks, with no fallback. Neither
study establishes a complete coupled optimization speedup.

## Deployment and timing scope

One Jacobi-preconditioned record in the first study used a different CUDA
driver. The strict deployment check stopped the automatic campaign. The archive
keeps that record and both remaining Jacobi records in their deployment groups;
neither group has a complete three-system Jacobi comparison. All reference
methods and the velocity-frozen control have complete matched coverage. The
low-rank CPU comparison also has complete matched coverage.

Costs are sums of per-system measurements grouped by repetition, with reference
construction charged once. They are not independently timed optimization
sequences. Factory creation and cleanup are included. Common preparation costs
remain separate. The amortization calculation repeats the same system mix at
constant measured costs; its replay blocks are not optimization queries. No
tested reference has positive online savings relative to its matched control.
The CPU and hybrid studies remain separate, and their costs are not pooled.
