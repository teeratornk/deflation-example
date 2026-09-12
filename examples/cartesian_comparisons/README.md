# Complete Cartesian comparison records

The [steady](steady.csv) and [transient](transient.csv) files retain all 60
original solver rows: complete-sequence median and full timing range,
iteration counts, requested rank, start policy and verified repetitions.
The compact SI tables show the same times and ranges with one row per
configuration. The original numerical measurements are unchanged.

Every sequence contains 16 targets. Five independent timing repetitions
use the same physical target sequence. Warm starts apply to both the outer
active-set iteration and the inner solve. Each completed sequence meets the
original-residual and KKT thresholds of `1e-10` and `1e-8`, respectively.
Ranks 20 and 200 add reference and recycling runs; the corresponding Jacobi
and AmgX controls are the rank-zero rows of the rank-100 comparison.

The [manifest](manifest.json) identifies the manuscript source, unchanged
source-row checksums and CSV checksums. The
[Cartesian reproduction guide](../../docs/reference-policy-study.md#frozen-complete-optimization-campaign)
provides the configurations, numerical source and complete-run commands.

Jacobi-CG has the lowest measured complete time in several refined transient
comparisons. These outcomes remain in both the SI and these records.
