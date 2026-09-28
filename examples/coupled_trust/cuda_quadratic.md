# CUDA fixed-quadratic assessment

This example evaluates the same coupled Gauss--Newton quadratic on CPU and CUDA.
The GPU stores the momentum/source factors, tangent and transpose operators,
restricted frozen preconditioner, and deflation data. Sparse factorization,
active-set updates, projected line searches, and independent verification remain
on the CPU. This is GPU acceleration of the inner solves, rather than an entirely
device-resident nonlinear optimizer.

The frozen preconditioner uses the CPU-factored slab blocks and the same number
of stationary sweeps on both devices. The device recurrence matches the CPU's
operator-preconditioned projected recurrence. Unknown operator preconditioners
are rejected. Only one inactive-set preconditioner remains cached; changing the
mask releases its predecessor's device plans. Residual-correction calls can reuse
the unchanged preconditioner. Transfers, triangular analysis, applications, and
verification are included in the measured solve intervals.

Install the pinned CUDA extra on a CUDA-12 compute node:

```sh
uv sync --locked --extra study --extra coupled-gpu
uv run --no-sync pytest tests/test_coupled_cuda.py tests/test_cuda_consistent_action.py \
  tests/test_coupled_cuda_frozen.py -q
```

These tests check CPU--GPU agreement for the coupled derivative, its transpose,
normal products, restricted frozen sweeps, reference deflation and recycling.
They also check strict optimality of small complete box quadratics, changing
masks, warm starts, cooperative stopping, and additive timing components.

After verification, run the same saved quadratic with and without a reference:

```sh
uv run --no-sync python -m deflation_example.coupled_qp_globalization \
  --trace INPUT/inactive-trace-00 --baseline BASELINE --quadratic 2 \
  --method projected --device cuda --rtol 1e-2 --reference-rank 0 \
  --budget-seconds 7200 --output OUTPUT/cuda-frozen
uv run --no-sync python -m deflation_example.coupled_qp_globalization \
  --trace INPUT/inactive-trace-00 --baseline BASELINE --quadratic 2 \
  --method projected --device cuda --rtol 1e-2 --reference-rank 16 --reference-steps 48 \
  --budget-seconds 7200 --output OUTPUT/cuda-reference
```

Repeat the same commands with `--device cpu` for a matched-source control. The
reference in these original commands is constructed from the clipped-zero temperature, over the full
domain, using 48 energy-Krylov steps and alternating low/high Ritz selection.
It uses no inactive masks or solved quadratic directions. Construction remains
on the CPU and is reported separately, alongside the reconstruction and complete
quadratic time; it belongs in any construction-inclusive comparison. The saved
reference and its checksum identify the actual basis. Numerical rank loss adds
no replacement directions.

The [projected repair and profiling example](projected_repair.md) adds explicit
initialization from the optimizer's declared snapshot, grouped independent GPU
solves, and CPU-sparse/GPU-coarse execution. These options identify separate
procedures; they do not relabel the earlier timing records.

The direction target is distinct from the unchanged weighted quadratic KKT
criterion stored in the trace. Original complete nonlinear residual and KKT
requirements remain unchanged. The output preserves every attempt and contains
30-second iteration checkpoints, independently verified linear residuals, KKT
components, and named device-storage and allocator measurements. Those memory
values are not total process peak memory. This diagnostic alone establishes
neither nonlinear optimality nor a complete-optimization speedup.

Retain both outcomes in the comparison, including a missing or failed attempt:

```sh
uv run --no-sync python -m deflation_example.coupled_qp_cuda_report \
  --records OUTPUT/cuda-frozen/record.json OUTPUT/cuda-reference/record.json \
  --ranks 0 16 --output OUTPUT/cuda-summary
```
