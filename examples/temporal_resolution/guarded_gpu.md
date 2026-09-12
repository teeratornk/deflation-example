# Guarded residual correction on the finer transformer

This follow-up evaluates the previously recommended initial-residual guard on
a GPU. It preserves the four-target finer-transformer problems and final
accuracy requirements of the correction study. It is separate from the six
primary body-fitted comparisons and from the changed-transport temporal pilot.

Use source `0b3b3f06a535f515256932c0dbf27537c242744d`, which introduced the
initial guard. The measured correction study remains attached to source
`926b37c40935fbbe7134a3ab5f87ebdfddb68837`. A new measurement does not change
that source mapping.

```bash
git checkout --detach 0b3b3f06a535f515256932c0dbf27537c242744d
uv sync --frozen --extra study --extra gpu
# Install the documented AmgX library and Python wrapper for the selected CUDA environment.
uv run --no-sync pytest tests/test_refinement.py tests/test_study_solvers.py tests/test_mesh_gpu.py
uv run --no-sync python -m deflation_example.benchmark_mesh \
  --config-name mesh_refinement_transformer transient=false horizon=0.1 \
  threads=4 timeout_seconds=3000 output=runs/guarded-refinement-steady
uv run --no-sync python -m deflation_example.benchmark_mesh \
  --config-name mesh_refinement_transformer transient=true \
  horizon=0.000048444553995823255 threads=4 timeout_seconds=3000 \
  output=runs/guarded-refinement-transient
```

The preset fixes level 1, rank 100, four targets, five independent complete
sequence repetitions, and matched outer/inner warm starts. All four methods
participate: Jacobi-CG, fixed-reference deflation, recycling and persistent-resource
AmgX. The original advective thermal operator is unchanged. The original-system
residual threshold is `1e-10`; the five KKT components must meet `1e-8`. At most
four error solves share the 50,000-iteration budget. Existing internal margins
and all setup, transfer, verification and cleanup costs remain in the records.
The existing GPU installation instructions are in `docs/gpu-benchmark.md`.
After installing the native binding, `--no-sync` keeps uv from removing it.

Retain every sequence and termination status. Report this guarded follow-up
separately, with its own hardware and source identifiers; do not pool its timing
samples with either frozen study. The result tests this procedure on these two
four-target problems and supplies no general robustness guarantee.
