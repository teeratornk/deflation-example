# Run the relocated studies

The repository contains executable examples as well as archived results.
Two routes serve different purposes: regenerating figures and tables from the
published records, and repeating the numerical calculations. A regenerated
timing plot retains its measured timings; it is not a new performance test.

## Regenerate all bundled evidence

Use Python 3.11 or 3.12 and the locked dependencies. No GPU, manuscript checkout,
private file or institutional filesystem is required for this route.

```bash
git clone https://github.com/teeratornk/deflation-example.git
cd deflation-example
git checkout repository-reproduction-v1
uv sync --frozen --extra study --extra plot
uv run --frozen python tools/reproduce_repository_examples.py --output runs/relocated
```

The command verifies archived inputs, checks all 317 relocated supplementary
table rows, regenerates the compact development figure and numerical summary,
rebuilds the detailed temporal plots for all three replay stages, and summarizes
all 40 guarded-solver GPU sequences. The development output also includes all
eight construction/stopping comparisons and all eight original finer-transformer
method/form attempts, including unsuccessful outcomes.

`runs/relocated/reproduction.json` records the command arguments, source version,
exit codes and output checksums. Each command has its own log. All selected
commands run even if one fails; the overall exit code is nonzero if any fail or
time out. Use a new output directory for another run. Existing results are never
overwritten. The default limit is 1800 seconds per command, adjustable with
`--timeout`. These limits concern report generation, not the numerical studies.

The compact temporal generator checks input hashes before loading its frozen
source and verifies that the generated numerical quantities and text match the
archived values. It publishes its output only after successful generation.
Its byte-verification route requires only the Python standard library:

```bash
python3 tools/reproduce_development_studies.py --verify-only
```

## Include the complete Cartesian and body-fitted data releases

The larger record collections have their own public data tags. Download them
beside the code checkout, then supply their paths to the same runner:

```bash
git clone --branch mesh-cht-data-v1 --depth 1 \
  https://github.com/teeratornk/deflation-example.git ../mesh-evidence
git clone --branch reference-policy-data-v1 --depth 1 \
  https://github.com/teeratornk/deflation-example.git ../cartesian-evidence
uv run --frozen python tools/reproduce_repository_examples.py \
  --mesh-records ../mesh-evidence --cartesian-records ../cartesian-evidence \
  --output runs/all-relocated
```

These additional routes verify the data manifests and regenerate the complete
comparison, accuracy, rank, preparation, memory and transfer summaries. They
retain the construction/stopping pilots and original failed attempts. No
optimization is rerun by this reporting command.

## Repeat the numerical calculations

The following guides specify inputs, source checkouts, configurations and run
commands. Use separate code checkouts for different measured procedures.
Numerical outputs must be generated before a command that reads saved controls
or optimization traces. The scalar archives do not substitute for those fields.

| Study | Numerical reproduction route | Requirements |
| --- | --- | --- |
| Construction, stopping and temporal-factor pilots | [Cartesian study and configuration controls](reference-policy-study.md#timing-and-initialization-pilots); complete records in `reference-policy-data-v1` | CPU for the small example; CUDA and AmgX for the four-way GPU comparisons |
| Original finer-transformer attempts and residual correction | [Body-fitted solver procedures](mesh-showcases.md#choose-the-numerical-procedure) and [correction commands](mesh-showcases.md#finer-transformer-residual-correction) | Packaged mesh inputs; CUDA and AmgX for measured GPU procedures |
| Guarded-solver comparison | [Versioned GPU commands](../examples/temporal_resolution/guarded_gpu.md) | Four targets, all four methods and five complete repetitions per form |
| Original and corrected temporal optimizations | [Prepare, run and report commands](../examples/temporal_resolution/README.md#reproduce-on-a-compute-node) | CPU; run all ten cases for each declared transport form |
| Unchanged-source forward refinement | [Initial, extended and final replay commands](../examples/temporal_resolution/README.md#finer-fixed-source-assessment) | Completed optimized fields; all four preselected sources |

The GPU installation requirements, pinned AmgX and PyAMGX sources, and library
configuration are given in the [GPU guide](gpu-benchmark.md). GPU calculations
should run on a compute node. Increasing resolution can require substantial
time and memory. An unsuccessful solve retains its status and cannot supply a
completed-solve speedup. The prescribed-flow temporal checks remain separate
from fully coupled flow–temperature optimization.

## Check the scripts

```bash
uv run --frozen pytest -m 'not gpu'
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

Tests cover output preservation, missing and corrupted records, unsafe paths,
incomplete populations, rendering failures, and invocation outside the checkout.
The continuous-integration workflow also runs the bundled reproduction command
on both supported Python versions. With the GPU extra and a CUDA device, run
`uv run --no-sync pytest -m gpu` for the GPU regression suite. AmgX installation
is separate from the PyTorch GPU extra.
