# Reference-policy evidence, version 1

This data-only tag supports *Reference-Space Reuse for Steady and Transient
State-Constrained Conjugate Heat Transfer*. It belongs to
[deflation-example](https://github.com/teeratornk/deflation-example).
The software release is [v0.4.0](https://github.com/teeratornk/deflation-example/tree/v0.4.0).
Final numerical workers used source cc611292f66025f4d0f96a406c6aeb67799b7d34.
Publication summaries used source 011fa5434bb40d8b82bae5d698c2dc5f6ed20137.

The manifest records relative paths, byte counts and SHA-256 hashes for
589 JSON files in 64 groups. The JSON records occupy about 249 MB before
compression. The release contains no scheduler logs or private execution paths.

## Contents

- final: all 17 declared populations, 300 complete sequences and 4800 target
  solves. Every final sequence passed the original-residual and KKT checks.
  Methods, ranks, starts and repetition counts remain separate.
- pilot: 35 construction, stopping, size, rank, calibration and implementation
  checks. These include failed and capped attempts and the larger transient
  feasibility tests. Pilot populations and timing repetitions differ from
  the final campaign and retain their own protocols and source identifiers.
- diagnostic: independent optimization verification, the controlled CHT
  conditioning study, source traces and five matched transfer replays.
- summary: the checked publication reports for complete performance,
  transfer diagnostics and supporting tables.

Complete sequence time includes construction, assembly, every PDAS update
and solve, transfers, verification and cleanup. Calibration and process
preparation have separate fields. Replay summaries sum per-kernel medians;
they do not represent independently timed complete optimization sequences.
Memory records report sampled host RSS and NVML process allocation, including
library caches, with each observed maximum sampling gap.

## Verify and regenerate

Clone the software and this data tag into adjacent directories:

    git clone --branch v0.4.0 --depth 1 https://github.com/teeratornk/deflation-example.git deflation-code
    git clone --branch reference-policy-data-v1 --depth 1 https://github.com/teeratornk/deflation-example.git deflation-evidence
    cd deflation-code
    uv sync --locked --extra plot --extra study
    uv run --locked --extra plot --extra study python -m deflation_example.benchmark_evidence --records ../deflation-evidence --output runs/submission

The command verifies every byte hash, validates final populations and
acceptance, reconstructs the controlled matrices, and regenerates the
submitted figures and tables. It uses a CPU and performs no new GPU timing.
Omit the output option to verify hashes alone. The
[study guide](https://github.com/teeratornk/deflation-example/blob/v0.4.0/docs/reference-policy-study.md)
also gives full GPU rerun commands, configuration and dependency information.

The final campaign imports the steady calibration from
pilot/steady24-moderate-pilot-v1/results.json and the transient calibration
from pilot/transient12-moderate-pilot-v1/results.json. It preserves each
physical bound across refinement. New calibrations have new measured costs;
the recorded costs reproduce the published preparation-inclusive accounting.

The summary records preserve numerical input hashes and generator versions.
PDF metadata and rendering can vary across systems. Regeneration checks
the numerical definitions and plotted values against the frozen inputs.
