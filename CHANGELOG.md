# Changelog

## Diagnostic tutorials, version 1

- Add runnable spectral, accuracy/stopping and cost tutorials with fixed protocols.
- Reconstruct controlled diffusion bounds and retain every secondary method and status.
- Export source/environment identifiers, raw-input hashes and compact CSV summaries.
- Preserve the benchmark solver implementations and all original measurement records.

## 0.3.0

- Extend complete CHT sequences to sixteen distinct targets on two grids.
- Match outer active-set and inner-solution warm starts across GPU solvers.
- Record complete-sequence cost components, failures and per-target KKT checks.
- Support right-hand-side-relative AmgX stopping with nonzero initial guesses.
- Preserve all three complete-sequence stopping protocols and their failures.
- Add rank-matched previous-system Krylov recycling and a controlled PDE angle bound.
- Release matrix-free ranks 0, 20 and 200 through 128 cubed with independent CPU and GPU residuals.
- Validate sequence hashes, populations, acceptance, resource policies and nonoverlapping timing components before reporting.
- Measure fresh-process imports, GPU initialization and shutdown separately, with preparation totals that avoid double counting.

## 0.2.0

- Add a self-contained GPU-QR/AmgX benchmark with nonoverlapping wall-time components and all repetitions.
- Compare recreated and persistent AmgX configuration/resources while rebuilding every hierarchy.
- Include per-call cleanup and independent original residuals in both timing boundaries.
- Add complete CHT PDAS sequence, restriction-bound and scaled/unscaled Ritz controls.
- Export checked cumulative instance medians, raw repetition CSV and the timing figure.
- Clarify quarter-turn repetitions in the default 2D examples.

## 0.1.2

- Publish tested distributions as release assets without a duplicate CI artifact upload.
- Retain all CPU, installation, style and publication checks as release gates.
- Disable Hydra sweep mode before it can write machine-specific configuration data.
- Numerical implementations are unchanged from 0.1.1.

## 0.1.1

- Fix state truncation when PDAS receives integer bounds.
- Prevent false convergence caused by norm underflow or overflow.
- Recompute failed-solve multipliers and inner residual diagnostics.
- Reject malformed, complex and nonfinite numerical inputs explicitly.
- Separate reporting and interchangeable kernel adapters from orchestration.
- Write reports atomically and independently verify injected kernels.
- Expand publication checks to Git history, credential formats and unsafe archives.

## 0.1.0

- Three self-contained quadratic control presets with a reusable reference space.
- Genuine PDAS and coarse-corrected deflated CG with independent accuracy checks.
- Locked uv installation and Hydra configuration.
- Optional float64 CUDA linear kernels and plots.
- Numerical records, public-facing documentation and automated CPU tests.
