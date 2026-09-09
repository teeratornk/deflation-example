# Changelog

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
