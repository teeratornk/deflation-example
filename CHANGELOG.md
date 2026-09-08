# Changelog

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
