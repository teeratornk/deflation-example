# Changelog

## Unreleased coupled-optimization verification

- Add temperature-based elimination with exact discrete buoyancy, transport and stabilization derivatives.
- Verify coupled steady and nonuniform-step transient tangents, adjoints and recovered controls.
- Add two-sided active-set Gauss--Newton optimization with independent nonlinear optimality checks.
- Compare the three existing CG policies on small nonlinear verification problems.
- Provide operating-point and derivative pilots with retained flow failures and optional residual-load continuation.
- Add a smooth 60-second target startup for the new 600-second trajectory study, with unchanged later targets and separate original-target pilots.
- Preserve the selected target in spatial and temporal replays; validate matched complete-sequence summaries and plot saved physical fields.
- Keep application speedup claims pending coupled convergence and resolution studies; preserve all prescribed-flow numerical sources and timings.

## 0.6.2

- Enlarge transformer panels, share spatial axes and separate target labels from physical time.
- Plot complete-sequence costs at integer completed-target counts, including initialization and the final cleanup-inclusive endpoint.
- Document the explicit residual-correction recommendation, local error-equation acceptance and final original-system acceptance.
- Test scalar-failure continuation, locally rejected improving corrections and preservation of all timing repetitions.
- Preserve the numerical kernels, frozen benchmark presets and all primary and follow-up timing records.

## 0.6.1

- Keep the complete normalized-temperature labels inside the saved body-fitted figure.
- Preserve the numerical implementation and the frozen correction-study records from 0.6.0.

## 0.6.0

- Diagnose projected-CG scalar breakdown with recurrence, original residual, curvature and coarse-orthogonality measurements.
- Add bounded residual correction within the original iteration and final-accuracy limits.
- Retain AmgX resources when changing the error-equation stopping target and remaining iteration budget.
- Provide a four-target finer-transformer correction preset and formula-checked comparison summaries.
- Separate temperature and active-set panels and identify the fastest tested alternative in body-fitted figures.
- Preserve the original complete-sequence records and their numerical protocols.

## 0.5.1

- Apply the configured source formatting; the formatted Python syntax trees are unchanged.
- Exercise the installed wheel's mesh trajectory, transfer replay, plotting, and independent verification in CI.
- Preserve all frozen numerical records and their exact source identifiers.

## 0.5.0

- Add self-contained axisymmetric transformer and three-dimensional bore-in-block meshes.
- Assemble weighted P1 steady and complete transient thermal optimization with explicit material, source, boundary, and time conventions.
- Prolong coarse spatial modes and construct mode-dependent space–time references.
- Retain reference factors and recycling vectors on the GPU with matched GPU-QR processing.
- Verify independent bounded least-squares solutions, temporal coupling, mesh integration, and control recovery.
- Record independently repeated complete costs, actual cumulative timestamps, accepted ranks, failures, and comparable process memory.
- Reconstruct target and constraint figures from numerical mesh inputs and recorded masks.
- Audit mesh transfer replays and generate comparison tables from checked raw records.
- Separate imported source version from installed-distribution metadata in environment reports.

## 0.4.0

- Add complete prescribed-flow steady and transient CHT optimization with independent trajectory verification.
- Compare rank-zero Jacobi-CG, fixed-reference deflation, Jacobi-scaled recycling that retains existing coarse vectors, and persistent-resource AmgX.
- Construct compact mode-dependent space–time references and retain the tensor-product construction control.
- Replay identical optimization traces to compare full-reference restriction with sequential zero-extension transfer.
- Measure coarse energy removal, small deflated spectra, complete sequence costs and comparable sampled process memory.
- Freeze spatial, temporal, rank and cold-start populations with every declared outcome retained.
- Reconstruct the controlled CHT condition-number bound before generating its table.
- Generate figures and tables from checked raw records and version the complete study data separately from the installable package.

## Unreleased robustness checks

- Fix the CUDA demo adapter to accept the GPU solver's detailed timing record.
- Accept array-like reference bases consistently in CPU and GPU solvers.
- Reject complex or nonscalar target angles and noninteger spatial dimensions.
- Repeat every default CPU preset and check the saved fields against the equations.
- Install wheels with locked dependencies in CI and exercise all presets and tutorials outside the checkout.
- Preserve the published releases and original numerical benchmark records.

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
