# Coupled thermal optimization

This example is under verification. It adds temperature-dependent buoyancy to
the axisymmetric transformer model and recomputes the distributed heat source.
The prescribed-flow timing records and their numerical implementations remain
unchanged. No coupled application speedup has been established.

The trial variable is the complete temperature trajectory. Each evaluation
solves momentum and continuity, then recovers the source from the thermal
equations. Sparse factorizations of the converged momentum Jacobians supply
tangent and transpose solves. The positive-definite Gauss--Newton model contains
the tracking weights and the weighted normal product of the source Jacobian.
Both temporal couplings and velocity-dependent streamline diffusion are included.
The stabilization formula is differentiable within each branch; exact branch
switches are rejected during linearization.

## Verification

From the repository root, run on an allocated compute node:

```bash
uv sync --frozen
uv run pytest tests/test_coupled_derivatives.py tests/test_coupled_optimizer.py tests/test_coupled_flow_solve.py
```

These tests check the steady and nonuniform-time-step tangent, transpose,
objective gradient, positive definiteness, feedback-disabled limit, and failed
flow evaluations. They verify discrete derivatives on small problems. Application
resolution, nonlinear optimality, and solver timings require separate evidence.

## Operating-point pilots

```bash
uv run python -m deflation_example.coupled_pilot inlet_factor=1.0 output=runs/inlet-1
```

The declared inlet factors are 1, 0.5, 0.25, and 0.125, in that order. Each run
retains viscosity-continuation attempts, the physical steady residuals, and mass
balance. A converged isothermal baseline is a prerequisite for further checks.
Selection also requires coupled convergence and resolution checks, independently
of optimizer performance. Output directories must be new.

Oil momentum properties use the correlations in
[Li et al., High Voltage 9 (2024), 230–240](https://doi.org/10.1049/hve2.12345).
Thermal properties retain the declared transformer coefficients. These inputs
define a computational model, with prescribed inlet conditions and frozen
properties apart from the Boussinesq forcing.

## Coupled derivative and optimization pilots

After independent baseline verification:

```bash
uv run python -m deflation_example.coupled_optimize baseline_directory=runs/inlet-1 mode=derivatives output=runs/derivatives
uv run python -m deflation_example.coupled_optimize baseline_directory=runs/inlet-1 mode=optimize output=runs/steady-control
```

The second command requires the application derivative checks to pass first.
Use `transient=true slabs=16` for a complete 600 s trajectory. These commands
write pilots, not final repeated timing populations. The
[declared study](protocol.json) identifies the final targets, temperature bounds,
comparators, tolerances and resolution checks.

The optional `flow_continuation=true` setting introduces a residual load that
makes the initial flow an equilibrium, then removes this artificial load in
verified stages. Its final test uses the original momentum equations. Every
failed stage remains visible. `grad_div_scale` in the baseline pilot defines a
separate momentum discretization and must be included in comparisons and source
identification.

## Numerical-policy pilots

The default nonlinear policy remains damped Gauss--Newton with halving
backtracking. `backtracking=quadratic` uses safeguarded quadratic interpolation
after an unsuccessful trial. The separate `secant_memory=10` pilot augments the
current Gauss--Newton operator with ten damped BFGS secants. Powell damping
maintains positive curvature in exact arithmetic. These settings change the
nonlinear model or line search and must be stated alongside their measurements.
They retain the original-residual and nonlinear KKT criteria. They have not been
selected for the final comparison.

The optional CUDA backend uses the same CPU-factored momentum derivatives and
independently verifies the original system on the CPU:

```bash
uv sync --frozen --extra coupled-gpu
uv run --extra coupled-gpu pytest tests/test_coupled_cuda.py -q
uv run --extra coupled-gpu python -m deflation_example.coupled_optimize baseline_directory=runs/inlet-1 device=cuda output=runs/cuda-pilot
```

It requires Linux, a CUDA-compatible GPU and the pinned CuPy dependency.
Triangular analyses are retained while the momentum Jacobian is unchanged;
vector solves use cuSPARSE SpSV and column blocks use SpSM. Each plan retains
its descriptors and analysis workspace. The vector binding uses the cuSPARSE
library already loaded by CuPy, avoiding incompatible opaque descriptors from
different library versions. Plans are confined to their original CUDA stream.
The basis and coarse factorization are rebuilt for each inactive system.

The diagnostic scripts in this directory have `--help` interfaces:

- `check_saved_gradient.py` tests an objective gradient at a saved nonlinear iterate.
- `momentum_spectrum.py` examines selected linearized momentum eigenvalues. It
  does not certify the stability of the complete spectrum or the physical flow.
- `benchmark_actions.py` compares CPU and CUDA applications of the same fixed
  normal operator, with two warmups and five timings at each block size.

Operator timings exclude complete optimization and must remain separate from
its cost comparisons. All pilot failures and iteration caps remain recorded.
