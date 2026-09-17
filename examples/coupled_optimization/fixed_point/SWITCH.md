# Streamline-coefficient switch diagnostic

This example examines a failed backward-Euler step saved by
`continue_repaired.py`. It keeps the source field, preceding state, spatial
mesh, time interval and original coupled residual unchanged. It is a local
forward-solver study, separate from optimization timing and physical resolution.

The streamline-diffusion coefficient is the minimum of an advection-dependent
expression and a diffusion-dependent expression. Their values agree at a
switch, but their derivatives generally differ. The optimization derivative
continues to use the strict policy, which rejects this switch. The optional
`active_newton` forward procedure differentiates the selected branch and uses
the diffusion branch at an exact tie. It neither smooths the coefficient nor
changes the equations. This selection alone supplies no convergence theorem.
The established nonsmooth-Newton framework requires additional assumptions;
see [Qi and Sun (1993), A nonsmooth version of Newton's method](https://doi.org/10.1007/BF01581275).

Run the following from this checkout, using the baseline and immutable outputs
produced by the fixed-point and continuation examples:

```sh
uv sync --locked --extra study --extra plot
uv run pytest tests/test_coupled_switch.py tests/test_switch_diagnostic.py
uv run python examples/coupled_optimization/fixed_point/switch_diagnostic.py \
  --baseline BASELINE --fine-baseline FINE_BASELINE \
  --optimization OPTIMIZATION --trajectory ORIGINAL_TRAJECTORY \
  --repair VERIFIED_REPAIR --continuation CONTINUATION \
  --policy inspect --output runs/switch-inspection
```

The inspection records the original equation residuals, distance to the
coefficient switch, selected branches, linear-solve residual and directional
finite differences. The trial table reports the actual residual decrease and
the number of cells whose selected branch changes. Small finite-difference
steps can suffer cancellation; their differences are diagnostics rather than
an accuracy certificate.

For a matched local comparison, repeat the command with `--policy active_newton`
and `--policy coupled_anderson`, using separate output directories. Newton uses
at most 100 updates and 40 backtracking trials per update. Complete-field
Anderson uses depth five, relaxation 0.5 and at most 300 updates. Both require
independently evaluated momentum, continuity and thermal residuals at most
`1e-12`, and relative mass and energy defects at most `1e-6`. The caps differ
because the local updates perform different operations; compare elapsed work
and termination status, rather than equating their update counts.

The loader verifies the original input identities, correction and continuation
hashes, field dimensions and time ordering. It independently re-evaluates the
preceding time-step equations. Returned fields are saved even when the solver
stagnates or reaches its cap. The strict Newton procedure now returns
`newton_stabilization_switch` with the retained fields and their residuals,
instead of losing those fields through an uncaught branch exception. Frozen
earlier records remain associated with their original implementation.

Every attempted policy belongs in the comparison, including nonconvergence.
Local solution costs exclude the earlier trajectory and failed attempt; they
cannot be presented as complete optimization times or deflation speedups.
