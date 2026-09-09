# Spectral diagnostics — tutorial version 1

This CPU tutorial computes the angles and bounds used to explain reference-space reuse. Run it from the companion source revision linked by the manuscript:

```sh
uv sync --locked
uv run --locked python -m deflation_example.tutorials spectral --output runs/tutorial-spectral
```

The command writes a [fixed protocol](../../src/deflation_example/conf/tutorial_spectral.json), `results.json` and `spectral.csv` to a new directory. Results record the executing source, numerical environment and hashes of the earlier records used for comparison. The [saved output](../../examples/tutorials/spectral/results.json) records its own source revision. The implementation is [tutorial_spectral.py](../../src/deflation_example/tutorial_spectral.py).

## Examples and definitions

All examples use unscaled Euclidean coordinates. For an orthonormal restricted basis `Z`, define `T = Z.T B Z`, `F = B Z - Z T` and `eta = lambda_(r+1)(B) - lambda_max(T)`. With retained rank `r` and `eta > 0`, the residual bound on the largest sine is `min(1, norm(F, 'fro') / eta)`. The coupling bound additionally estimates the residual using removed-mode coupling and the minimum restricted singular value. The output names the formulas separately. A value of one is a trivial bound; `null` means the required rank or separation condition fails.

| Example | Computation and interpretation |
|---|---|
| Restriction-induced rank loss | Delete a degree of freedom supporting one of two reference vectors. The retained rank becomes one. The requested equal-rank comparison and its bound are undefined; a separately labeled rank-one comparison has angle zero. |
| Repeated-eigenvalue cutoff | Two exact rank-one choices in the lowest eigenspace of `diag(1,1,3)` have angle 90°. Their residuals vanish and separation is zero. Retaining the complete two-dimensional cluster gives angle and bound zero. |
| Eigenvalue crossing | The lowest eigenspaces of `diag(1,2)` and `diag(1.6,1.4)` are orthogonal. The unconditional perturbation/reference-gap quotient is 0.6, below the actual sine of one. The sufficient no-crossing hypothesis fails. |
| Controlled SPD restriction | The eight-dimensional example removes one degree of freedom and varies a rank-one perturbation through eight amplitudes. Every row reports the actual angle, residual bound, coupling-based sufficient bound and separation status. |
| Diffusion restriction | The five-point Dirichlet Laplacian gives `I + 0.001 L.T L`. Grids 4, 6 and 12 use one analytical reference mode and corner widths 1, 2 and 3. All nine masks are prescribed. |

For diffusion at `(n,q)=(4,1)` and `(6,1)`, the coupling bounds are approximately 0.737969 and 0.896046. The tighter residual bounds are 0.721523 and 0.892952. Substitution into the deflated condition-number estimate gives upper bounds above the original condition numbers. These are nontrivial angle-bound examples, without a certified convergence improvement. Other rows retain trivial bounds of one or nonpositive separation.

The earlier [controlled-SPD records](../../examples/benchmarks/structure.json) and [diffusion records](../../examples/benchmarks/extended/controls.json) contain their original numerical source identifiers and environments. The tutorial reconstructs the diffusion operators and checks the derived quantities against those records. The production solver and benchmark timings are unchanged.
