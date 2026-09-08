# What the example solves

With interior-node state `y`, control `u`, desired state `d` and upper bound
`psi`, the objective is

`j(y,u) = 0.5 ||y-d||² + 0.5 alpha ||u||²`, with `A y = u` and `y <= psi`.

Eliminating the control gives `H = I + alpha A.T A`. Its reduced objective
is `0.5 y.T H y - d.T y`; the omitted constant `0.5 d.T d` does not affect
the optimum or objective differences within one query. JSON identifies this
quantity as `objective_reduced`.

For upper bounds the KKT equations are `H y - d + lambda = 0`,
`lambda >= 0`, `y <= psi`, and `lambda * (psi-y) = 0`. PDAS imposes active
state values, solves the inactive equations and updates the set using signed
multipliers. A wrongly active node can leave the set. Acceptance checks all
KKT components, not just an unchanged mask.

On inactive indices `I`, with active indices `J`, the linear equations are
`B y_I = b`, where `B = H[I,I]` and `b = d[I] - H[I,J] psi[J]`.
The full normal product is essential: `H[I,I]` is generally different from
`I + alpha A[I,I].T A[I,I]`.

## Reference-space deflation

Construct full-domain modes `Phi` once per sequence. For every inactive solve,
restrict their rows and use a rank-revealing SVD to obtain `Z`. Define
`E = Z.T B Z`, `Q = Z E^{-1} Z.T`, and `P = I - Q B`.
The implementation uses a Cholesky factorization, not an explicit inverse.

The initial correction is `x <- x + Q (b-Bx)`. Subsequent directions use
`P diag(B)^{-1} r`. This is the coarse-corrected projected-direction form
associated with exact-coarse A-DEF2. It returns a solution of the original
equations, not an uncorrected unknown from a projected system.

Singular values below `1e-12` times the largest are discarded. Coarse condition
numbers above `1e10` trigger an explicitly reported Jacobi fallback. These
safeguards address arithmetic, not the quality of the selected directions.
Every accepted kernel has a freshly recomputed relative residual at most
`1e-10`. A zero right-hand side uses the corresponding absolute criterion.
PDAS uses a normalized KKT tolerance of `1e-9`, with an absolute inner budget
of `1e-10` as well as a relative inner limit of `1e-10`.

## Coordinates and scope

Coordinates use `meshgrid(indexing="ij")` and C-order flattening. In 2D the
fast axis is `x2`. The thermal preset uses the discrete transport
`C = diag(v1) D2 + diag(v2) D1`, where
`v1 = 100 sin(pi*x1) cos(pi*x2)` and
`v2 = -100 cos(pi*x1) sin(pi*x2)`. Thus `C` is defined by this expression,
not by exchanging the two derivative matrices.

In CHT the solid is `x3 < 0.5`, with conductivity 100; the fluid conductivity
is 1. Harmonic face values define diffusion. Fluid transport is in the `x1`
direction with velocity `50 * 4*x3*(1-x3)`. Homogeneous Dirichlet conditions
apply on the outer boundary. The reference ignores this heterogeneity and
uses the homogeneous Laplacian.

The GPU option accelerates linear kernels only. Complete PDAS timings are
CPU measurements. Kernel comparisons use the same mask, operator, load and
acceptance criterion. Reference, problem assembly and bound calibration are
separate timing intervals. Timing differences from a single small run are not
statistical performance estimates.

The package contains no trained model, snapshot archive or external dataset.
It demonstrates a fixed reference policy on Cartesian grids, not industrial
scalability or a guarantee for arbitrary operator changes.
