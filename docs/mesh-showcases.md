# Optimization on body-fitted thermal meshes

The mesh examples extend the Cartesian examples with a 2D axisymmetric
transformer and a 3D gas bore inside a conducting metal block. Each query
optimizes temperature and a distributed thermal source. The transient example
optimizes the complete trajectory, with an upper temperature bound at every
positive time level. Velocity and material properties remain fixed during
optimization.

## Run the examples

The small engine example runs on a CPU and uses the packaged mesh:

```bash
uv run --locked python -m deflation_example.benchmark_mesh
uv run --locked python -m deflation_example.benchmark_mesh transient=true output=runs/engine-transient
```

The transformer has 10,849 nodes. Its prescribed through-flow produces a
substantially more difficult thermal operator. Use its separate parameter
preset and retain the complete acceptance report:

```bash
uv run --locked python -m deflation_example.benchmark_mesh --config-name mesh_transformer output=runs/transformer
```

`level=1` subdivides each triangle into four or each tetrahedron into eight.
The mesh hierarchy preserves the material interfaces, baffle thickness, and
boundary labels. The reference uses the packaged coarse mesh and P1
prolongation. `rank` denotes the total deployed candidate count, including
space and time together in transient runs. Numerical rank loss is reported.

Use `spatial_reference=diffusion` for generalized diffusion modes or
`spatial_reference=scaled_schur` for modes of the coarse Jacobi-scaled reduced
operator. The latter includes the prescribed transport. Temporal construction
uses `construction=mode_dependent` or `construction=tensor`.

The GPU comparison requires the existing optional PyTorch and native AmgX
installation described in the companion's GPU instructions:

```bash
uv run --locked --extra gpu --extra study python -m deflation_example.benchmark_mesh \
  device=cuda 'methods=[jacobi,reference,recycling,amgx]' \
  level=1 rank=100 targets=16 repeats=5 monitor_memory=true phase=final \
  output=runs/engine-gpu
```

Every method and repetition runs in a fresh process. The four-way GPU workers
perform the same small PyTorch and AmgX warmup. AmgX configuration and resources
persist within the sequence; its hierarchy is constructed for each inactive
matrix. Its native absolute stopping target is
`amgx_factor * rtol * norm(rhs)`. CG uses `cg_factor * rtol` internally. The
default factors are both 0.1. Every method must satisfy the separate original
residual and KKT acceptance checks.

The complete Hessian is applied through its factors. The default GPU path
assembles each inactive Hessian for the shared sparse GPU solver interface.
`matrix_free_inner=true` supports CPU Jacobi, reference, and recycling solves
without assembling the inactive Hessian. These are distinct implementations
and their timings must remain identifiable.

## Discretization and units

The temperature equation is

\[
K y=M_\ell u+f.
\]

Here, `K` contains diffusion, prescribed transport, and the declared streamline
diffusion. The positive lumped mass `M_l` defines both tracking and control
weights. Dirichlet temperature values are lifted before elimination. The
reduced operator is

\[
H=W+\alpha A^\top W A,\qquad A=M_\ell^{-1}K.
\]

Restriction retains every row of `A`. The implementation forms `A[:, I]`
when constructing an inactive operator. Backward Euler replaces `A` with its
coupled block temporal operator and uses right-endpoint time weights. Recovery
uses this same equation and its exact transpose.

The transformer uses `(r,z)` coordinates and the measure `2*pi*r dr dz`.
Conductivity is anisotropic in the winding. Oil properties are frozen at
323.15 K. Temperature represents excess above the inlet temperature, scaled
by 20 K. The input manifest records source, capacity, velocity, and time units.
The quadratic Stokes field is integrated with degree-five triangle quadrature
and evaluated exactly on each refined child element.

The engine follows the documented bore dimensions, conductivity ratio of
1000, inward interface heat flux, and fixed outer coolant temperature. Its
prescribed circulation is independent of temperature. Time uses the fluid
diffusion scale; both materials have unit dimensionless thermal capacity.
The temperature scale is `q*L/k_s`, with the numerical physical parameters in
the input manifest. Desired profiles and bounds define idealized tracking
problems; they do not represent measured engine operating trajectories.

Both examples use conforming P1 temperatures and nodal upper bounds. A constant
upper bound at every node also bounds the P1 field throughout each element.
The scalar control acts throughout the optimized spatial domain.

## Verification and output

```bash
uv run --locked python -m deflation_example.validate_mesh --output runs/mesh-verification
uv run --locked --extra plot python -m deflation_example.mesh_presentation \
  --records runs/engine-transient --output runs/engine-figures --fields
```

The independent verification compares PDAS with bounded-variable least squares
on the weighted tracking/control factor. It covers the complete coarse engine
mesh and a transformer inlet-region subproblem, including unequal time steps
and nonzero initial temperature. Unit tests additionally check integration,
affine fields, refinement, the transpose, control recovery, and restriction.

Each sequence writes its resolved configuration, source identifiers, input
hashes, timing components, numerical ranks, active masks, and acceptance
components. Optional field files store desired and optimized temperatures,
controls, multipliers, and active degrees of freedom. Failed sequences retain
their outcomes. Summary generation checks populations, distinct target hashes,
timing sums, and accuracy before reporting accepted timings.

Mesh loading, refinement, assembly, reference construction, every PDAS update,
inner solve, transfer, verification, and solver cleanup belong to complete
sequence cost. Common initialization and finalization are recorded separately.
Field serialization and plotting occur after the timed sequence. Memory uses
sampled process RSS and NVML process allocation when enabled.

The optional `mesh` dependency regenerates the engine mesh. It is unnecessary
for running the packaged inputs:

```bash
uv run --locked --extra mesh python -m deflation_example.generate_engine_mesh \
  --output runs/regenerated-engine --size 0.12
```

Gmsh may require platform OpenGL system libraries. The packaged examples have
no Gmsh runtime dependency and require no access to AI4PDEs or application data.
