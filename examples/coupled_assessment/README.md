# Two-way-coupled forward assessment

This example prepares a forward assessment of saved transformer controls.
The prescribed-flow optimization and its published solver comparisons remain
unchanged. The implementation freezes the control selection, verifies the
original forward equations, and solves coupled axisymmetric momentum and
thermal equations. Application assessment and resolution checks are in progress.

## Freeze the controls

Use the saved `reference-0` outputs from the steady and four-slab transformer
populations in `mesh-cht-data-v1`. Each source directory must contain
`record.json` and all 16 `fields-N.npz` files. The
[body-fitted reproduction instructions](../../docs/mesh-showcases.md) describe
the optimization runs and their numerical source versions. The selection
requires source `dc89ae8518ff6a975eadf2f6f3bae1acbd17b83b`; runs from another
version require a separately declared assessment population.

Run from the repository root:

```bash
uv run --locked python -m deflation_example.assessment_controls \
  --protocol examples/coupled_assessment/protocol.json \
  --source INPUT_STEADY/reference-0 --population steady \
  --output runs/assessment-controls/steady
uv run --locked python -m deflation_example.assessment_controls \
  --protocol examples/coupled_assessment/protocol.json \
  --source INPUT_TRANSIENT/reference-0 --population transient_four_slabs \
  --output runs/assessment-controls/transient
```

`INPUT_STEADY` and `INPUT_TRANSIENT` denote the corresponding saved output
directories. The command requires a new output directory. It records every
candidate score and preserves three distinct cases selected in this order:

1. The target nearest the central query parameter.
2. The remaining target with the largest final active fraction.
3. The remaining target with the largest weighted mean-square source control.

Ties use the smallest query index. Selection uses the original optimization
records alone. All subsequent forward outcomes belong to the assessment,
including nonconvergence and unresolved temperature violations. The case
with the largest active fraction is described by its measured fraction.

The numeric bundles contain the saved source, temperature, target, active
set, initial condition and slab endpoints. Each bundle also contains the
same source in W/m³. SHA-256 identifiers cover the source arrays, input
bundles and original records. An independent steady solve or sequential
forward substitution checks the imported source against the saved state.

The source remains piecewise constant on each original time slab. On a
refined spatial mesh, P1 interpolation evaluates the original source field.
The inlet source values use a declared zero extension. Resolution studies
must report the source-integral change associated with nodal mass lumping
and the corresponding frozen-flow temperature difference.

## Verify the momentum operators

```bash
uv run --locked pytest tests/test_assessment_controls.py tests/test_axisymmetric_flow.py \
  tests/test_oil_properties.py tests/test_coupled_forward.py tests/test_assess_transformer.py
```

The flow module uses P2 velocity and P1 kinematic pressure on fluid triangles.
Its weak form includes the cylindrical divergence, radial viscous term and
axisymmetric integration measure. Its positive triangle quadrature integrates
polynomials through degree eight; the axisymmetric convection products have
degree six. Tests check an affine manufactured
Navier–Stokes field, analytical annular Poiseuille flow under refinement,
boundary volume flux, transient momentum storage, and the buoyancy sign.
Both flow iterations rebuild the convection matrix at the updated velocity
and check the full continuity residual. A finite-difference test verifies the
Newton Jacobian. Steady and transient manufactured solutions check the coupled
equations, fixed source, buoyancy, storage and independently integrated boundary
advection. The coupled stopping test recomputes momentum after the temperature
update.

These tests establish discrete solver correctness on their specified problems.
The application assessment also requires conservation and resolution checks.

## Published momentum properties

The assessment uses the mineral-oil correlations in Li et al.,
*Thermofluidic investigations of oil natural transformer: Closed-loop modelling
and experimental validation*, High Voltage 9 (2024), 230–240,
[Section 3.2, Equations (5)–(8)](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/hve2.12345).
The correlations use kelvin and cover 273.15–373.15 K:

\[
\rho=1093-0.826T+3.612\times10^{-4}T^2,\qquad
\mu=7.863\times10^{-5}\exp\!\left(\frac{632}{T-176}\right).
\]

Density has units kg/m³ and dynamic viscosity has units Pa s. The code derives
the expansion coefficient from \(\beta=-\rho'(T)/\rho(T)\). At 75 °C, the
values are 849.209 kg/m³, 0.00309014 Pa s and 0.000676508 K⁻¹.
The gravity vector is `(0, -9.81)` m/s² in radial–axial coordinates.
Hydrostatic lifting uses the inlet temperature, 341.3 K, as its buoyancy
reference. The material evaluation temperature is 348.15 K.

The nominal calculation freezes these momentum properties at 75 °C.
Two additional calculations use constant viscosities evaluated at the inlet
temperature (68.15 °C) and bound temperature (84.15 °C). They retain the same
density and expansion coefficient. Dynamic viscosity spans 0.00256750–0.00359790
Pa s. All three settings preserve the original conductivities, volumetric
thermal capacities, sources and thermal boundary conditions.

This experiment assesses buoyancy and viscosity sensitivity using published
parameters. The original oil identity remains unspecified. The new citation
applies to the assessment parameters; the earlier thermal coefficients retain
their original definitions. Application temperatures outside the published
domain receive an explicit termination label.

The original Stokes calculation uses a simplified weighted scalar Laplacian
for each velocity component. The new discretization includes the radial
vector-Laplacian term and convective inertia. A computed-flow baseline with
feedback disabled must therefore be measured separately from the original
prescribed field. The baseline uses the same thermal inputs and saved
controls as the subsequent feedback-enabled calculation.

The new momentum equations use fluid velocity \(\mathbf v\) and kinematic
pressure \(\pi\):

\[
\partial_t\mathbf v+(\mathbf v\cdot\nabla)\mathbf v
-\nu\Delta_{\mathrm{axi}}\mathbf v+\nabla\pi
=-\beta(T-T_{\mathrm{in}})\mathbf g,\qquad
\nabla_{\mathrm{axi}}\cdot\mathbf v=0.
\]

The steady calculation omits the time derivative. The radial component of the
vector Laplacian includes \(-v_r/r^2\). The thermal equation retains solid
conduction, fluid advection, the original volumetric capacities and the saved
distributed source \(u^*\). Temperature and normal conductive flux are
continuous at the conforming transformer interfaces. The transformer has no
additional interface source.

The flow initialization pilots compare adaptive viscosity continuation and
pseudo-transient continuation. Both require an independently verified steady
solution at the declared physical viscosity. Intermediate continuation states
remain initialization data. Newton uses a backtracking line search and records
linear residuals, damping and explicit stagnation or iteration-limit outcomes.
The final initialization policy remains subject to the application checks.

The initialization comparison also checks the advective and skew-symmetric
convection forms. The skew form adds
\(\tfrac12(\nabla_{\mathrm{axi}}\cdot\mathbf v)\mathbf v\), which vanishes
for an incompressible continuous velocity and supplies the discrete
kinetic-energy flux identity. Its role with weakly enforced incompressibility
is discussed by [Charnyi et al. (2017)](https://doi.org/10.1016/j.jcp.2017.02.039).
`convection_form=skew` selects this form. The diagnostic script
`diagnose_flow.py` reports the divergence contribution, viscous dissipation
and an independently integrated boundary kinetic-energy flux.

Small density variation supports the density approximation in buoyancy.
The frozen viscosity and other transport coefficients require their own
check. Roche's [Boussinesq analysis](https://arxiv.org/abs/0710.3497) explicitly
assumes constant properties before deriving its density and stratification
criteria. A buoyancy-only calculation with frozen viscosity can serve as a
specified sensitivity study. Its interpretation requires a decision about
the observed property variation and the applicable flow regime.

## Run the fixed-control assessment

The original [selection protocol](protocol.json) fixes the six controls.
The [forward protocol](forward-protocol.json) declares the material settings,
comparison groups, conservation criteria and resolution targets.

```bash
uv run --locked python -m deflation_example.assess_transformer \
  controls_directory=runs/assessment-controls/steady \
  output=runs/forward-steady-nominal role=nominal mode=coupled
uv run --locked python -m deflation_example.assess_transformer \
  controls_directory=runs/assessment-controls/transient \
  output=runs/forward-transient-nominal role=nominal mode=coupled time_step_s=37.5
```

Use `mode=prescribed` for the original velocity, `mode=frozen` for the computed
baseline with thermal feedback disabled, and `mode=coupled` for the coupled
calculation. Use `viscosity=low` or `viscosity=high` for the declared sensitivity
values. The three roles are `nominal`, `active` and `demanding`. Each command
requires a new output directory. `level=1` and `level=2` apply uniform nested
spatial refinement. The time step must divide every original control slab.
The stored source on `(t[n-1], t[n]]` applies to every finer step in that
interval, including its right endpoint.

After a verified baseline run, `baseline_directory=PATH` loads its saved flow
for another selected control. The command checks the material settings, mesh
identifiers, field checksum and current steady equations. Baseline construction
time and the cost of loading and rechecking it are recorded separately. A
numerical failure produces a failure record and exit code 2. Intermediate
baseline fields and iteration diagnostics support inspection of interrupted
initialization attempts.
`baseline_restart_directory=PATH` resumes a viscosity-continuation checkpoint
after rechecking its equations at the recorded intermediate viscosity.
Resumed initialization still has to reach the declared physical viscosity.
Its record separates additional elapsed time from previously recorded
initialization time; missing earlier timing remains explicit.

Each accepted coupled step satisfies independently recomputed momentum,
continuity and thermal residuals of at most 10⁻⁸, a relative boundary mass
imbalance of at most 10⁻⁶, and a relative discrete energy defect of at most
10⁻⁶. The energy check includes transient storage, independently integrated
boundary advection, fixed control, background heating and discrete Dirichlet
reactions. Those reactions include streamline stabilization. Resolution of
physical conductive fluxes requires a separate refinement check.

The output contains a source-version record, the authenticated control
identifiers, every completed or failed time step, residual histories and
numeric temperature, velocity, pressure and applied-source fields. Temperature
violations are measured at free P1 nodes and the stated simulation times.
Resolution uncertainty remains pending until the refinement comparisons are
complete. Forward timings remain separate from the optimization comparisons.

The main manuscript will receive an assessment subsection after the matched
calculations and resolution studies establish the accuracy and scope of the
results.
