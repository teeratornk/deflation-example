# Corrected coupled optimization

The [declared design](protocol.json) separates complete optimization pilots from
the subsequent resolution and timing comparisons. The thermal formulation
includes cylindrical diffusion in the strong residual and integrates the P2
velocity in trial advection. Storage, control and background source carry the
same residual weight. The tangent and transpose include velocity feedback.

The [construction and retention ablations](ablations.md) add six matched
single-run comparisons on the unchanged numerical source. They retain the
original four pilots and remain separate from the final repeated campaign.

The neighboring transport study motivated testing a larger regularization
weight: its fixed-budget comparison reduced Krylov work at alpha = 1e-11.
Those capped runs used an earlier thermal formulation. They neither qualify
the corrected optimization nor establish a complete-solve speedup. The frozen
tangent study is a different approximation; this design uses exact derivatives.

## Run and verify

Run numerical commands on a compute node. Install the declared checkout with
`uv sync --frozen --group dev`. Use the GPU dependencies described in the main
coupled example for the hybrid backend. A supplied baseline must match the
field checksum in the design; the runner also checks geometry and properties.

```sh
uv run pytest tests/test_flow_conservation.py tests/test_coupled_protocol.py \
  tests/test_axisymmetric_residual_weighting.py \
  tests/test_thermal_stabilization_consistency.py \
  tests/test_consistent_control_jacobian.py tests/test_coupled_time_integration.py

uv run python -m deflation_example.coupled_sequence --config-name coupled_corrected \
  baseline_directory=runs/baseline method=reference alpha=1e-14 rank=200 \
  output=runs/corrected-alpha14-reference
```

Repeat for `method=jacobi rank=0 recycle_window=1` and for `alpha=1e-11`, always
using distinct output directories. The configuration fixes target 7, 64 slabs,
600 seconds, a smooth 60-second startup and zero inward margin. All four cases
remain in the pilot record. The final rank and three-method timing population
are selected only after the declared verification and resolution gates.

For the declared pilot, `run_case.py` additionally enforces the design's baseline
checksum and the full frozen checkout identity. Run cases 0 through 3:

```sh
uv run python examples/coupled_optimization/corrected_study/run_case.py \
  --case 0 --baseline runs/baseline --source FULL_COMMIT_ID \
  --output runs/corrected-case0
```

Replace `FULL_COMMIT_ID` with the selected checkout's forty-character commit ID.
The generic Hydra command above also supports other baselines for independent
experiments, which are outside this declared pilot population.

Summarize the four saved pilots in case order with the reporting checkout:

```sh
uv run python -m deflation_example.coupled_corrected_report \
  --runs runs/corrected-case0 runs/corrected-case1 runs/corrected-case2 runs/corrected-case3 \
  --design examples/coupled_optimization/corrected_study/protocol.json \
  --source FULL_NUMERICAL_COMMIT_ID --output runs/corrected-pilot-summary
```

The numerical source argument identifies the checkout used to optimize, which
can differ from the reporting checkout. The report rechecks equation and KKT
criteria, timer sums, source/formulation identity, inner histories and paired
saved fields. It keeps missing, capped and unsuccessful outcomes. A paired
complete-time ratio requires both optimizations to meet the same criteria and
agree in objective and temperature. Ratios below one remain in the output.
Resumed attempts require separate complete-cost assembly before comparison.
The pilot summary always leaves submission readiness false: resolution, bound
satisfaction and the repeated three-method campaign are still separate gates.

After a qualifying pair, assess the saved reference control at temporal
subdivisions 1, 2 and 4 with `coupled_newton_replay`, retaining every outcome.
Use the reporting checkout's strict temporal comparison:

```sh
uv run python -m deflation_example.coupled_time_resolution_report \
  --replays runs/forward-64 runs/forward-128 runs/forward-256 \
  --temperature-scale 20 --initial-value 0 --verify-equations --plot \
  --output runs/corrected-time-resolution
```

This option checks the recorded independently evaluated equations at each
returned step, the declared 600-second grid, and differences at all refined
time levels. It requires equation residuals at most 1e-12 and mass/energy
defects at most 1e-6. Failed and missing trajectories remain in the population.
It does not reassemble the equations or replace spatial and feasibility checks.
Earlier summary commands without `--verify-equations` retain their original
field-comparison role; their convergence labels alone do not establish this
additional verification.

The final equation tolerance is explicit (`equation_acceptance_tolerance=1e-12`).
It is independent of the optimizer's KKT tolerance and the inner original-system
residual. Legacy input files retain their original 1e-8 equation-acceptance
default; their results are never relabeled as evidence for the tighter setting.

Each retained nonlinear iterate produces a checkpoint. Restarting requires the
same numerical source and configuration; all attempts and discarded work must
be retained when computing complete elapsed cost. A wall limit or iteration
cap is an incomplete result, not an accepted timing.

## Conservation and discretization

The Taylor-Hood method enforces continuity against continuous P1 pressure tests.
Its verified global volume balance does not imply zero flux imbalance in every
element. The new diagnostics integrate all P2 face fluxes independently, check
the cylindrical divergence theorem, and report local flux defects and RMS
divergence separately. No projection or velocity modification is applied.
The thermal check includes storage, control, background sources, boundary
transport and discrete Dirichlet reactions. It verifies discrete energy balance,
not mesh convergence of the physical conductive flux.

The following command audits saved, independently verified replay states:

```sh
uv run python -m deflation_example.flow_conservation \
  --baseline runs/baseline --replay runs/forward-256 \
  --times 75 150 300 600 --output runs/mass-audit
```

Requested states must lie in the verified prefix and match their checkpoints.
The output records source and field identities, the baseline, and each selected
physical time. Local imbalance remains visible even when global balance passes.

The same face/volume check can compare existing computed-flow baselines across
mesh levels without reading a thermal trajectory:

```sh
uv run python -m deflation_example.flow_conservation \
  --baseline runs/baseline --baseline-only --output runs/baseline-mass
uv run python -m deflation_example.flow_conservation \
  --baseline runs/refined-baseline --baseline-only --output runs/refined-baseline-mass
```

Each result contains the actual momentum and weak-continuity residuals of its
saved baseline. These diagnostics use the baseline loader's existing accuracy
requirement; they do not relabel baseline records under the tighter trajectory
acceptance tolerance.

Temporal and spatial studies apply the saved source unchanged. Optimization
verification alone does not establish a resolved temperature trajectory or
temperature-bound satisfaction between optimization slabs. The resolution,
margin, independent-assessment and final-comparison gates are in the design.
