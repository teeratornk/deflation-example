# Coupled trajectory consistency

These checks compare a saved optimization trajectory with the response to its
unchanged heat source. They use the same mesh, thermal properties, boundaries,
initial conditions and physical time intervals. This example supplies numerical
diagnostics; the original optimization timings remain attached to their sources.

Run from a checkout containing `deflation_example.coupled_replay_check`, on an
allocated compute node. Replace the input paths with the verified baseline and
optimization outputs described in the [parent example](../README.md).

```bash
uv sync --frozen
uv run python -m deflation_example.coupled_replay_check \
  --baseline runs/stabilized-baseline --optimization runs/optimized \
  --method reference --threads 8 --output runs/saved-equations
```

The command recomputes momentum, continuity, thermal residuals, mass balance and
discrete energy balance at every saved time slab. A direct thermal solve with the
saved velocity and preceding state checks source recovery independently of the
optimization recurrence. It leaves the saved fields unchanged.

## Fixed-source tolerance study

The original forward protocol uses a coupled residual target of `1e-8` and an
internal momentum target of `1e-9`. The following matched runs tighten both.
The optimizer's final residual and KKT criteria remain unchanged. Temperature is
never clipped, and the control is never recovered from the replayed response.

```bash
for tolerance in 1e-8 1e-10 1e-11; do
  uv run python -m deflation_example.coupled_resolution \
    --baseline runs/stabilized-baseline --optimization runs/optimized \
    --method reference --subdivision 1 --threads 8 \
    --tolerance "$tolerance" --relaxation 0.5 --coupling-cap 100 \
    --output "runs/replay-$tolerance"
done
uv run python -m deflation_example.coupled_replay_check \
  --baseline runs/stabilized-baseline --optimization runs/optimized \
  --method reference --replay runs/replay-1e-8 --threads 8 \
  --output runs/saved-versus-replay
```

Each output records the source hashes, stopping targets, every step's status,
residual histories and temperature differences. The CLI permits stricter targets
only. The internal momentum target is one tenth of the coupled target. Iteration
caps, nonconvergence and incomplete trajectories remain visible.

With `--replay`, the diagnostic also solves the thermal equation using the saved
velocity and the replay's preceding temperature. It reports that propagated
temperature difference separately from the remaining coupled difference. Their
maximum norms are not additive. This decomposition does not establish which
nonlinear solution branch was reached.

For a local initial-guess comparison, append `--local-slabs 0 15 27` and
`--local-tolerance 1e-10` to the saved-equation command and select a new output
directory. Each selected step uses the exact saved preceding temperature and
velocity. One solve starts from the saved current fields; the other starts from
the saved preceding fields. Both retain the same control and convergence rules.
The output retains every local solve, its residual history, and the temperature
and velocity differences. These local checks separate errors accumulated over
earlier steps from sensitivity to the current initial guess.

Compare the complete tolerance sequence before interpreting temperature-bound
violations or tracking changes. Small equation residuals alone do not bound
trajectory error. A remaining unexplained discrepancy requires further diagnosis
before the coupled example can support an application claim. Temporal or spatial
refinement is a separate check of discretization error.

## Generate the comparison

The transformer example uses a temperature scale of 20 K. After the replays,
regenerate the summary and temperature-difference curves from their saved fields:

```bash
uv run --extra plot python -m deflation_example.coupled_replay_report \
  --optimization-fields runs/optimized/reference-fields.npz \
  --replays runs/replay-1e-8 runs/replay-1e-10 runs/replay-1e-11 \
  --temperature-scale 20 --plot --output runs/replay-summary
```

The report checks the common optimized source, time grid and configuration. It
recomputes each temperature difference and checks it against the saved summary.
Incomplete replays retain their termination status and partial trace; their
full-trajectory difference remains unavailable. A missing run remains a row in
the summary. Scheduler timeouts must be recorded separately when a killed
process leaves its last numerical status as `running`.
