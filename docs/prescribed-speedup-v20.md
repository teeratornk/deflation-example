# Scale, trajectory and operating-condition comparisons

This page documents the comparisons that extend the prescribed-flow CHT study to
larger meshes, longer trajectories and changing operating conditions. All runs
use the complete query procedure: one reference is computed before the first
query, stored on the GPU, and restricted to the inactive set at every
primal-dual active-set update. Body-fitted references come from generalized
eigenvectors of the reduced Hessian on a coarse mesh level, prolonged to the
target mesh; Cartesian references use closed-form Laplacian modes.

RefDef denotes reference deflation, the full-domain reference restricted directly
to each inactive set; its method key in configurations and records remains
`reference`.

The transformer comparisons of this page used the original advective transport.
The [skew-transport page](prescribed-speedup-v21.md) repeats every one of them with
the skew-symmetric transport and gives the current manuscript tables; its evidence
archive also contains the Cartesian and bore-in-block comparisons of this page.

## Source

| Component | Branch | Commit |
| --- | --- | --- |
| Operating scenarios (load profiles, flow rates, chained windows) | `study/prescribed-speedup-v20` | [`e741f1b`](https://github.com/teeratornk/deflation-example/tree/e741f1b0b9cc01a7ecb028e75a2ef81cd6ddc670) |
| Skew-symmetric transformer transport | `study/prescribed-speedup-v20` | [`5fc1e3d`](https://github.com/teeratornk/deflation-example/tree/5fc1e3d400a81ac4a3c6c775689ce97cb68f65db) |
| Evidence digests, tables and figure | `study/prescribed-speedup-v20-evidence` | [`9c0ec6d`](https://github.com/teeratornk/deflation-example/tree/9c0ec6d3bea5bb2972683c51dc0dcffe494f81ea) |

The [records](../examples/prescribed_speedup_v20/README.md) contain the evidence
archive, the complete comparison listing and the checksums.

## Definitions

- **Converged repetition:** a complete sequence in which every query meets the
  independently recomputed relative residual `1e-10` and every KKT measure `1e-8`.
- **Speedup:** the median complete-sequence time of the fastest alternative with
  at least one converged repetition, divided by the median time of RefDef. Medians use converged repetitions only.
- **Iteration ratio:** the median inner iterations of Jacobi-CG divided by those
  of RefDef.
- **Not converged:** a method with at least one repetition that misses the
  residual or KKT criteria, reaches an iteration cap, or fails.
- **AmgX (1e-2), AmgX (1e-3):** AmgX with tenfold and hundredfold tighter
  internal tolerances.

Complete-sequence time covers assembly, reference or resource construction, every
query, all active-set updates and inner solves, transfers, verification and
cleanup. Library and process startup are excluded.

## Comparisons

- **Scale:** steady Cartesian grids from `32^3` to `128^3`, the transformer and
  bore-in-block meshes Bore 1 to Bore 4 (up to 2,540,695 unknowns).
- **Trajectories:** Cartesian `12^3` to `64^3` with 8 to 32 slabs, the
  transformer and Bore 1 to Bore 3 with 4 or 8 slabs.
- **Operating conditions:** a 24-window transformer day with a residential load
  cycle, a 16-window drive cycle on Bore 2, off-design flow rates from 0.5 to 2
  times nominal, emergency overloads, and sorted and shuffled query orders.
- **Supplementary variations:** tolerances, regularization, cold starts, rank and
  reference level, 64 and 256 targets, long horizons, sparse direct and
  block-in-time CPU solvers, and time refinement of the transformer transport.

The [complete listing](../examples/prescribed_speedup_v20/populations.md) reports
every comparison, including exploratory single-repetition runs, with converged
and attempted repetitions per method.

## Reproduce the tables and figure

From a checkout of this revision, copy the archive and switch to the evidence commit:

```bash
cp examples/prescribed_speedup_v20/evidence.tar.gz /tmp/v20-evidence.tar.gz
git checkout --detach 9c0ec6d3bea5bb2972683c51dc0dcffe494f81ea
mkdir /tmp/v20 && tar -xzf /tmp/v20-evidence.tar.gz -C /tmp/v20
uv run --locked --extra plot python -c "
from deflation_example import v20_evidence as ev
ev.verify('/tmp/v20'); ev.export('/tmp/v20', '/tmp/v20-out')
"
```

`verify` checks every file of the archive against its manifest. `export` writes
the table rows, the macros, the summary, the listing and the figure to `/tmp/v20-out`.
The regenerated macros match the manuscript byte for byte. The current tables
come from the [skew-transport export](prescribed-speedup-v21.md#reproduce-the-tables),
which replaces the transformer rows of this export. The export reads the run digests
only and needs neither a GPU nor a network connection.
