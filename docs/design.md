# Design and extension points

The package uses small modules and a callable protocol to keep responsibilities
separate. The CPU numerical layer does not import Hydra, Torch, plotting or reporting.

| Responsibility | Module |
|---|---|
| Operators, targets and reference modes | `problems.py` |
| PDAS, deflated CG and accuracy checks | `solvers.py` |
| Shared real-valued input contracts | `validation.py` |
| Interchangeable timed linear kernels | `backends.py` |
| Running and checking an experiment | `runner.py` |
| Portable metadata and serialization | `reporting.py` |
| Configuration and presentation | `cli.py`, `plotting.py` |

This applies the SOLID principles at the module and callable boundaries:

- **Single responsibility:** numerical routines do not write files or interpret
  command-line options. Serialization can be tested without running an optimizer.
- **Open for extension:** pass a mapping of named kernels to `run_demo(kernels=...)`
  to add a comparison without editing its experiment loop. PDAS also accepts an
  inactive-solver callback independently of the kernel comparisons.
- **Substitution:** CPU, CUDA and custom kernels use the same `LinearKernel`
  contract. They must preserve inputs and return a `LinearResult` plus nonnegative
  timing fields including `total_seconds`. The runner recomputes the original
  residual, checks the solution shape and rejects misleading success reports.
- **Small interfaces:** kernels only solve a supplied linear system; they do not
  need to implement configuration, plotting, optimization or output storage.
- **Dependency inversion:** experiment execution accepts callable implementations;
  the adapter module provides the default CPU and optional CUDA choices.

`run_demo` owns a new output directory and refuses to overwrite an existing one.
Reports are serialized into a temporary sibling file and replaced atomically.
An I/O failure therefore does not leave half a JSON document in place of the
previous record. Metadata uses an allowlist and bounds optional Git queries with
a timeout. No solver requires Git or network access.

These contracts are tested with injected kernels, including one that deliberately
returns an incorrect solution while claiming convergence. They are not a claim
that arbitrary third-party code is safe: extensions must be trusted code and
must respect the input-preservation contract.

The algorithms require real SPD systems. Explicit matrix entries are checked for
finiteness; the package does not perform an expensive global SPD certification
for every solve, and a matrix-free operator remains the caller's responsibility.
Very large or tiny values can exceed float64's arithmetic range and cause a
reported breakdown. They must not produce a false convergence certificate.
