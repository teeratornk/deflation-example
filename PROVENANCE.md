# Provenance and attribution

This example accompanies *Reference-Space Reuse for Steady and Transient
State-Constrained Conjugate Heat Transfer* by Teeratorn Kadeethum, Francesco
Ballarin, Youngsoo Choi and Sanghyun Lee. The earlier
[reviewed manuscript version](https://github.com/teeratornk/opt_control/tree/reviewed-main-si-2026-09-08)
defines the original small presets and version-0.3 benchmark conventions.

Version 0.4.0 supplies the complete steady/transient reference-policy study,
stronger Jacobi-scaled recycling, matched transfer replay and its reporting
tools. The final GPU measurements use clean numerical source commit
`cc611292f66025f4d0f96a406c6aeb67799b7d34`. Later reporting and packaging
changes preserve those records. Their per-module hashes identify the
implementation used for each measurement. The
[reference-policy data version 1](https://github.com/teeratornk/deflation-example/tree/reference-policy-data-v1)
contains final comparisons and pilots in separate directories, with an exact
file-hash manifest. The source package excludes that larger data checkout.

Version 0.3.0 supplies the matched GPU, extended complete-sequence,
conditional PDE, recycling and matrix-free benchmarks. Their reports identify the exact
source commit and per-module SHA-256 values used for each measurement; these
are authoritative when comparing records with a later documentation release.
The earlier complete-sequence stopping protocols retain all failed attempts
in separate datasets. Later reports use their declared source and protocol.

The numerical core and problem definitions were adapted from the authors'
verified implementation. Packaging adds typed configuration, portable execution,
explicit input validation, optional CUDA kernels and output-preservation checks.
The small numerical fixtures test operator and target parity; they are not
performance targets or replacements for independent residual and KKT checks.

The example implements established PDAS and deflated-CG methods. The paper
provides their literature references and distinguishes its reference-reuse
policy from the underlying algorithms. This package does not claim a new
active-set or Krylov algorithm.

Code and documentation in this repository are distributed under the MIT license
in LICENSE. Manuscript text and third-party LaTeX templates are not bundled or
relicensed here. Dependency packages retain their own licenses.

AI assistance was used in code preparation, testing and documentation. The
authors are responsible for the released content. Runtime outputs record
software versions and source hashes, but not usernames, hostnames, absolute
installation paths or scheduler job identifiers.
