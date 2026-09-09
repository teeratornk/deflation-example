# Provenance and attribution

This example accompanies *Online Spectral Deflation for State Constrained
Optimal Control Problems* by Teeratorn Kadeethum, Francesco Ballarin, Youngsoo
Choi and Sanghyun Lee. The
[reviewed manuscript version](https://github.com/teeratornk/opt_control/tree/reviewed-main-si-2026-09-08)
defines the methods and problem conventions used here.

Version 0.2.0 additionally supplies the matched GPU and complete-sequence
benchmarks used in the submission revision. Their reports identify the exact
source commit and per-module SHA-256 values used for each measurement; these
are authoritative when comparing records with a later documentation release.

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
