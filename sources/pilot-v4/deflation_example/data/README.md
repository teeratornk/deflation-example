# Body-fitted thermal meshes

The numeric bundles support the mesh optimization examples. All NumPy inputs
load with `allow_pickle=False`.

`engine_3d` contains a conforming tetrahedral mesh of a cylindrical gas bore
inside a metal block. The package generates it with
`python -m deflation_example.generate_engine_mesh`. The geometry, conductivity
ratio, interface heat flux, and coolant temperature follow the engine case in
the GPU thermal-solver manuscript. The optimization example prescribes a smooth
azimuthal velocity independently of temperature. It uses the documented unit
dimensionless thermal capacity in both materials.

`transformer_2d` contains an axisymmetric triangular mesh with winding, oil,
and baffle regions. Its numeric input bundle supplies frozen material
properties, a distributed source, and a quadratic prescribed Stokes velocity.
The inlet and outlet fluxes agree within the tolerance recorded in
`parameters.json`. The coordinate ordering is `(r,z)` in metres.

The mesh owner supplied these self-built geometries for this companion.
The transformer exporter records hashes of its two input files and exports
only the listed numerical fields. Measurement records, application case files,
and machine paths are excluded. The packaged mesh and flow are sufficient to
run and refine the thermal optimization without the source application.

Refinement subdivides the packaged simplices and preserves material regions.
The transformer flow is evaluated from its parent P2 polynomial on each child.
This preserves the original baffle geometry and requires no flow recalibration.
The engine refinements preserve the initial faceted representation of the bore.
Geometry approximation and finite-element refinement are separate errors.

Mesh construction and interpretation draw on the AI4PDEs thermal examples.
Their software license is reproduced below for attribution.

MIT License

Copyright (c) 2024 Imperial College London

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
