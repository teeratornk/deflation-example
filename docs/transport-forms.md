# Thermal transport forms of the prescribed-flow models

The body-fitted models solve the temperature equation with a prescribed velocity
$v$, a cellwise constant volumetric capacity $c$ and conductivity $K$:

$$c\,(\partial_t T + v\cdot\nabla T) - \nabla\cdot(K\nabla T) = q .$$

## Discrete forms

`assemble_thermal` (`src/deflation_example/meshes.py`) uses P1 temperature and, on
the axisymmetric transformer, the prescribed continuous P2 Stokes velocity. All
integrals carry the axisymmetric weight $2\pi r$. With test function $w$ and trial
function $T$ on a cell $e$:

- diffusion: $\int_e 2\pi r\, \nabla w\cdot K\nabla T$ (exact);
- capacity, mass and load: lumped, $\int_e 2\pi r\, N_i$ (exact);
- original ("advective") transport: $a_e(T,w)=\int_e 2\pi r\, c\, w\,(v_h\cdot\nabla T)$;
- skew-symmetric transport: $a_e(T,w)+\tfrac12\int_e 2\pi r\, c\,(\nabla\cdot v_h)\, T\, w$,
  where the axisymmetric divergence is $\partial_r v_r + \partial_z v_z + v_r/r$;
- streamline diffusion (both forms): $\tau_e\int_e 2\pi r\,(c\,\bar v_e\cdot\nabla w)(c\,\bar v_e\cdot\nabla T)$
  with $\bar v_e$ the centroid velocity and
  $\tau_e=\min\{h_e/(2c|\bar v_e|),\,h_e^2/(12\lambda_{\min}(K_e))\}$.

The transport integrands have polynomial degree four; the seven-point degree-five
triangle rule integrates them exactly.

## Energy identity and boundary terms

For $T=w$ and $c$ constant on $e$,
$\tfrac12\nabla\cdot(r\,v\,T^2) = r\,T\,v\cdot\nabla T + \tfrac12 r\,(\nabla\cdot v)\,T^2$
in the $(r,z)$ plane, so the skew form of each cell equals
$\tfrac12\oint_{\partial e} 2\pi r\, c\,(v_h\cdot n)\,T^2$. The velocity is continuous
and vanishes in the solid and baffle cells, so interior contributions cancel and

$$T^\top A_{\rm skew} T = \tfrac12\oint_{\partial\Omega} 2\pi r\, c\,(v_h\cdot n)\,T^2 .$$

The inlet temperature is prescribed and eliminated, walls carry $v_h=0$, the axis
carries weight $r=0$, and the outlet carries outflow $v_h\cdot n\ge 0$. The
symmetric part of the skew transport is therefore positive semidefinite on the free
nodes. The original form differs by $-\tfrac12\int 2\pi r\,c\,(\nabla\cdot v_h)T^2$,
which has no sign: the P2 Stokes velocity is divergence-free only weakly.

## Relation to the governing equation

For $\nabla\cdot v = 0$ both forms discretize the same equation; the skew form adds
$\tfrac12 c(\nabla\cdot v_h)T$, a consistency term of the size of the discrete
divergence. Streamline diffusion is an artificial diffusion without residual
weighting: first-order consistent where it is active ($\tau_e\propto h_e$) and second
order where the cell Péclet number is small ($\tau_e\propto h_e^2$).

## Verification

- `tests/test_transport_verification.py`
  - manufactured axisymmetric solutions with a stream-function velocity: both forms
    converge at second order without streamline diffusion and at resolved Péclet
    numbers with it; advection-dominated flow with streamline diffusion converges at
    first order;
  - the symmetric part of the skew transport on the packaged transformer mesh equals an
    independently assembled outflow-energy matrix, and every boundary edge with a free
    node carries outflow or no flow;
  - exact inertia of the symmetric part of the free-node operator (symmetric-mode LU
    without row interchanges): positive definite for the skew transformer operator,
    indefinite for the original one, positive definite for Bore 1.
- `python -m deflation_example.transport_certificate` records the same inertia for
  every mesh used in a timed comparison. A positive-definite symmetric part gives
  $\operatorname{Re}\lambda = x^*Kx/x^*Cx > 0$ for every eigenpair $Kx=\lambda Cx$, so
  no backward-Euler mode grows at any time step.
