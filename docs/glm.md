# GLM divergence cleaning

## Why \(\nabla\cdot\mathbf{B}=0\)

Maxwell’s equations require a divergence-free magnetic field. Cell-centred FV schemes
accumulate \(\nabla\cdot\mathbf{B}\) errors that can seed unphysical forces.

## Dedner hyperbolic/parabolic GLM

This code implements Dedner’s GLM approach: an additional scalar \(\psi\) transports
divergence errors at speed \(c_h\) and damps them with rate \(\propto \alpha c_h / h\).

Parameters:

| Symbol | Code | Role |
|--------|------|------|
| \(c_h\) | from CFL | hyperbolic cleaning speed |
| \(\alpha\) | `glm_alpha` (default `0.1`) | parabolic damping strength |
| \(h\) | `min(dx,dy)` | local mesh scale |

Fluxes: \(F(B_x)=\psi\), \(F(\psi)=c_h^2 B_x\) (and analogously in \(y\)), plus \(\psi\mathbf{B}\)
contribution to the energy flux.

## Diagnostics

`MHDSolver::max_div_b()` / `max_abs_div_b` — max absolute cell-centred
\(\partial_x B_x+\partial_y B_y\) on the interior. Verification tests assert finiteness and
problem-dependent upper bounds; they do **not** claim machine-zero divergence.
