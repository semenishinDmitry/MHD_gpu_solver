# Equations

This code integrates the **ideal GLM-MHD** system in two spatial dimensions (with optional
non-ideal Ohmic / Hall / Ambipolar electric fields). Units are such that \(\mu_0 = 1\).

## Conserved and primitive variables

Conserved state \(\mathbf{U} = (\rho,\, \rho\mathbf{v},\, E,\, \mathbf{B},\, \psi)^\mathsf{T}\).

Primitives \(\mathbf{W} = (\rho,\, \mathbf{v},\, p,\, \mathbf{B},\, \psi)^\mathsf{T}\).

Equation of state:
\[
E = \frac{p}{\gamma-1} + \tfrac12\rho|\mathbf{v}|^2 + \tfrac12|\mathbf{B}|^2.
\]

## Ideal MHD + GLM (Dedner)

\[
\partial_t\rho + \nabla\cdot(\rho\mathbf{v}) = 0
\]
\[
\partial_t(\rho\mathbf{v})
  + \nabla\cdot\big(
      \rho\mathbf{v}\mathbf{v} + (p+B^2/2)\mathbf{I} - \mathbf{B}\mathbf{B}
    \big) = 0
\]
\[
\partial_t E
  + \nabla\cdot\big(
      (E+p+B^2/2)\mathbf{v} - (\mathbf{v}\cdot\mathbf{B})\mathbf{B} + \psi\mathbf{B}
    \big) = 0
\]
\[
\partial_t\mathbf{B}
  + \nabla\cdot(\mathbf{v}\mathbf{B}-\mathbf{B}\mathbf{v}+\psi\mathbf{I}) = 0
\]
\[
\partial_t\psi + c_h^2\nabla\cdot\mathbf{B} = -\frac{\alpha c_h}{h}\psi
\]

Here \(c_h\) is taken from the hyperbolic CFL estimate and \(h=\min(\Delta x,\Delta y)\).
The scalar \(\psi\) is Dedner’s hyperbolic cleaning variable; \(\alpha\) is the parabolic
damping parameter (`glm_alpha` in the code).

## Non-ideal terms (optional)

When enabled, an additional electric field \(\mathbf{E}_{\mathrm{ni}}\) enters Faraday’s law
and the energy flux as \(\mathbf{B}\times\mathbf{E}_{\mathrm{ni}}\):

\[
\mathbf{E}_{\mathrm{ni}}
  = \eta_{\mathrm{ohm}}\mathbf{J}
  + \eta_{\mathrm{Hall}}\mathbf{J}\times\hat{\mathbf{B}}
  + \eta_{\mathrm{A}}(\mathbf{J}\times\mathbf{B})\times\mathbf{B}/B^2,
\quad \mathbf{J}=\nabla\times\mathbf{B}.
\]

See `include/mhd/nonideal.hpp` for the discrete face fluxes.

## Constraint

Ideal MHD requires \(\nabla\cdot\mathbf{B}=0\). The GLM subsystem transports and damps
divergence errors; the diagnostic `max_div_b` monitors
\(\max|\partial_x B_x + \partial_y B_y|\) on cell centres.
