# Reconstruction (MUSCL)

Interface states are obtained from cell-centred primitives with limited slopes.

## Piecewise linear reconstruction

For each conserved/primitive component, left/right states at an interface use

\[
W_{i+1/2}^L = W_i + \tfrac12 \Delta_i,\qquad
W_{i+1/2}^R = W_{i+1} - \tfrac12 \Delta_{i+1},
\]

with limited differences \(\Delta\) from neighbouring cells (`include/mhd/reconstruction.hpp`).

## Limiters

- **Minmod** — most dissipative; preferred for discontinuous tests (Sod, Brio–Wu, rotor).
- **MC (monotonized central)** — less dissipative; default for smooth/vortex runs (Orszag–Tang, Alfvén).

## Order of accuracy

On smooth solutions the reconstruction is formally **second-order** in space when the limiter
is inactive. Near discontinuities the limiter reduces the local accuracy to first order —
this is intentional TVD behaviour, not a bug.

## Positivity

There is no dedicated positivity-preserving limiter. Unphysical states after reconstruction
fall back to first-order (piecewise constant) in the MUSCL path when primitives are invalid.
Fast CI verification asserts \(\rho>0\) and \(p>0\) after evolution.
