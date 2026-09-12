# Riemann solvers

## Implemented: HLL

The code uses the **Harten–Lax–van Leer (HLL)** approximate Riemann solver for both \(x\) and
\(y\) interfaces (`include/mhd/hll.hpp`).

Idea: two-wave approximation with signal speeds \(S_L,S_R\) bounding the fan; the interface
flux is the HLL average of the left/right physical fluxes when \(S_L < 0 < S_R\).

## GLM-aware speeds

Wave estimates include the cleaning speed \(c_h\) in addition to fast magnetosonic speeds, so
that \(\psi\)–\(B_n\) coupling remains stable under CFL.

## Assumptions / limitations

- Single solver (HLL only); no Roe / HLLC / HLLD in this tree.
- HLL is more dissipative than HLLC/HLLD on contact and Alfvén waves — acceptable for a
  robust baseline, but convergence rates on smooth Alfvén waves may be below design order
  at coarse resolution.
