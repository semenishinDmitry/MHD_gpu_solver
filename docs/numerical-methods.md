# Numerical methods

## Spatial discretization

- **Method:** cell-centred finite volume on a uniform Cartesian mesh (`Grid2D`).
- **Ghosts:** `ng ≥ 2` required for MUSCL (two cells each side).
- **Reconstruction:** MUSCL with **Minmod** or **MC** slope limiter (`include/mhd/reconstruction.hpp`).
- **Riemann solver:** HLL with GLM-aware wave estimates (`include/mhd/hll.hpp`).
- **Fluxes:** physical ideal + GLM fluxes in `include/mhd/mhd_physics.hpp`; optional non-ideal face corrections in `include/mhd/nonideal.hpp`.

The update is the standard FV divergence of face fluxes stored in SoA `StateField` buffers.

## Time integration

- **Scheme:** SSP-RK2 (Heun) in `include/mhd/time_integrator.hpp`.
- **CFL:** hyperbolic estimate from \(\max(|v|+c_f)\) plus optional parabolic/dispersive
  restriction from non-ideal diffusivities (`include/mhd/cfl.hpp`).

## Boundary treatment

Supported types (per axis): **Periodic**, **Outflow**. Reflective is declared but not implemented.
Use `MHDSolver::set_bc(type_x, type_y)` for mixed boundaries (needed for 1D-in-2D shock tubes).

## Source terms

- GLM parabolic damping on \(\psi\).
- No gravity / geometry sources.

## Implemented vs planned

| Feature | Status |
|---------|--------|
| Ideal GLM-MHD 2D | Implemented |
| MUSCL + HLL + SSP-RK2 | Implemented |
| Ohmic / Hall / Ambipolar | Implemented (modular) |
| 1D shock tubes via thin 2D strip | Supported |
| Constrained transport / face-centred B | Not implemented |
| Adaptive mesh | Not implemented |
| GPU kernels | Design goal / not in this CPU path |
