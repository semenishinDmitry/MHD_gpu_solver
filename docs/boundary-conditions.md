# Boundary conditions

Implemented in `include/boundary_conditions/boundary_conditions.hpp`.

| Type | Meaning | Implementation notes |
|------|---------|----------------------|
| **Periodic** | Identify opposite sides | Copies interior into ghosts |
| **Outflow** | Zero-order extrapolation | Copies nearest interior into ghosts |
| **Reflective** | Enum exists | **Not implemented** (throws) |

Per-axis configuration via `BoundaryConditions{type_x, type_y}` or
`MHDSolver::set_bc(type_x, type_y)`.

## Typical usage

| Problem | BC |
|---------|----|
| Orszag–Tang, Rotor, KH, Alfvén | Periodic × Periodic |
| Sod, Brio–Wu (1D-in-2D strip) | Outflow × Periodic |

1D tubes use `ny=1` and a thin physical height \(\approx\Delta x\) so that the y-direction is a
periodic dummy dimension.
