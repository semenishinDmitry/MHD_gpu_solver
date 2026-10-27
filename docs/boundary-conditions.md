# Boundary conditions

Implemented in `include/boundary_conditions/boundary_conditions.hpp`.

| Type | Meaning | Implementation notes |
|------|---------|----------------------|
| **Periodic** | Identify opposite sides | Copies interior into ghosts; source index wraps modulo the interior size (`periodic_source`), so it also works when there are fewer interior cells than ghosts (`ny = 1` strips) |
| **Outflow** | Zero-order extrapolation | Copies nearest interior into ghosts |
| **Reflective** | Enum exists | **Not implemented** on domain edges (throws) |

Per-axis configuration via `BoundaryConditions{type_x, type_y}` or
`MHDSolver::set_bc(type_x, type_y)`.

## Interior solid walls (staircase IB)

Fluid/solid masking (`include/mhd/mask.hpp`, see [geometry.md](geometry.md)) provides
**reflective (slip) walls** at fluid–solid interfaces:

1. Before each RHS evaluation, solid cells are filled from a cardinal fluid neighbor:
   density, pressure, \(\mathbf{B}\), and \(\psi\) are copied; the **normal** velocity
   component is negated (slip / reflective).
2. The RHS on solid cells is forced to zero so solids stay inert under time integration.
3. CFL scanning skips solid cells.

Assumptions for ideal MHD staircase IB:

- Geometry is staircased to cell faces (not cut-cell / body-fitted).
- Magnetic field is copied into solids (no special conducting/insulating wall model).
- Tangential velocity is not forced to zero (slip), matching a simple reflective wall.

Toggle with `MHDSolver::set_walls_reflective(true)` (default). Domain-edge Reflective
remains unimplemented; place solid cells against the boundary if you need a wall there.

## Typical usage

| Problem | BC |
|---------|----|
| Orszag–Tang, Rotor, KH, Alfvén | Periodic × Periodic |
| Sod, Brio–Wu (1D-in-2D strip) | Outflow × Periodic |
| Channel + obstacle | Outflow (or Periodic) edges + solid mask |

1D tubes use `ny=1` and a thin physical height \(\approx\Delta x\) so that the y-direction is a
periodic dummy dimension. With `ny = 1 < ng = 2`, every y-ghost row must be a copy of the single
interior row. Earlier versions used the index `ny + g`, so ghost row 0 copied ghost row 1 and
lagged one boundary update behind (all zeros at `t = 0`). The modulo formula fixes this;
`BoundaryConditions.PeriodicFewerCellsThanGhosts` covers it. The GPU backends use the same
formula.
