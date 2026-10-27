# Geometry (Cartesian fluid/solid mask)

Interior obstacles are represented as a **staircase immersed boundary** on the
existing uniform Cartesian mesh. The outer domain remains a rectangle
\([x_{\min},x_{\max}]\times[y_{\min},y_{\max}]\) with \(n_x\times n_y\) cells.

## Mask layout

- Storage: `FluidMask` (`include/mhd/mask.hpp`), row-major over **interior** cells only
  (`ny * nx`).
- Values: `1` = fluid, `0` = solid.
- Default: all fluid (identical to the pre-mask solver path).

Cell centers match `Grid2D`:

\[
x_i = x_{\min} + (i + \tfrac12)\,\Delta x,\qquad
y_j = y_{\min} + (j + \tfrac12)\,\Delta y.
\]

Python rasterization lives in `gui/geometry.py` (`Rectangle`, `Circle`, `Polygon`,
`rasterize` / `rasterize_scene`).

## Wall treatment

At fluid–solid faces the solver fills solid cells from a neighboring fluid cell
(slip / reflective on the normal velocity component) and **zeros the RHS** on
solid cells so they are not evolved. See [boundary-conditions.md](boundary-conditions.md).

## Limitations (v1)

- Not body-fitted: walls follow cell faces (staircase). Fine features need fine grids.
- No cut-cell volume fractions, unstructured mesh, or CAD import.
- Domain-edge `Reflective` BC enum remains unimplemented; use solid cells for walls.
- Accuracy near curved obstacles is first-order in grid spacing for geometry error.

## API sketch

```cpp
MHDSolver solver(nx, ny, x_min, x_max, y_min, y_max);
solver.set_mask(mask_uint8, nx * ny);  // 1=fluid, 0=solid
solver.clear_mask();
solver.set_walls_reflective(true);     // default
```

Python:

```python
solver.set_mask(mask)       # uint8 (ny, nx)
mask = solver.get_mask()
solver.set_field("rho", arr)
solver.set_primitives(rho, vx, vy, vz, p, bx, by, bz)
```
