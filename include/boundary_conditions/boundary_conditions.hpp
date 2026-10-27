#pragma once

#include "grid/grid.hpp"
#include "state/state_field.hpp"

#include <stdexcept>

enum class BoundaryConditionType { Periodic, Reflective, Outflow };

struct BoundaryCondition2D {
    BoundaryConditionType type_x;
    BoundaryConditionType type_y;

    BoundaryCondition2D(BoundaryConditionType type_x_, BoundaryConditionType type_y_)
        : type_x(type_x_), type_y(type_y_)
    {
    }
};

using BoundaryConditions = BoundaryCondition2D;

// Interior source (global index) of periodic ghost g on the low side (low = true) or the high
// side of an axis with n interior cells and ng ghosts. Wraps modulo n, so it is also correct
// when n < ng (e.g. ny = 1 strips used for 1D problems), where a plain `n + g` would read
// another, not yet updated ghost cell.
inline int periodic_source(int g, int n, int ng, bool low)
{
    const int offset = low ? (g - ng) : g;
    return ng + ((offset % n) + n) % n;
}

inline void apply_boundary_conditions_x(StateField& U, const Grid2D& grid,
                                        BoundaryConditionType type)
{
    const int ng = grid.ng;
    const int nx = grid.nx;
    const int ny_tot = grid.get_size_y();

    for (int j = 0; j < ny_tot; ++j) {
        for (int g = 0; g < ng; ++g) {
            const int i_left_ghost = g;
            const int i_right_ghost = ng + nx + g;

            switch (type) {
            case BoundaryConditionType::Periodic:
                U.copy_cell(i_left_ghost, j, periodic_source(g, nx, ng, true), j);
                U.copy_cell(i_right_ghost, j, periodic_source(g, nx, ng, false), j);
                break;
            case BoundaryConditionType::Outflow:
                U.copy_cell(i_left_ghost, j, ng, j);
                U.copy_cell(i_right_ghost, j, ng + nx - 1, j);
                break;
            case BoundaryConditionType::Reflective:
                throw std::invalid_argument(
                    "Reflective boundary conditions are not implemented yet");
            }
        }
    }
}

inline void apply_boundary_conditions_y(StateField& U, const Grid2D& grid,
                                        BoundaryConditionType type)
{
    const int ng = grid.ng;
    const int ny = grid.ny;
    const int nx_tot = grid.get_size_x();

    for (int i = 0; i < nx_tot; ++i) {
        for (int g = 0; g < ng; ++g) {
            const int j_bottom_ghost = g;
            const int j_top_ghost = ng + ny + g;

            switch (type) {
            case BoundaryConditionType::Periodic:
                U.copy_cell(i, j_bottom_ghost, i, periodic_source(g, ny, ng, true));
                U.copy_cell(i, j_top_ghost, i, periodic_source(g, ny, ng, false));
                break;
            case BoundaryConditionType::Outflow:
                U.copy_cell(i, j_bottom_ghost, i, ng);
                U.copy_cell(i, j_top_ghost, i, ng + ny - 1);
                break;
            case BoundaryConditionType::Reflective:
                throw std::invalid_argument(
                    "Reflective boundary conditions are not implemented yet");
            }
        }
    }
}

inline void apply_boundary_conditions(StateField& U, const Grid2D& grid,
                                      const BoundaryConditions& bc)
{
    if (grid.ng < 1) {
        throw std::invalid_argument("apply_boundary_conditions requires at least one ghost cell");
    }
    if (U.nx != grid.get_size_x() || U.ny != grid.get_size_y()) {
        throw std::invalid_argument("StateField size does not match Grid2D");
    }

    // Fill x-ghosts first, then y-ghosts over the full x-range (including corners).
    apply_boundary_conditions_x(U, grid, bc.type_x);
    apply_boundary_conditions_y(U, grid, bc.type_y);
}
