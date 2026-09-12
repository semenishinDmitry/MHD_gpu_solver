#include <gtest/gtest.h>

#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "mhd/mhd_types.hpp"
#include "state/state_field.hpp"

TEST(BoundaryConditions, PeriodicCopiesInteriorToGhosts)
{
    Grid2D grid(4, 3, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            MHDState S{};
            S.rho = static_cast<double>(10 * i + j);
            S.psi = static_cast<double>(i);
            U.set_state(i, j, S);
        }
    }

    apply_boundary_conditions(
        U, grid, BoundaryConditions(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic));

    EXPECT_DOUBLE_EQ(U.get_state(0, grid.j_begin()).rho, U.get_state(grid.nx + 0, grid.j_begin()).rho);
    EXPECT_DOUBLE_EQ(U.get_state(1, grid.j_begin()).psi, U.get_state(grid.nx + 1, grid.j_begin()).psi);
    EXPECT_DOUBLE_EQ(U.get_state(grid.ng + grid.nx, grid.j_begin()).rho,
                     U.get_state(grid.ng, grid.j_begin()).rho);
}

TEST(BoundaryConditions, OutflowCopiesNearestInterior)
{
    Grid2D grid(3, 3, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            MHDState S{};
            S.rho = static_cast<double>(i + 100 * j);
            U.set_state(i, j, S);
        }
    }

    apply_boundary_conditions(
        U, grid, BoundaryConditions(BoundaryConditionType::Outflow, BoundaryConditionType::Outflow));

    EXPECT_DOUBLE_EQ(U.get_state(0, grid.j_begin()).rho, U.get_state(grid.ng, grid.j_begin()).rho);
    EXPECT_DOUBLE_EQ(U.get_state(grid.ng + grid.nx, grid.j_begin()).rho,
                     U.get_state(grid.ng + grid.nx - 1, grid.j_begin()).rho);
}
