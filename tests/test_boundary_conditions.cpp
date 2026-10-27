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
        U, grid,
        BoundaryConditions(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic));

    EXPECT_DOUBLE_EQ(U.get_state(0, grid.j_begin()).rho,
                     U.get_state(grid.nx + 0, grid.j_begin()).rho);
    EXPECT_DOUBLE_EQ(U.get_state(1, grid.j_begin()).psi,
                     U.get_state(grid.nx + 1, grid.j_begin()).psi);
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
        U, grid,
        BoundaryConditions(BoundaryConditionType::Outflow, BoundaryConditionType::Outflow));

    EXPECT_DOUBLE_EQ(U.get_state(0, grid.j_begin()).rho, U.get_state(grid.ng, grid.j_begin()).rho);
    EXPECT_DOUBLE_EQ(U.get_state(grid.ng + grid.nx, grid.j_begin()).rho,
                     U.get_state(grid.ng + grid.nx - 1, grid.j_begin()).rho);
}

// ny = 1 < ng = 2: every periodic ghost row must equal the single interior row
// (previously ghost row 0 copied ghost row 1 and lagged one BC application behind).
TEST(BoundaryConditions, PeriodicFewerCellsThanGhosts)
{
    Grid2D grid(4, 1, 0.0, 1.0, 0.0, 0.25, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());
    for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
        U.rho[static_cast<std::size_t>(U.index(i, grid.j_begin()))] = 1.0 + i;
    }
    apply_boundary_conditions(
        U, grid,
        BoundaryConditions(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic));
    for (int j = 0; j < grid.get_size_y(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            EXPECT_DOUBLE_EQ(U.rho[static_cast<std::size_t>(U.index(i, j))], 1.0 + i) << "j=" << j;
        }
    }
    EXPECT_EQ(periodic_source(0, 1, 2, true), 2);
    EXPECT_EQ(periodic_source(1, 1, 2, true), 2);
    EXPECT_EQ(periodic_source(0, 8, 2, true), 8); // unchanged for n >= ng
    EXPECT_EQ(periodic_source(1, 8, 2, false), 3);
}
