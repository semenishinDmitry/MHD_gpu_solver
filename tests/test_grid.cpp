#include <gtest/gtest.h>

#include "grid/grid.hpp"

#include <cmath>

TEST(Grid2D, DefaultConstructedIsZero) {
    Grid2D grid;
    EXPECT_EQ(grid.nx, 0);
    EXPECT_EQ(grid.ny, 0);
    EXPECT_EQ(grid.ng, 0);
    EXPECT_DOUBLE_EQ(grid.x_min, 0.0);
    EXPECT_DOUBLE_EQ(grid.x_max, 0.0);
    EXPECT_DOUBLE_EQ(grid.y_min, 0.0);
    EXPECT_DOUBLE_EQ(grid.y_max, 0.0);
    EXPECT_DOUBLE_EQ(grid.dx, 0.0);
    EXPECT_DOUBLE_EQ(grid.dy, 0.0);
}

TEST(Grid2D, ConstructorSetsExtentsAndSpacing) {
    Grid2D grid(100, 50, 0.0, 1.0, -1.0, 1.0);

    EXPECT_EQ(grid.nx, 100);
    EXPECT_EQ(grid.ny, 50);
    EXPECT_DOUBLE_EQ(grid.x_min, 0.0);
    EXPECT_DOUBLE_EQ(grid.x_max, 1.0);
    EXPECT_DOUBLE_EQ(grid.y_min, -1.0);
    EXPECT_DOUBLE_EQ(grid.y_max, 1.0);
    EXPECT_DOUBLE_EQ(grid.dx, 1.0 / 99.0);
    EXPECT_DOUBLE_EQ(grid.dy, 2.0 / 49.0);
}

TEST(Grid2D, GetDxDyUseCellCount) {
    Grid2D grid(100, 50, 0.0, 1.0, 0.0, 2.0);

    EXPECT_DOUBLE_EQ(grid.get_dx(), 0.01);
    EXPECT_DOUBLE_EQ(grid.get_dy(), 0.04);
}

TEST(Grid2D, SizeWithoutGhostsEqualsInterior) {
    Grid2D grid(10, 20, 0.0, 1.0, 0.0, 1.0);

    EXPECT_EQ(grid.get_size_x(), 10);
    EXPECT_EQ(grid.get_size_y(), 20);
}

TEST(Grid2D, SizeIncludesGhosts) {
    Grid2D grid(10, 20, 0.0, 1.0, 0.0, 1.0);
    grid.ng = 2;

    EXPECT_EQ(grid.get_size_x(), 14);
    EXPECT_EQ(grid.get_size_y(), 24);
}

TEST(Grid2D, CellCentersUseStoredDx) {
    Grid2D grid(4, 4, 0.0, 1.0, 0.0, 2.0);

    // dx = 1/3, first interior cell center: 0 + 0.5 * dx
    EXPECT_NEAR(grid.get_x(0), 0.5 / 3.0, 1e-12);
    EXPECT_NEAR(grid.get_y(0), 0.5 * (2.0 / 3.0), 1e-12);
    EXPECT_NEAR(grid.get_x(1), 1.5 / 3.0, 1e-12);
}

TEST(Grid2D, GhostOffsetShiftsCoordinates) {
    Grid2D grid(4, 4, 0.0, 1.0, 0.0, 1.0);
    const double x_no_ghost = grid.get_x(0);

    grid.ng = 1;
    EXPECT_NEAR(grid.get_x(0), x_no_ghost - grid.dx, 1e-12);
}
