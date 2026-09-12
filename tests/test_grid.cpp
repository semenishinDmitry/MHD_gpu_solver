#include <gtest/gtest.h>

#include "grid/grid.hpp"

TEST(Grid2D, DefaultConstructedIsZero)
{
    Grid2D grid;
    EXPECT_EQ(grid.nx, 0);
    EXPECT_EQ(grid.ny, 0);
    EXPECT_EQ(grid.ng, 0);
}

TEST(Grid2D, DefaultGhostDepthIsTwo)
{
    Grid2D grid(100, 50, 0.0, 1.0, -1.0, 1.0);
    EXPECT_EQ(grid.ng, 2);
    EXPECT_DOUBLE_EQ(grid.dx, 0.01);
    EXPECT_DOUBLE_EQ(grid.dy, 0.04);
    EXPECT_EQ(grid.get_size_x(), 104);
    EXPECT_EQ(grid.get_size_y(), 54);
}

TEST(Grid2D, InteriorIndexRange)
{
    Grid2D grid(10, 20, 0.0, 1.0, 0.0, 1.0, 2);
    EXPECT_EQ(grid.i_begin(), 2);
    EXPECT_EQ(grid.i_end(), 12);
    EXPECT_EQ(grid.j_begin(), 2);
    EXPECT_EQ(grid.j_end(), 22);
}

TEST(Grid2D, CellCenters)
{
    Grid2D grid(4, 4, 0.0, 1.0, 0.0, 2.0, 2);
    EXPECT_NEAR(grid.get_x(grid.i_begin()), 0.5 * grid.dx, 1e-12);
    EXPECT_NEAR(grid.get_y(grid.j_begin()), 0.5 * grid.dy, 1e-12);
}
