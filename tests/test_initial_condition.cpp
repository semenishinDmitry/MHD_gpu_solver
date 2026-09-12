#include <gtest/gtest.h>

#include "grid/grid.hpp"
#include "initial_condition/inititial_condition.hpp"
#include "mhd/mhd_physics.hpp"

#include <cmath>
#include <numbers>

TEST(InitialCondition, UniformReturnsConstantPrimitive)
{
    const auto W = uniform_initial_condition(0.3, 0.7);
    EXPECT_DOUBLE_EQ(W.rho, 1.0);
    EXPECT_DOUBLE_EQ(W.pressure, 1.0);
    EXPECT_DOUBLE_EQ(W.psi, 0.0);
}

TEST(InitialCondition, OrszagTangBasicProperties)
{
    const auto W = orszag_tang_initial_condition(0.25, 0.1);
    EXPECT_NEAR(W.rho, 25.0 / (36.0 * std::numbers::pi), 1e-12);
    EXPECT_NEAR(W.pressure, 5.0 / (12.0 * std::numbers::pi), 1e-12);
    EXPECT_NEAR(W.vx, -std::sin(2.0 * std::numbers::pi * 0.1), 1e-12);
}

TEST(InitialCondition, InitializeStateFieldUniformUsesEnergy)
{
    constexpr double gamma = 5.0 / 3.0;
    Grid2D grid(4, 3, 0.0, 1.0, 0.0, 1.0, 2);
    StateField field(grid.get_size_x(), grid.get_size_y());
    initialize_state_field(field, grid, InitialConditionType::Uniform, gamma);

    const MHDState expected = to_conservative(uniform_initial_condition(0.0, 0.0), gamma);
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            EXPECT_NEAR(field.get_state(i, j).energy, expected.energy, 1e-12);
            EXPECT_NEAR(field.get_state(i, j).psi, 0.0, 1e-12);
        }
    }
}
