#include <gtest/gtest.h>

#include "initial_condition/inititial_condition.hpp"

#include <cmath>
#include <numbers>

TEST(InitialCondition, UniformReturnsConstantState) {
    const auto state = uniform_initial_condition(0.3, 0.7);

    EXPECT_DOUBLE_EQ(state.rho, 1.0);
    EXPECT_DOUBLE_EQ(state.vx, 0.0);
    EXPECT_DOUBLE_EQ(state.vy, 0.0);
    EXPECT_DOUBLE_EQ(state.vz, 0.0);
    EXPECT_DOUBLE_EQ(state.E, 1.0);
    EXPECT_DOUBLE_EQ(state.Bx, 0.0);
    EXPECT_DOUBLE_EQ(state.By, 0.0);
    EXPECT_DOUBLE_EQ(state.Bz, 0.0);
}

TEST(InitialCondition, SineWaveDependsOnX) {
    const double x = 0.25;
    const auto state = sine_wave_initial_condition(x, 0.0);
    const double expected_rho = 1.0 + 0.1 * std::sin(2.0 * std::numbers::pi * x);

    EXPECT_NEAR(state.rho, expected_rho, 1e-12);
    EXPECT_DOUBLE_EQ(state.E, 1.0);
    EXPECT_DOUBLE_EQ(state.Bx, 0.0);
}

TEST(InitialCondition, BlastWaveHasMagneticField) {
    const auto state = blast_wave_initial_condition(0.1, 0.2);

    EXPECT_DOUBLE_EQ(state.rho, 1.0);
    EXPECT_DOUBLE_EQ(state.E, 1.0);
    EXPECT_DOUBLE_EQ(state.Bx, 1.0);
    EXPECT_DOUBLE_EQ(state.By, 0.0);
    EXPECT_DOUBLE_EQ(state.Bz, 0.0);
}

TEST(InitialCondition, InitializeStateFieldUniform) {
    Grid2D grid(4, 3, 0.0, 1.0, 0.0, 1.0);
    StateField field(grid.get_size_x(), grid.get_size_y());

    initialize_state_field(field, grid, InitialConditionType::Uniform);

    for (int j = 0; j < grid.ny; ++j) {
        for (int i = 0; i < grid.nx; ++i) {
            const int idx = field.index(i, j);
            EXPECT_DOUBLE_EQ(field.rho[idx], 1.0);
            EXPECT_DOUBLE_EQ(field.mx[idx], 0.0);
            EXPECT_DOUBLE_EQ(field.my[idx], 0.0);
            EXPECT_DOUBLE_EQ(field.mz[idx], 0.0);
            EXPECT_DOUBLE_EQ(field.energy[idx], 1.0);
            EXPECT_DOUBLE_EQ(field.bx[idx], 0.0);
        }
    }
}

TEST(InitialCondition, InitializeStateFieldSineWave) {
    Grid2D grid(8, 2, 0.0, 1.0, 0.0, 1.0);
    StateField field(grid.get_size_x(), grid.get_size_y());

    initialize_state_field(field, grid, InitialConditionType::SineWave);

    for (int i = 0; i < grid.nx; ++i) {
        const int idx = field.index(i, 0);
        const double x = grid.get_x(i);
        const double expected_rho = 1.0 + 0.1 * std::sin(2.0 * std::numbers::pi * x);
        EXPECT_NEAR(field.rho[idx], expected_rho, 1e-12);
    }
}

TEST(InitialCondition, InitializeStateFieldBlastWave) {
    Grid2D grid(3, 3, 0.0, 1.0, 0.0, 1.0);
    StateField field(grid.get_size_x(), grid.get_size_y());

    initialize_state_field(field, grid, InitialConditionType::BlastWave);

    const int idx = field.index(1, 1);
    EXPECT_DOUBLE_EQ(field.rho[idx], 1.0);
    EXPECT_DOUBLE_EQ(field.bx[idx], 1.0);
    EXPECT_DOUBLE_EQ(field.energy[idx], 1.0);
}
