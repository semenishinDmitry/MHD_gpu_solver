#include <gtest/gtest.h>

#include "mhd/solver_api.hpp"

TEST(SolverAPI, InitializeAndRunIdeal)
{
    MHDSolver solver(16, 16);
    solver.set_cfl(0.4);
    solver.set_glm_alpha(0.1);
    solver.initialize("orszag_tang");

    EXPECT_DOUBLE_EQ(solver.time(), 0.0);
    const auto r = solver.run(0.02);
    EXPECT_GT(r.steps, 0);
    EXPECT_NEAR(solver.time(), 0.02, 1e-12);
    EXPECT_TRUE(std::isfinite(solver.max_div_b()));
}

TEST(SolverAPI, NonIdealOhmicCanBeEnabled)
{
    MHDSolver solver(16, 16);
    solver.enable_ohmic(0.01);
    solver.initialize("sine");
    const auto r = solver.run(0.01);
    EXPECT_GT(r.steps, 0);
}

TEST(SolverAPI, UnknownInitialConditionThrows)
{
    MHDSolver solver(8, 8);
    EXPECT_THROW(solver.initialize("nope"), std::invalid_argument);
}

TEST(SolverAPI, CopyFieldRoundTripSize)
{
    MHDSolver solver(8, 8);
    solver.initialize("uniform");
    const auto rho = solver.copy_field("rho");
    EXPECT_EQ(rho.size(), static_cast<std::size_t>(solver.size_x() * solver.size_y()));
    EXPECT_THROW(solver.copy_field("nope"), std::invalid_argument);
}

TEST(SolverAPI, AdvanceToIsMonotonic)
{
    MHDSolver solver(8, 8);
    solver.initialize("uniform");
    solver.advance_to(0.01);
    solver.advance_to(0.02);
    EXPECT_NEAR(solver.time(), 0.02, 1e-12);
    EXPECT_THROW(solver.advance_to(0.01), std::invalid_argument);
}
