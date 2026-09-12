#include <gtest/gtest.h>

#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/nonideal.hpp"
#include "mhd/time_integrator.hpp"
#include "physics_config/mhd_config.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace {

constexpr double kGamma = 5.0 / 3.0;

double max_abs_bz(const StateField& U, const Grid2D& grid)
{
    double m = 0.0;
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            m = std::max(m, std::abs(U.get_state(i, j).bz));
        }
    }
    return m;
}

} // namespace

TEST(NonIdeal, ElectricFieldOhmicOnly)
{
    const Vec3 J{1.0, -2.0, 0.5};
    const Vec3 B{0.0, 0.0, 1.0};
    const auto E = nonideal_electric_field(J, B, NonIdealConfig::with_ohmic(0.3));
    EXPECT_NEAR(E.x, 0.3, 1e-12);
    EXPECT_NEAR(E.y, -0.6, 1e-12);
    EXPECT_NEAR(E.z, 0.15, 1e-12);
}

TEST(NonIdeal, ElectricFieldHallOnly)
{
    const Vec3 J{0.0, 0.0, 1.0};
    const Vec3 B{1.0, 0.0, 0.0};
    // J × B = (0, 1, 0), |B|=1 => E = η_H (0,1,0)
    const auto E = nonideal_electric_field(J, B, NonIdealConfig::with_hall(0.2));
    EXPECT_NEAR(E.x, 0.0, 1e-12);
    EXPECT_NEAR(E.y, 0.2, 1e-12);
    EXPECT_NEAR(E.z, 0.0, 1e-12);
}

TEST(NonIdeal, ElectricFieldAmbipolarOnly)
{
    const Vec3 J{0.0, 1.0, 0.0};
    const Vec3 B{1.0, 0.0, 0.0};
    // J×B=(0,0,-1), (J×B)×B=(0,-1,0), /B² => E = η_A (0,-1,0)
    const auto E = nonideal_electric_field(J, B, NonIdealConfig::with_ambipolar(0.5));
    EXPECT_NEAR(E.x, 0.0, 1e-12);
    EXPECT_NEAR(E.y, -0.5, 1e-12);
    EXPECT_NEAR(E.z, 0.0, 1e-12);
}

TEST(NonIdeal, IdealPathLeavesUniformStateUnchanged)
{
    Grid2D grid(16, 16, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());
    TimeIntegratorWorkspace work(grid);
    const BoundaryConditions bc(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic);

    const MHDState U0 =
        to_conservative(MHDPrimitive{1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0}, kGamma);
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            U.set_state(i, j, U0);
        }
    }

    SolveParams params;
    params.t_end = 0.02;
    params.cfl = 0.4;
    params.gamma = kGamma;
    params.nonideal = NonIdealConfig::ideal();
    solve(U, work, grid, bc, params);

    EXPECT_NEAR(U.get_state(grid.i_begin(), grid.j_begin()).bx, U0.bx, 1e-10);
}

TEST(NonIdeal, OhmicDiffusesSinusoidalBz)
{
    Grid2D grid(32, 32, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());
    TimeIntegratorWorkspace work(grid);
    const BoundaryConditions bc(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic);

    // Pure magnetic sine wave: v=0, ρ=const, p=const, Bz = sin(2πx)
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            MHDPrimitive W{};
            W.rho = 1.0;
            W.pressure = 1.0;
            W.bz = std::sin(2.0 * std::numbers::pi * grid.get_x(i));
            U.set_state(i, j, to_conservative(W, kGamma));
        }
    }

    const double bz0 = max_abs_bz(U, grid);

    SolveParams params;
    params.t_end = 0.1;
    params.cfl = 0.3;
    params.gamma = kGamma;
    params.glm_alpha = 0.1;
    params.limiter = SlopeLimiter::Minmod;
    params.nonideal = NonIdealConfig::with_ohmic(0.1);

    solve(U, work, grid, bc, params);

    const double bz1 = max_abs_bz(U, grid);
    EXPECT_LT(bz1, 0.9 * bz0);
    EXPECT_GT(bz1, 0.0);
}

TEST(NonIdeal, CFLTightensWithDiffusivity)
{
    Grid2D grid(16, 16, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());
    const MHDState state =
        to_conservative(MHDPrimitive{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, kGamma);
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            U.set_state(i, j, state);
        }
    }

    const CFLResult ideal = compute_cfl_dt(U, grid, kGamma, 0.4, NonIdealConfig::ideal());
    const CFLResult ohmic = compute_cfl_dt(U, grid, kGamma, 0.4, NonIdealConfig::with_ohmic(1.0));
    EXPECT_LT(ohmic.dt, ideal.dt);
}
