#include <gtest/gtest.h>

#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "initial_condition/inititial_condition.hpp"
#include "mhd/cfl.hpp"
#include "mhd/reconstruction.hpp"
#include "mhd/time_integrator.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <cmath>

TEST(Limiter, MinmodBoundsOnStep)
{
    // Jump: left differences zero/nonzero → slope must vanish across discontinuity.
    EXPECT_DOUBLE_EQ(minmod(1.0, -1.0), 0.0);
    EXPECT_DOUBLE_EQ(minmod(2.0, 3.0), 2.0);
    EXPECT_DOUBLE_EQ(minmod(-4.0, -1.0), -1.0);
}

TEST(Limiter, MinmodBoundsOnSmoothGradient)
{
    // Uniform gradient Δ = 1 → limited slope equals Δ.
    EXPECT_DOUBLE_EQ(limit_slope(1.0, 1.0, SlopeLimiter::Minmod), 1.0);
}

TEST(Limiter, MCIsLessCompressiveThanMinmodOnSmooth)
{
    const double a = 1.0;
    const double b = 1.0;
    EXPECT_DOUBLE_EQ(limit_slope(a, b, SlopeLimiter::MC), 1.0);

    // Steepening pair: MC clips to 2*minmod region.
    const double a2 = 1.0;
    const double b2 = 10.0;
    EXPECT_DOUBLE_EQ(limit_slope(a2, b2, SlopeLimiter::Minmod), 1.0);
    EXPECT_DOUBLE_EQ(limit_slope(a2, b2, SlopeLimiter::MC), 2.0);
}

TEST(Limiter, MUSCLInterfaceOnLinearRamp)
{
    MHDPrimitive Wm{};
    MHDPrimitive W0{};
    MHDPrimitive Wp{};
    MHDPrimitive Wmm{};
    Wm.rho = 1.0;
    W0.rho = 2.0;
    Wp.rho = 3.0;
    Wmm.rho = 0.0;
    Wm.pressure = W0.pressure = Wp.pressure = Wmm.pressure = 1.0;

    MHDPrimitive WL{};
    MHDPrimitive WR{};
    muscl_interface_x(Wmm, Wm, W0, Wp, SlopeLimiter::Minmod, WL, WR);

    // For linear rho, slopes are 1 → face value 1.5 from both sides.
    EXPECT_NEAR(WL.rho, 1.5, 1e-12);
    EXPECT_NEAR(WR.rho, 1.5, 1e-12);
}

TEST(GLM, OrszagTangDivBIsSuppressed)
{
    constexpr double gamma = 5.0 / 3.0;
    Grid2D grid(32, 32, 0.0, 1.0, 0.0, 1.0, 2);
    const BoundaryConditions bc(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic);

    StateField U_clean(grid.get_size_x(), grid.get_size_y());
    StateField U_raw(grid.get_size_x(), grid.get_size_y());
    TimeIntegratorWorkspace work_clean(grid);
    TimeIntegratorWorkspace work_raw(grid);

    initialize_state_field(U_clean, grid, InitialConditionType::OrszagTang, gamma);
    initialize_state_field(U_raw, grid, InitialConditionType::OrszagTang, gamma);

    apply_boundary_conditions(U_clean, grid, bc);
    apply_boundary_conditions(U_raw, grid, bc);

    SolveParams params_clean;
    params_clean.t_end = 0.05;
    params_clean.cfl = 0.4;
    params_clean.gamma = gamma;
    params_clean.glm_alpha = 0.1;
    params_clean.limiter = SlopeLimiter::MC;

    SolveParams params_raw = params_clean;
    params_raw.glm_alpha = 0.0; // no parabolic damping; still has hyperbolic c_h from CFL

    // Raw comparison: force c_h = 0 by using a modified short integrate without GLM fluxes.
    // Run cleaned solve normally.
    solve(U_clean, work_clean, grid, bc, params_clean);

    // Without cleaning: integrate with c_h=0 and alpha=0 via direct RK steps.
    {
        SolveParams p = params_raw;
        double t = 0.0;
        while (t < p.t_end) {
            apply_boundary_conditions(U_raw, grid, bc);
            // Use MHD-only CFL estimate but pass c_h = 0 into the RHS/HLL.
            const CFLResult cfl = compute_cfl_dt(U_raw, grid, p.gamma, p.cfl);
            const double dt = std::min(cfl.dt, p.t_end - t);
            ssp_rk2_step(U_raw, work_raw, grid, bc, p.gamma, dt, /*c_h=*/0.0, /*glm_alpha=*/0.0, p.limiter);
            t += dt;
        }
        apply_boundary_conditions(U_raw, grid, bc);
    }

    const double div_clean = max_abs_div_b(U_clean, grid);
    const double div_raw = max_abs_div_b(U_raw, grid);

    EXPECT_TRUE(std::isfinite(div_clean));
    EXPECT_TRUE(std::isfinite(div_raw));
    EXPECT_GT(div_raw, 0.0);
    // GLM hyperbolic cleaning + damping should suppress monopoles vs unc leaned MHD.
    EXPECT_LT(div_clean, div_raw);
    EXPECT_LT(div_clean / div_raw, 0.75);
}
