#include <gtest/gtest.h>

#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "mhd/cfl.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/mhd_types.hpp"
#include "mhd/time_integrator.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr double kGamma = 5.0 / 3.0;

void fill_uniform(StateField& U, const Grid2D& grid, const MHDState& state)
{
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            U.set_state(i, j, state);
        }
    }
}

MHDState sum_interior(const StateField& U, const Grid2D& grid)
{
    MHDState sum{};
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const MHDState S = U.get_state(i, j);
            sum.rho += S.rho;
            sum.mx += S.mx;
            sum.my += S.my;
            sum.mz += S.mz;
            sum.energy += S.energy;
            sum.bx += S.bx;
            sum.by += S.by;
            sum.bz += S.bz;
        }
    }
    return sum;
}

} // namespace

TEST(CFL, ScalesWithMeshSpacingAndReportsCh)
{
    Grid2D coarse(8, 8, 0.0, 1.0, 0.0, 1.0, 2);
    Grid2D fine(16, 16, 0.0, 1.0, 0.0, 1.0, 2);

    const MHDState state =
        to_conservative(MHDPrimitive{1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, kGamma);

    StateField Uc(coarse.get_size_x(), coarse.get_size_y());
    StateField Uf(fine.get_size_x(), fine.get_size_y());
    fill_uniform(Uc, coarse, state);
    fill_uniform(Uf, fine, state);

    const CFLResult cfl_c = compute_cfl_dt(Uc, coarse, kGamma, 0.5);
    const CFLResult cfl_f = compute_cfl_dt(Uf, fine, kGamma, 0.5);

    EXPECT_NEAR(cfl_c.dt / cfl_f.dt, 2.0, 1e-10);
    EXPECT_NEAR(cfl_c.c_h, cfl_f.c_h, 1e-12);
    EXPECT_GT(cfl_c.c_h, 0.0);
}

TEST(SSPRK2, UniformStateRemainsUniform)
{
    Grid2D grid(16, 16, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());
    TimeIntegratorWorkspace work(grid);
    const BoundaryConditions bc(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic);

    const MHDPrimitive W{1.4, 0.2, -0.1, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0};
    const MHDState state0 = to_conservative(W, kGamma);
    fill_uniform(U, grid, state0);

    SolveParams params;
    params.t_end = 0.05;
    params.cfl = 0.4;
    params.gamma = kGamma;
    params.glm_alpha = 0.1;
    params.limiter = SlopeLimiter::MC;

    const SolveResult result = solve(U, work, grid, bc, params);
    EXPECT_GT(result.steps, 0);

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const MHDState S = U.get_state(i, j);
            EXPECT_NEAR(S.rho, state0.rho, 1e-9);
            EXPECT_NEAR(S.energy, state0.energy, 1e-9);
            EXPECT_NEAR(S.bx, state0.bx, 1e-9);
            EXPECT_NEAR(S.psi, 0.0, 1e-9);
        }
    }
}

TEST(SSPRK2, PeriodicConservesTotals)
{
    Grid2D grid(12, 12, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());
    TimeIntegratorWorkspace work(grid);
    const BoundaryConditions bc(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic);

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            MHDPrimitive W{1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
            W.rho = 1.0 + 0.05 * std::sin(2.0 * 3.141592653589793 * grid.get_x(i));
            U.set_state(i, j, to_conservative(W, kGamma));
        }
    }

    const MHDState sum0 = sum_interior(U, grid);

    SolveParams params;
    params.t_end = 0.02;
    params.cfl = 0.3;
    params.gamma = kGamma;
    params.limiter = SlopeLimiter::Minmod;
    solve(U, work, grid, bc, params);

    const MHDState sum1 = sum_interior(U, grid);
    EXPECT_NEAR(sum1.rho, sum0.rho, 1e-9);
    EXPECT_NEAR(sum1.mx, sum0.mx, 1e-9);
    EXPECT_NEAR(sum1.energy, sum0.energy, 1e-9);
}
