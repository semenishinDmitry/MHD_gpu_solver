#include <gtest/gtest.h>

#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/rhs.hpp"
#include "mhd/time_integrator.hpp"
#include "state/state_field.hpp"

#include <vector>

// Invariant tests that document allocation-free hot-path contracts.
// Workspace buffers are allocated once; RK / RHS reuse them.

TEST(PerformanceInvariants, WorkspaceBuffersArePreallocated)
{
    Grid2D grid(16, 16, 0.0, 1.0, 0.0, 1.0, 2);
    TimeIntegratorWorkspace work(grid);

    const auto* rho_star = work.U_star.rho.data();
    const auto* rho_rhs = work.rhs.rho.data();
    const auto* prim = work.rhs_work.prim.rho.data();
    const auto* fx = work.rhs_work.fx.rho.data();
    const auto* jx = work.rhs_work.Jx.data();

    EXPECT_NE(rho_star, nullptr);
    EXPECT_NE(rho_rhs, nullptr);
    EXPECT_NE(prim, nullptr);
    EXPECT_NE(fx, nullptr);
    EXPECT_NE(jx, nullptr);

    EXPECT_EQ(work.U_star.rho.capacity(), work.U_star.rho.size());
    EXPECT_EQ(work.rhs.rho.capacity(), work.rhs.rho.size());
    EXPECT_EQ(work.rhs_work.Jx.capacity(), work.rhs_work.Jx.size());
}

TEST(PerformanceInvariants, RhsReusesSameBufferPointers)
{
    Grid2D grid(12, 12, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());
    TimeIntegratorWorkspace work(grid);
    const BoundaryConditions bc(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic);

    const MHDState s =
        to_conservative(MHDPrimitive{1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0}, 5.0 / 3.0);
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            U.set_state(i, j, s);
        }
    }

    const double* p0 = work.rhs.rho.data();
    const double* f0 = work.rhs_work.fx.rho.data();
    const double* j0 = work.rhs_work.Jx.data();

    apply_boundary_conditions(U, grid, bc);
    compute_rhs(U, work.rhs, grid, work.rhs_work, 5.0 / 3.0, 1.0, 0.1, SlopeLimiter::MC);
    compute_rhs(U, work.rhs, grid, work.rhs_work, 5.0 / 3.0, 1.0, 0.1, SlopeLimiter::MC,
                NonIdealConfig::with_ohmic(1e-3));

    EXPECT_EQ(work.rhs.rho.data(), p0);
    EXPECT_EQ(work.rhs_work.fx.rho.data(), f0);
    EXPECT_EQ(work.rhs_work.Jx.data(), j0);
}

TEST(PerformanceInvariants, SspRk2DoesNotReallocateStateStorage)
{
    Grid2D grid(10, 10, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());
    TimeIntegratorWorkspace work(grid);
    const BoundaryConditions bc(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic);

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            U.set_state(i, j,
                        to_conservative(MHDPrimitive{1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                                        5.0 / 3.0));
        }
    }

    const auto* u_ptr = U.rho.data();
    const auto* star_ptr = work.U_star.rho.data();
    const std::size_t u_cap = U.rho.capacity();
    const std::size_t star_cap = work.U_star.rho.capacity();

    ssp_rk2_step(U, work, grid, bc, 5.0 / 3.0, 1e-4, 2.0, 0.1, SlopeLimiter::Minmod);

    EXPECT_EQ(U.rho.data(), u_ptr);
    EXPECT_EQ(work.U_star.rho.data(), star_ptr);
    EXPECT_EQ(U.rho.capacity(), u_cap);
    EXPECT_EQ(work.U_star.rho.capacity(), star_cap);
}

TEST(PerformanceInvariants, IdealPathSkipsNonIdealWhenDisabled)
{
    NonIdealConfig ideal = NonIdealConfig::ideal();
    EXPECT_FALSE(ideal.any());
    EXPECT_DOUBLE_EQ(ideal.max_diffusivity(), 0.0);

    NonIdealConfig ohm = NonIdealConfig::with_ohmic(0.0);
    // Flag on but eta==0 => any() is false (no work in hot path).
    EXPECT_FALSE(ohm.any());
}
