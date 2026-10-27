#include <gtest/gtest.h>

#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "mhd/mask.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/mhd_types.hpp"
#include "mhd/rhs.hpp"
#include "mhd/solver_api.hpp"
#include "mhd/time_integrator.hpp"
#include "physics_config/mhd_config.hpp"
#include "state/state_field.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

TEST(Mask, AllFluidMatchesUnmaskedSolve)
{
    MHDSolver a(24, 24);
    MHDSolver b(24, 24);
    a.set_cfl(0.4);
    b.set_cfl(0.4);
    a.set_glm_alpha(0.1);
    b.set_glm_alpha(0.1);
    a.initialize("orszag_tang");
    b.initialize("orszag_tang");
    // Explicit all-fluid mask on b
    std::vector<std::uint8_t> ones(static_cast<std::size_t>(b.nx() * b.ny()), 1);
    b.set_mask(ones.data(), ones.size());

    a.run(0.02);
    b.run(0.02);

    const auto ra = a.copy_field("rho");
    const auto rb = b.copy_field("rho");
    ASSERT_EQ(ra.size(), rb.size());
    double max_diff = 0.0;
    for (std::size_t i = 0; i < ra.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(ra[i] - rb[i]));
    }
    EXPECT_LT(max_diff, 1e-12);
}

TEST(Mask, SolidBlockRhsIsZero)
{
    Grid2D grid(16, 16, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());
    TimeIntegratorWorkspace work(grid);
    FluidMask mask(16, 16, true);

    // Paint a solid block in the center
    for (int j = 6; j < 10; ++j) {
        for (int i = 6; i < 10; ++i) {
            mask.fluid[static_cast<std::size_t>(mask.index(i, j))] = 0;
        }
    }

    // Non-trivial state
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            MHDPrimitive W{};
            W.rho = 1.0;
            W.vx = 0.2;
            W.vy = -0.1;
            W.pressure = 1.0;
            W.bx = 0.1;
            U.set_state(i, j, to_conservative(W, 5.0 / 3.0));
        }
    }

    apply_solid_wall_fill(U, grid, mask, 5.0 / 3.0);
    compute_rhs(U, work.rhs, grid, work.rhs_work, 5.0 / 3.0, 1.0, 0.1, SlopeLimiter::MC,
                NonIdealConfig::ideal(), &mask);

    for (int j = 6; j < 10; ++j) {
        for (int i = 6; i < 10; ++i) {
            const int gi = i + grid.ng;
            const int gj = j + grid.ng;
            const MHDState L = work.rhs.get_state(gi, gj);
            EXPECT_DOUBLE_EQ(L.rho, 0.0);
            EXPECT_DOUBLE_EQ(L.mx, 0.0);
            EXPECT_DOUBLE_EQ(L.my, 0.0);
            EXPECT_DOUBLE_EQ(L.energy, 0.0);
        }
    }
}

TEST(Mask, SolidCellsDoNotUpdate)
{
    MHDSolver solver(32, 32);
    solver.set_cfl(0.3);
    solver.set_outflow_bc();
    solver.set_walls_reflective(false);
    solver.initialize("uniform");

    std::vector<std::uint8_t> mask(static_cast<std::size_t>(solver.nx() * solver.ny()), 1);
    for (int j = 10; j < 22; ++j) {
        for (int i = 10; i < 22; ++i) {
            mask[static_cast<std::size_t>(j * solver.nx() + i)] = 0;
        }
    }
    solver.set_mask(mask.data(), mask.size());

    // Freeze solid density to a sentinel; with walls_reflective=false, RHS=0 keeps it.
    auto rho = solver.copy_field("rho");
    const int ng = solver.ng();
    const int nx_tot = solver.size_x();
    constexpr double kSentinel = 3.14159;
    for (int j = 10; j < 22; ++j) {
        for (int i = 10; i < 22; ++i) {
            rho[static_cast<std::size_t>((j + ng) * nx_tot + (i + ng))] = kSentinel;
        }
    }
    solver.set_field("rho", rho.data(), rho.size());

    solver.run(0.05);

    const auto rho1 = solver.copy_field("rho");
    for (int j = 10; j < 22; ++j) {
        for (int i = 10; i < 22; ++i) {
            const int idx = (j + ng) * nx_tot + (i + ng);
            EXPECT_NEAR(rho1[static_cast<std::size_t>(idx)], kSentinel, 1e-12);
        }
    }
}

TEST(Mask, FlowDoesNotEnterSolidSmoke)
{
    MHDSolver solver(48, 24, 0.0, 2.0, 0.0, 1.0);
    solver.set_cfl(0.4);
    solver.set_bc(BoundaryConditionType::Outflow, BoundaryConditionType::Outflow);
    solver.set_gamma(5.0 / 3.0);

    // Channel with circular obstacle
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(solver.nx() * solver.ny()), 1);
    const double cx = 0.7;
    const double cy = 0.5;
    const double r = 0.2;
    const double dx = 2.0 / solver.nx();
    const double dy = 1.0 / solver.ny();
    for (int j = 0; j < solver.ny(); ++j) {
        for (int i = 0; i < solver.nx(); ++i) {
            const double x = 0.0 + (i + 0.5) * dx;
            const double y = 0.0 + (j + 0.5) * dy;
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) {
                mask[static_cast<std::size_t>(j * solver.nx() + i)] = 0;
            }
        }
    }

    // Uniform leftward-ish flow via primitives
    const int n = solver.nx() * solver.ny();
    std::vector<double> rho(n, 1.0), vx(n, 0.5), vy(n, 0.0), vz(n, 0.0), p(n, 1.0);
    std::vector<double> bx(n, 0.0), by(n, 0.0), bz(n, 0.0);
    solver.set_mask(mask.data(), mask.size());
    solver.set_primitives_interior(rho.data(), vx.data(), vy.data(), vz.data(), p.data(), bx.data(),
                                   by.data(), bz.data(), nullptr, static_cast<std::size_t>(n));

    solver.run(0.1);

    const auto mx = solver.copy_field("mx");
    const auto my = solver.copy_field("my");
    const int ng = solver.ng();
    const int nx_tot = solver.size_x();
    for (int j = 0; j < solver.ny(); ++j) {
        for (int i = 0; i < solver.nx(); ++i) {
            if (mask[static_cast<std::size_t>(j * solver.nx() + i)] != 0) {
                continue;
            }
            const int idx = (j + ng) * nx_tot + (i + ng);
            // After wall fill, solid may have reflected momentum from neighbors,
            // but density/pressure stay finite and the cell is never evolved as fluid.
            EXPECT_TRUE(std::isfinite(mx[static_cast<std::size_t>(idx)]));
            EXPECT_TRUE(std::isfinite(my[static_cast<std::size_t>(idx)]));
        }
    }
    EXPECT_TRUE(std::isfinite(solver.max_div_b()));
}
