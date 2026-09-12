#include <gtest/gtest.h>

#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "mhd/hll.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/mhd_types.hpp"
#include "mhd/rhs.hpp"
#include "mhd/time_integrator.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr double kGamma = 5.0 / 3.0;

MHDPrimitive sample_primitive()
{
    return MHDPrimitive{1.2, 0.3, -0.1, 0.05, 0.8, 0.2, -0.15, 0.05, 0.0};
}

bool is_finite_flux(const MHDFlux& F)
{
    return std::isfinite(F.rho) && std::isfinite(F.mx) && std::isfinite(F.my) &&
           std::isfinite(F.mz) && std::isfinite(F.energy) && std::isfinite(F.bx) &&
           std::isfinite(F.by) && std::isfinite(F.bz) && std::isfinite(F.psi);
}

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
            sum.psi += S.psi;
        }
    }
    return sum;
}

} // namespace

TEST(MHDPhysics, ConservativePrimitiveRoundTrip)
{
    const MHDPrimitive W = sample_primitive();
    const MHDState U = to_conservative(W, kGamma);
    const MHDPrimitive W2 = to_primitive(U, kGamma);

    EXPECT_NEAR(W2.rho, W.rho, 1e-12);
    EXPECT_NEAR(W2.pressure, W.pressure, 1e-12);
    EXPECT_NEAR(W2.psi, W.psi, 1e-12);
}

TEST(MHDPhysics, GLMFluxCarriesPsi)
{
    MHDPrimitive W = sample_primitive();
    W.psi = 0.3;
    W.bx = 0.5;
    const double c_h = 1.7;
    const MHDFlux F = physical_flux_x(W, kGamma, c_h);
    EXPECT_NEAR(F.bx, W.psi, 1e-12);
    EXPECT_NEAR(F.psi, c_h * c_h * W.bx, 1e-12);
}

TEST(HLL, EqualStatesReturnPhysicalFluxX)
{
    const MHDState U = to_conservative(sample_primitive(), kGamma);
    const double c_h = 2.0;
    const MHDFlux F_hll = hll_flux_x(U, U, kGamma, c_h);
    const MHDFlux F_phys = physical_flux_x(U, kGamma, c_h);

    EXPECT_NEAR(F_hll.rho, F_phys.rho, 1e-12);
    EXPECT_NEAR(F_hll.bx, F_phys.bx, 1e-12);
    EXPECT_NEAR(F_hll.psi, F_phys.psi, 1e-12);
}

TEST(HLL, NontrivialStatesAreFinite)
{
    MHDPrimitive WL = sample_primitive();
    MHDPrimitive WR = sample_primitive();
    WR.rho = 0.7;
    WR.vx = -0.4;
    WR.pressure = 1.5;
    WR.bx = -0.1;
    WR.psi = 0.2;

    const MHDFlux Fx =
        hll_flux_x(to_conservative(WL, kGamma), to_conservative(WR, kGamma), kGamma, 1.5);
    EXPECT_TRUE(is_finite_flux(Fx));
}

TEST(RHS, UniformStateGivesNearZero)
{
    Grid2D grid(16, 16, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());
    TimeIntegratorWorkspace work(grid);

    const MHDState state = to_conservative(sample_primitive(), kGamma);
    fill_uniform(U, grid, state);
    apply_boundary_conditions(
        U, grid,
        BoundaryConditions(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic));
    compute_rhs(U, work.rhs, grid, work.rhs_work, kGamma, 1.0, 0.1, SlopeLimiter::MC);

    double max_abs = 0.0;
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const MHDState L = work.rhs.get_state(i, j);
            max_abs = std::max({max_abs, std::abs(L.rho), std::abs(L.mx), std::abs(L.energy),
                                std::abs(L.bx), std::abs(L.by), std::abs(L.psi)});
        }
    }
    EXPECT_NEAR(max_abs, 0.0, 1e-10);
}

TEST(RHS, PeriodicOperatorConservesMassMomentum)
{
    Grid2D grid(12, 10, 0.0, 1.0, 0.0, 1.0, 2);
    StateField U(grid.get_size_x(), grid.get_size_y());
    TimeIntegratorWorkspace work(grid);

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            MHDPrimitive W = sample_primitive();
            W.rho = 1.0 + 0.05 * std::sin(2.0 * 3.141592653589793 * grid.get_x(i));
            W.pressure = 1.0 + 0.02 * std::cos(2.0 * 3.141592653589793 * grid.get_y(j));
            U.set_state(i, j, to_conservative(W, kGamma));
        }
    }

    apply_boundary_conditions(
        U, grid,
        BoundaryConditions(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic));
    compute_rhs(U, work.rhs, grid, work.rhs_work, kGamma, 1.0, 0.1, SlopeLimiter::Minmod);

    const MHDState dUdt_sum = sum_interior(work.rhs, grid);
    EXPECT_NEAR(dUdt_sum.rho, 0.0, 1e-10);
    EXPECT_NEAR(dUdt_sum.mx, 0.0, 1e-10);
    EXPECT_NEAR(dUdt_sum.my, 0.0, 1e-10);
}
