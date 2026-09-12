#include <gtest/gtest.h>

#include "mhd/hll.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/mhd_types.hpp"
#include "mhd/nonideal.hpp"
#include "mhd/reconstruction.hpp"

#include <cmath>

constexpr double kGamma = 5.0 / 3.0;

TEST(PhysicsExtra, TotalEnergyDecomposition)
{
    MHDPrimitive W{1.2, 0.3, -0.2, 0.1, 0.9, 0.4, -0.1, 0.2, 0.0};
    const MHDState U = to_conservative(W, kGamma);

    const double e_kin = kinetic_energy_from_primitive(W);
    const double e_mag = magnetic_pressure(W.bx, W.by, W.bz);
    const double e_int = W.pressure / (kGamma - 1.0);
    EXPECT_NEAR(U.energy, e_int + e_kin + e_mag, 1e-12);
}

TEST(PhysicsExtra, GLMEnergyFluxIncludesPsiBx)
{
    MHDPrimitive W{1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.25};
    const double c_h = 2.0;
    const MHDFlux F = physical_flux_x(W, kGamma, c_h);
    const MHDFlux F0 =
        physical_flux_x(MHDPrimitive{1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0}, kGamma, c_h);
    EXPECT_NEAR(F.energy - F0.energy, W.psi * W.bx, 1e-12);
}

TEST(PhysicsExtra, FastSpeedZeroWhenPressureAndBZero)
{
    MHDPrimitive W{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    EXPECT_NEAR(fast_magnetosonic_speed_x(W, kGamma), 0.0, 1e-14);
}

TEST(PhysicsExtra, HllSymmetricJump)
{
    MHDPrimitive WL{1.0, 0.5, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0};
    MHDPrimitive WR = WL;
    WR.vx = -0.5;
    const auto FL = hll_flux_x(WL, WR, kGamma, 2.0);
    const auto FR = hll_flux_x(WR, WL, kGamma, 2.0);
    // Mass flux should reverse under left-right swap with opposite vx.
    EXPECT_NEAR(FL.rho, -FR.rho, 1e-10);
}

TEST(PhysicsExtra, MusclFallsBackWhenUnphysical)
{
    MHDPrimitive Wmm{1.0, 0, 0, 0, 1.0, 0, 0, 0, 0};
    MHDPrimitive Wm{1.0, 0, 0, 0, 1.0, 0, 0, 0, 0};
    MHDPrimitive W0{1.0, 0, 0, 0, 1.0, 0, 0, 0, 0};
    MHDPrimitive Wp{-10.0, 0, 0, 0, -1.0, 0, 0, 0, 0}; // unphysical neighbor

    MHDPrimitive WL{}, WR{};
    muscl_interface_x(Wmm, Wm, W0, Wp, SlopeLimiter::MC, WL, WR);
    // Fallback to first-order cell averages for the face.
    EXPECT_DOUBLE_EQ(WL.rho, Wm.rho);
    EXPECT_DOUBLE_EQ(WR.rho, W0.rho);
}

TEST(PhysicsExtra, CombinedNonIdealElectricField)
{
    NonIdealConfig cfg;
    cfg.enable_ohmic(0.1).enable_hall(0.2).enable_ambipolar(0.3);
    const Vec3 J{0.0, 0.0, 1.0};
    const Vec3 B{1.0, 0.0, 0.0};
    const Vec3 E = nonideal_electric_field(J, B, cfg);
    // J×B=(0,1,0)
    // Ohmic (0,0,0.1) + Hall (0,0.2,0) + Amb (J×B)×B/B²=(0,0,-1)*0.3
    // => (0, 0.2, -0.2)
    EXPECT_NEAR(E.x, 0.0, 1e-12);
    EXPECT_NEAR(E.y, 0.2, 1e-12);
    EXPECT_NEAR(E.z, -0.2, 1e-12);
}
