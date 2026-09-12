#pragma once

#include "mhd/mhd_physics.hpp"
#include "mhd/mhd_types.hpp"

#include <algorithm>
#include <cmath>

namespace detail {

inline MHDFlux scale_flux(double a, const MHDFlux& F)
{
    return MHDFlux{
        a * F.rho, a * F.mx, a * F.my, a * F.mz, a * F.energy,
        a * F.bx, a * F.by, a * F.bz, a * F.psi,
    };
}

inline MHDFlux add_flux(const MHDFlux& A, const MHDFlux& B)
{
    return MHDFlux{
        A.rho + B.rho, A.mx + B.mx, A.my + B.my, A.mz + B.mz, A.energy + B.energy,
        A.bx + B.bx, A.by + B.by, A.bz + B.bz, A.psi + B.psi,
    };
}

inline MHDState scale_state(double a, const MHDState& U)
{
    return MHDState{
        a * U.rho, a * U.mx, a * U.my, a * U.mz, a * U.energy,
        a * U.bx, a * U.by, a * U.bz, a * U.psi,
    };
}

inline MHDState sub_state(const MHDState& A, const MHDState& B)
{
    return MHDState{
        A.rho - B.rho, A.mx - B.mx, A.my - B.my, A.mz - B.mz, A.energy - B.energy,
        A.bx - B.bx, A.by - B.by, A.bz - B.bz, A.psi - B.psi,
    };
}

inline MHDFlux state_to_flux(const MHDState& U)
{
    return MHDFlux{U.rho, U.mx, U.my, U.mz, U.energy, U.bx, U.by, U.bz, U.psi};
}

inline MHDFlux hll_combine(double sL,
                           double sR,
                           const MHDFlux& FL,
                           const MHDFlux& FR,
                           const MHDState& UL,
                           const MHDState& UR)
{
    constexpr double eps = 1.0e-14;

    if (sL >= 0.0) {
        return FL;
    }
    if (sR <= 0.0) {
        return FR;
    }

    const double denom = sR - sL;
    if (std::abs(denom) < eps) {
        return scale_flux(0.5, add_flux(FL, FR));
    }

    const MHDFlux term1 = scale_flux(sR, FL);
    const MHDFlux term2 = scale_flux(-sL, FR);
    const MHDFlux term3 = state_to_flux(scale_state(sL * sR, sub_state(UR, UL)));
    return scale_flux(1.0 / denom, add_flux(add_flux(term1, term2), term3));
}

} // namespace detail

// HLL with GLM: wave estimates include cleaning speed c_h.
inline MHDFlux hll_flux_x(const MHDPrimitive& WL, const MHDPrimitive& WR, double gamma, double c_h)
{
    const double cfL = fast_magnetosonic_speed_x(WL, gamma);
    const double cfR = fast_magnetosonic_speed_x(WR, gamma);

    const double sL = std::min({WL.vx - cfL, WR.vx - cfR, -c_h});
    const double sR = std::max({WL.vx + cfL, WR.vx + cfR, c_h});

    const MHDState UL = to_conservative(WL, gamma);
    const MHDState UR = to_conservative(WR, gamma);
    const MHDFlux FL = physical_flux_x(WL, gamma, c_h);
    const MHDFlux FR = physical_flux_x(WR, gamma, c_h);
    return detail::hll_combine(sL, sR, FL, FR, UL, UR);
}

inline MHDFlux hll_flux_y(const MHDPrimitive& WB, const MHDPrimitive& WT, double gamma, double c_h)
{
    const double cfB = fast_magnetosonic_speed_y(WB, gamma);
    const double cfT = fast_magnetosonic_speed_y(WT, gamma);

    const double sL = std::min({WB.vy - cfB, WT.vy - cfT, -c_h});
    const double sR = std::max({WB.vy + cfB, WT.vy + cfT, c_h});

    const MHDState UB = to_conservative(WB, gamma);
    const MHDState UT = to_conservative(WT, gamma);
    const MHDFlux FB = physical_flux_y(WB, gamma, c_h);
    const MHDFlux FT = physical_flux_y(WT, gamma, c_h);
    return detail::hll_combine(sL, sR, FB, FT, UB, UT);
}

inline MHDFlux hll_flux_x(const MHDState& left, const MHDState& right, double gamma, double c_h)
{
    return hll_flux_x(to_primitive(left, gamma), to_primitive(right, gamma), gamma, c_h);
}

inline MHDFlux hll_flux_y(const MHDState& bottom, const MHDState& top, double gamma, double c_h)
{
    return hll_flux_y(to_primitive(bottom, gamma), to_primitive(top, gamma), gamma, c_h);
}
