#pragma once

#include "mhd/mhd_types.hpp"

#include <algorithm>
#include <cmath>

inline double minmod(double a, double b)
{
    if (a * b <= 0.0) {
        return 0.0;
    }
    return (std::abs(a) < std::abs(b)) ? a : b;
}

// Monotonized-central limiter on consecutive differences a = W_i - W_{i-1}, b = W_{i+1} - W_i.
inline double mc_limiter(double a, double b)
{
    return minmod(0.5 * (a + b), minmod(2.0 * a, 2.0 * b));
}

inline double limit_slope(double a, double b, SlopeLimiter limiter)
{
    switch (limiter) {
    case SlopeLimiter::Minmod:
        return minmod(a, b);
    case SlopeLimiter::MC:
        return mc_limiter(a, b);
    }
    return 0.0;
}

inline double component_diff(const MHDPrimitive& A, const MHDPrimitive& B, int comp)
{
    switch (comp) {
    case 0:
        return A.rho - B.rho;
    case 1:
        return A.vx - B.vx;
    case 2:
        return A.vy - B.vy;
    case 3:
        return A.vz - B.vz;
    case 4:
        return A.pressure - B.pressure;
    case 5:
        return A.bx - B.bx;
    case 6:
        return A.by - B.by;
    case 7:
        return A.bz - B.bz;
    case 8:
        return A.psi - B.psi;
    default:
        return 0.0;
    }
}

// Limited slope vector δW for cell centered between Wm (i-1) and Wp (i+1).
inline MHDPrimitive limited_slope(const MHDPrimitive& Wm, const MHDPrimitive& W0,
                                  const MHDPrimitive& Wp, SlopeLimiter limiter)
{
    MHDPrimitive dW{};
    for (int c = 0; c < 9; ++c) {
        const double a = component_diff(W0, Wm, c);
        const double b = component_diff(Wp, W0, c);
        const double s = limit_slope(a, b, limiter);
        switch (c) {
        case 0:
            dW.rho = s;
            break;
        case 1:
            dW.vx = s;
            break;
        case 2:
            dW.vy = s;
            break;
        case 3:
            dW.vz = s;
            break;
        case 4:
            dW.pressure = s;
            break;
        case 5:
            dW.bx = s;
            break;
        case 6:
            dW.by = s;
            break;
        case 7:
            dW.bz = s;
            break;
        case 8:
            dW.psi = s;
            break;
        }
    }
    return dW;
}

inline MHDPrimitive apply_slope(const MHDPrimitive& W0, const MHDPrimitive& dW, double sign)
{
    MHDPrimitive W = W0;
    W.rho += 0.5 * sign * dW.rho;
    W.vx += 0.5 * sign * dW.vx;
    W.vy += 0.5 * sign * dW.vy;
    W.vz += 0.5 * sign * dW.vz;
    W.pressure += 0.5 * sign * dW.pressure;
    W.bx += 0.5 * sign * dW.bx;
    W.by += 0.5 * sign * dW.by;
    W.bz += 0.5 * sign * dW.bz;
    W.psi += 0.5 * sign * dW.psi;
    return W;
}

inline bool is_physical_primitive(const MHDPrimitive& W)
{
    return W.rho > 0.0 && W.pressure > 0.0 && std::isfinite(W.rho) && std::isfinite(W.pressure);
}

// MUSCL interface states from a 4-point stencil (i-2,i-1,i,i+1) for face i-1/2.
// Left from cell i-1, right from cell i. Falls back to 1st order if unphysical.
inline void muscl_interface_x(const MHDPrimitive& Wmm, const MHDPrimitive& Wm,
                              const MHDPrimitive& W0, const MHDPrimitive& Wp, SlopeLimiter limiter,
                              MHDPrimitive& WL, MHDPrimitive& WR)
{
    const MHDPrimitive dWm = limited_slope(Wmm, Wm, W0, limiter);
    const MHDPrimitive dW0 = limited_slope(Wm, W0, Wp, limiter);

    WL = apply_slope(Wm, dWm, +1.0);
    WR = apply_slope(W0, dW0, -1.0);

    if (!is_physical_primitive(WL) || !is_physical_primitive(WR)) {
        WL = Wm;
        WR = W0;
    }
}

inline void muscl_interface_y(const MHDPrimitive& Wmm, const MHDPrimitive& Wm,
                              const MHDPrimitive& W0, const MHDPrimitive& Wp, SlopeLimiter limiter,
                              MHDPrimitive& WB, MHDPrimitive& WT)
{
    muscl_interface_x(Wmm, Wm, W0, Wp, limiter, WB, WT);
}
