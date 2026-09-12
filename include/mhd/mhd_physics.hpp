#pragma once

#include "mhd/mhd_types.hpp"

#include <algorithm>
#include <cmath>

inline double magnetic_pressure(double bx, double by, double bz)
{
    return 0.5 * (bx * bx + by * by + bz * bz);
}

inline double kinetic_energy_from_primitive(const MHDPrimitive& W)
{
    return 0.5 * W.rho * (W.vx * W.vx + W.vy * W.vy + W.vz * W.vz);
}

inline double kinetic_energy_from_conservative(const MHDState& U)
{
    return 0.5 * (U.mx * U.mx + U.my * U.my + U.mz * U.mz) / U.rho;
}

inline MHDPrimitive to_primitive(const MHDState& U, double gamma)
{
    MHDPrimitive W;
    W.rho = U.rho;
    W.vx = U.mx / U.rho;
    W.vy = U.my / U.rho;
    W.vz = U.mz / U.rho;
    W.bx = U.bx;
    W.by = U.by;
    W.bz = U.bz;
    W.psi = U.psi;

    const double e_kin = kinetic_energy_from_conservative(U);
    const double e_mag = magnetic_pressure(U.bx, U.by, U.bz);
    W.pressure = (gamma - 1.0) * (U.energy - e_kin - e_mag);
    return W;
}

inline MHDState to_conservative(const MHDPrimitive& W, double gamma)
{
    MHDState U;
    U.rho = W.rho;
    U.mx = W.rho * W.vx;
    U.my = W.rho * W.vy;
    U.mz = W.rho * W.vz;
    U.bx = W.bx;
    U.by = W.by;
    U.bz = W.bz;
    U.psi = W.psi;

    const double e_kin = kinetic_energy_from_primitive(W);
    const double e_mag = magnetic_pressure(W.bx, W.by, W.bz);
    U.energy = W.pressure / (gamma - 1.0) + e_kin + e_mag;
    return U;
}

inline double total_pressure(const MHDPrimitive& W)
{
    return W.pressure + magnetic_pressure(W.bx, W.by, W.bz);
}

inline double v_dot_b(const MHDPrimitive& W)
{
    return W.vx * W.bx + W.vy * W.by + W.vz * W.bz;
}

// Ideal MHD + Dedner GLM fluxes.
// Fx(Bx) = psi, Fx(psi) = c_h^2 Bx; energy flux gains + psi * Bx.
inline MHDFlux physical_flux_x(const MHDPrimitive& W, double gamma, double c_h)
{
    const MHDState U = to_conservative(W, gamma);
    const double pstar = total_pressure(W);
    const double vb = v_dot_b(W);

    MHDFlux F{};
    F.rho = U.rho * W.vx;
    F.mx = U.rho * W.vx * W.vx + pstar - W.bx * W.bx;
    F.my = U.rho * W.vx * W.vy - W.bx * W.by;
    F.mz = U.rho * W.vx * W.vz - W.bx * W.bz;
    F.energy = (U.energy + pstar) * W.vx - W.bx * vb + W.psi * W.bx;
    F.bx = W.psi;
    F.by = W.vx * W.by - W.vy * W.bx;
    F.bz = W.vx * W.bz - W.vz * W.bx;
    F.psi = c_h * c_h * W.bx;
    return F;
}

inline MHDFlux physical_flux_y(const MHDPrimitive& W, double gamma, double c_h)
{
    const MHDState U = to_conservative(W, gamma);
    const double pstar = total_pressure(W);
    const double vb = v_dot_b(W);

    MHDFlux G{};
    G.rho = U.rho * W.vy;
    G.mx = U.rho * W.vx * W.vy - W.bx * W.by;
    G.my = U.rho * W.vy * W.vy + pstar - W.by * W.by;
    G.mz = U.rho * W.vy * W.vz - W.by * W.bz;
    G.energy = (U.energy + pstar) * W.vy - W.by * vb + W.psi * W.by;
    G.bx = W.vy * W.bx - W.vx * W.by;
    G.by = W.psi;
    G.bz = W.vy * W.bz - W.vz * W.by;
    G.psi = c_h * c_h * W.by;
    return G;
}

// Backward-compatible wrappers from conservatives.
inline MHDFlux physical_flux_x(const MHDState& U, double gamma, double c_h)
{
    return physical_flux_x(to_primitive(U, gamma), gamma, c_h);
}

inline MHDFlux physical_flux_y(const MHDState& U, double gamma, double c_h)
{
    return physical_flux_y(to_primitive(U, gamma), gamma, c_h);
}

inline double fast_magnetosonic_speed_x(const MHDPrimitive& W, double gamma)
{
    const double cs2 = gamma * W.pressure / W.rho;
    const double b2 = W.bx * W.bx + W.by * W.by + W.bz * W.bz;
    const double va2 = b2 / W.rho;
    const double vax2 = (W.bx * W.bx) / W.rho;

    const double sum = cs2 + va2;
    double disc = sum * sum - 4.0 * cs2 * vax2;
    disc = std::max(disc, 0.0);

    const double cf2 = 0.5 * (sum + std::sqrt(disc));
    return std::sqrt(std::max(cf2, 0.0));
}

inline double fast_magnetosonic_speed_y(const MHDPrimitive& W, double gamma)
{
    const double cs2 = gamma * W.pressure / W.rho;
    const double b2 = W.bx * W.bx + W.by * W.by + W.bz * W.bz;
    const double va2 = b2 / W.rho;
    const double vay2 = (W.by * W.by) / W.rho;

    const double sum = cs2 + va2;
    double disc = sum * sum - 4.0 * cs2 * vay2;
    disc = std::max(disc, 0.0);

    const double cf2 = 0.5 * (sum + std::sqrt(disc));
    return std::sqrt(std::max(cf2, 0.0));
}
