#pragma once

#include "grid/grid.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/mhd_types.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

using InitialCondition2D = MHDPrimitive (*)(double x, double y);

enum class InitialConditionType {
    Uniform,
    SineWave,
    BlastWave,
    OrszagTang,
    Sod,
    BrioWu,
    Rotor,
    KelvinHelmholtz,
    AlfvenWave,
};

inline MHDPrimitive uniform_initial_condition(double /*x*/, double /*y*/)
{
    return MHDPrimitive{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
}

inline MHDPrimitive sine_wave_initial_condition(double x, double /*y*/)
{
    MHDPrimitive W{};
    W.rho = 1.0 + 0.1 * std::sin(2.0 * std::numbers::pi * x);
    W.pressure = 1.0;
    return W;
}

// Pressurized disk blast in a magnetized medium (domain [0,1]^2).
inline MHDPrimitive blast_wave_initial_condition(double x, double y)
{
    constexpr double x0 = 0.5;
    constexpr double y0 = 0.5;
    constexpr double r0 = 0.1;
    const double r = std::hypot(x - x0, y - y0);

    MHDPrimitive W{};
    W.rho = 1.0;
    W.pressure = (r < r0) ? 10.0 : 0.1;
    W.bx = 1.0 / std::sqrt(2.0);
    W.by = 1.0 / std::sqrt(2.0);
    return W;
}

// Classic Orszag–Tang vortex on [0,1]^2 (gamma = 5/3).
inline MHDPrimitive orszag_tang_initial_condition(double x, double y)
{
    constexpr double pi = std::numbers::pi;
    MHDPrimitive W{};
    W.rho = 25.0 / (36.0 * pi);
    W.vx = -std::sin(2.0 * pi * y);
    W.vy = std::sin(2.0 * pi * x);
    W.pressure = 5.0 / (12.0 * pi);
    W.bx = -std::sin(2.0 * pi * y);
    W.by = std::sin(4.0 * pi * x);
    return W;
}

// Sod shock tube along x (hydro; B = 0). Use gamma = 1.4. Domain [0,1].
inline MHDPrimitive sod_initial_condition(double x, double /*y*/)
{
    MHDPrimitive W{};
    if (x < 0.5) {
        W.rho = 1.0;
        W.pressure = 1.0;
    } else {
        W.rho = 0.125;
        W.pressure = 0.1;
    }
    return W;
}

// Brio–Wu MHD shock tube along x. Use gamma = 2. Domain [0,1].
inline MHDPrimitive brio_wu_initial_condition(double x, double /*y*/)
{
    MHDPrimitive W{};
    W.bx = 0.75;
    if (x < 0.5) {
        W.rho = 1.0;
        W.pressure = 1.0;
        W.by = 1.0;
    } else {
        W.rho = 0.125;
        W.pressure = 0.1;
        W.by = -1.0;
    }
    return W;
}

// MHD rotor (Balsara-like) on [0,1]^2, gamma = 5/3.
inline MHDPrimitive rotor_initial_condition(double x, double y)
{
    constexpr double x0 = 0.5;
    constexpr double y0 = 0.5;
    constexpr double r0 = 0.1;
    constexpr double r1 = 0.115;
    const double dx = x - x0;
    const double dy = y - y0;
    const double r = std::hypot(dx, dy);

    MHDPrimitive W{};
    W.bx = 5.0 / std::sqrt(4.0 * std::numbers::pi);
    W.pressure = 1.0;

    double f = 0.0;
    if (r <= r0) {
        f = 1.0;
        W.rho = 10.0;
    } else if (r >= r1) {
        f = 0.0;
        W.rho = 1.0;
    } else {
        f = (r1 - r) / (r1 - r0);
        W.rho = 1.0 + 9.0 * f;
    }

    constexpr double v0 = 2.0;
    if (r > 0.0) {
        W.vx = -f * v0 * dy / r0;
        W.vy = f * v0 * dx / r0;
    }
    return W;
}

// Double shear Kelvin–Helmholtz layer with a small seed on [0,1]^2.
inline MHDPrimitive kelvin_helmholtz_initial_condition(double x, double y)
{
    constexpr double pi = std::numbers::pi;
    MHDPrimitive W{};
    W.rho = (y > 0.25 && y < 0.75) ? 2.0 : 1.0;
    W.vx = (y > 0.25 && y < 0.75) ? 0.5 : -0.5;
    W.vy = 0.01 * std::sin(2.0 * pi * x);
    W.pressure = 2.5;
    W.bx = 0.0;
    W.by = 0.0;
    return W;
}

// Circularly polarized Alfvén wave on periodic [0,1] (x). Smooth; gamma = 5/3.
// Background: ρ=1, p=0.1, Bx=1. Amplitude δ=1e-3. Exact phase speed v_A = Bx/√ρ = 1.
inline MHDPrimitive alfven_wave_initial_condition(double x, double /*y*/)
{
    constexpr double pi = std::numbers::pi;
    constexpr double delta = 1.0e-3;
    constexpr double rho = 1.0;
    const double phase = 2.0 * pi * x;
    const double va = 1.0; // Bx / sqrt(rho)

    MHDPrimitive W{};
    W.rho = rho;
    W.pressure = 0.1;
    W.bx = 1.0;
    W.by = delta * std::cos(phase);
    W.bz = delta * std::sin(phase);
    // Right-going Alfvén: δv = -δB / sqrt(ρ) for B·x > 0
    W.vy = -(delta / std::sqrt(rho)) * std::cos(phase);
    W.vz = -(delta / std::sqrt(rho)) * std::sin(phase);
    (void)va;
    return W;
}

// Exact Alfvén wave primitives at time t (same background as above).
inline MHDPrimitive alfven_wave_exact(double x, double t)
{
    constexpr double pi = std::numbers::pi;
    constexpr double delta = 1.0e-3;
    constexpr double rho = 1.0;
    constexpr double va = 1.0;
    const double phase = 2.0 * pi * (x - va * t);

    MHDPrimitive W{};
    W.rho = rho;
    W.pressure = 0.1;
    W.bx = 1.0;
    W.by = delta * std::cos(phase);
    W.bz = delta * std::sin(phase);
    W.vy = -(delta / std::sqrt(rho)) * std::cos(phase);
    W.vz = -(delta / std::sqrt(rho)) * std::sin(phase);
    return W;
}

inline InitialConditionType parse_initial_condition_name(const std::string& name)
{
    if (name == "uniform" || name == "Uniform") {
        return InitialConditionType::Uniform;
    }
    if (name == "sine" || name == "SineWave") {
        return InitialConditionType::SineWave;
    }
    if (name == "blast" || name == "BlastWave") {
        return InitialConditionType::BlastWave;
    }
    if (name == "orszag_tang" || name == "OrszagTang") {
        return InitialConditionType::OrszagTang;
    }
    if (name == "sod" || name == "Sod") {
        return InitialConditionType::Sod;
    }
    if (name == "brio_wu" || name == "BrioWu") {
        return InitialConditionType::BrioWu;
    }
    if (name == "rotor" || name == "Rotor") {
        return InitialConditionType::Rotor;
    }
    if (name == "kelvin_helmholtz" || name == "KelvinHelmholtz" || name == "kh") {
        return InitialConditionType::KelvinHelmholtz;
    }
    if (name == "alfven" || name == "alfven_wave" || name == "AlfvenWave") {
        return InitialConditionType::AlfvenWave;
    }
    throw std::invalid_argument("Unknown initial condition: " + name);
}

inline void initialize_state_field(StateField& state_field, const Grid2D& grid,
                                   InitialConditionType initial_condition_type,
                                   double gamma = 5.0 / 3.0)
{
    InitialCondition2D initial_condition = nullptr;

    switch (initial_condition_type) {
    case InitialConditionType::Uniform:
        initial_condition = uniform_initial_condition;
        break;
    case InitialConditionType::SineWave:
        initial_condition = sine_wave_initial_condition;
        break;
    case InitialConditionType::BlastWave:
        initial_condition = blast_wave_initial_condition;
        break;
    case InitialConditionType::OrszagTang:
        initial_condition = orszag_tang_initial_condition;
        break;
    case InitialConditionType::Sod:
        initial_condition = sod_initial_condition;
        break;
    case InitialConditionType::BrioWu:
        initial_condition = brio_wu_initial_condition;
        break;
    case InitialConditionType::Rotor:
        initial_condition = rotor_initial_condition;
        break;
    case InitialConditionType::KelvinHelmholtz:
        initial_condition = kelvin_helmholtz_initial_condition;
        break;
    case InitialConditionType::AlfvenWave:
        initial_condition = alfven_wave_initial_condition;
        break;
    default:
        throw std::invalid_argument("Unknown initial condition type");
    }

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const MHDPrimitive W = initial_condition(grid.get_x(i), grid.get_y(j));
            state_field.set_state(i, j, to_conservative(W, gamma));
        }
    }
}

// Cell-centered central-difference divergence of B (interior only).
inline double max_abs_div_b(const StateField& U, const Grid2D& grid)
{
    double max_abs = 0.0;
    const double inv_2dx = 0.5 / grid.dx;
    const double inv_2dy = 0.5 / grid.dy;

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const double dbx = (U.bx[U.index(i + 1, j)] - U.bx[U.index(i - 1, j)]) * inv_2dx;
            const double dby = (U.by[U.index(i, j + 1)] - U.by[U.index(i, j - 1)]) * inv_2dy;
            max_abs = std::max(max_abs, std::abs(dbx + dby));
        }
    }
    return max_abs;
}
