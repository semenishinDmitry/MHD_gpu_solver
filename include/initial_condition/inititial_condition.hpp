#pragma once

#include "grid/grid.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/mhd_types.hpp"
#include "state/state_field.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

using InitialCondition2D = MHDPrimitive (*)(double x, double y);

enum class InitialConditionType { Uniform, SineWave, BlastWave, OrszagTang };

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

inline MHDPrimitive blast_wave_initial_condition(double /*x*/, double /*y*/)
{
    MHDPrimitive W{};
    W.rho = 1.0;
    W.pressure = 1.0;
    W.bx = 1.0;
    return W;
}

// Classic Orszag-Tang vortex on [0,1]^2 (gamma = 5/3).
inline MHDPrimitive orszag_tang_initial_condition(double x, double y)
{
    constexpr double pi = std::numbers::pi;
    MHDPrimitive W{};
    W.rho = 25.0 / (36.0 * pi);
    W.vx = -std::sin(2.0 * pi * y);
    W.vy = std::sin(2.0 * pi * x);
    W.vz = 0.0;
    W.pressure = 5.0 / (12.0 * pi);
    W.bx = -std::sin(2.0 * pi * y);
    W.by = std::sin(4.0 * pi * x);
    W.bz = 0.0;
    W.psi = 0.0;
    return W;
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
