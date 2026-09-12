#pragma once

#include "mhd/mhd_physics.hpp"
#include "mhd/solver_api.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// Scalar diagnostics for verification / regression (no large field dumps).
struct VerificationDiagnostics {
    double t = 0.0;
    int steps = 0;
    double min_rho = 0.0;
    double max_rho = 0.0;
    double min_pressure = 0.0;
    double max_pressure = 0.0;
    double mass = 0.0;
    double total_energy = 0.0;
    double max_div_b = 0.0;
    bool finite = true;
};

inline VerificationDiagnostics compute_diagnostics(const MHDSolver& solver)
{
    VerificationDiagnostics d;
    d.t = solver.time();
    d.steps = solver.steps();
    d.max_div_b = solver.max_div_b();
    d.min_rho = std::numeric_limits<double>::infinity();
    d.max_rho = -std::numeric_limits<double>::infinity();
    d.min_pressure = std::numeric_limits<double>::infinity();
    d.max_pressure = -std::numeric_limits<double>::infinity();

    const auto& grid = solver.grid();
    const auto& U = solver.state();
    const double gamma = solver.params().gamma;
    const double cell = grid.dx * grid.dy;

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const MHDState s = U.get_state(i, j);
            const MHDPrimitive W = to_primitive(s, gamma);
            d.min_rho = std::min(d.min_rho, W.rho);
            d.max_rho = std::max(d.max_rho, W.rho);
            d.min_pressure = std::min(d.min_pressure, W.pressure);
            d.max_pressure = std::max(d.max_pressure, W.pressure);
            d.mass += W.rho * cell;
            d.total_energy += s.energy * cell;
            if (!std::isfinite(W.rho) || !std::isfinite(W.pressure) || !std::isfinite(s.energy) ||
                !std::isfinite(s.bx) || !std::isfinite(s.by) || !std::isfinite(s.bz)) {
                d.finite = false;
            }
        }
    }
    return d;
}

// L2 error of (by, bz, vy, vz) vs exact circular Alfvén wave (interior).
inline double alfven_wave_l2_error(const MHDSolver& solver)
{
    const auto& grid = solver.grid();
    const double gamma = solver.params().gamma;
    const double t = solver.time();
    double num = 0.0;
    double den = 0.0;
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const MHDPrimitive W = to_primitive(solver.state().get_state(i, j), gamma);
            const MHDPrimitive E = alfven_wave_exact(grid.get_x(i), t);
            const double dby = W.by - E.by;
            const double dbz = W.bz - E.bz;
            const double dvy = W.vy - E.vy;
            const double dvz = W.vz - E.vz;
            num += dby * dby + dbz * dbz + dvy * dvy + dvz * dvz;
            den += E.by * E.by + E.bz * E.bz + E.vy * E.vy + E.vz * E.vz;
        }
    }
    if (den == 0.0) {
        return std::sqrt(num);
    }
    return std::sqrt(num / den);
}

inline double alfven_wave_l1_error(const MHDSolver& solver)
{
    const auto& grid = solver.grid();
    const double gamma = solver.params().gamma;
    const double t = solver.time();
    double num = 0.0;
    double den = 0.0;
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const MHDPrimitive W = to_primitive(solver.state().get_state(i, j), gamma);
            const MHDPrimitive E = alfven_wave_exact(grid.get_x(i), t);
            num += std::abs(W.by - E.by) + std::abs(W.bz - E.bz) + std::abs(W.vy - E.vy) +
                   std::abs(W.vz - E.vz);
            den += std::abs(E.by) + std::abs(E.bz) + std::abs(E.vy) + std::abs(E.vz);
        }
    }
    if (den == 0.0) {
        return num;
    }
    return num / den;
}
