#pragma once

#include "grid/grid.hpp"
#include "mhd/compat.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/mhd_types.hpp"
#include "physics_config/mhd_config.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

struct CFLResult {
    double dt = 0.0;
    double c_h = 0.0; // max_ij max(|v| + c_f)
};

// Hyperbolic + optional parabolic/dispersive CFL for non-ideal terms.
//   dt = min(dt_hyp, dt_diff)
//   dt_hyp  = cfl / (c_h/dx + c_h/dy)
//   dt_diff = cfl * h^2 / (2 * d * η_max),  d=2, h=min(dx,dy)
inline CFLResult compute_cfl_dt(const StateField& U,
                               const Grid2D& grid,
                               double gamma,
                               double cfl_number,
                               const NonIdealConfig& nonideal = NonIdealConfig::ideal())
{
    if (cfl_number <= 0.0) {
        throw std::invalid_argument("CFL number must be positive");
    }
    if (U.nx != grid.get_size_x() || U.ny != grid.get_size_y()) {
        throw std::invalid_argument("StateField size does not match Grid2D");
    }

    const double inv_dx = 1.0 / grid.dx;
    const double inv_dy = 1.0 / grid.dy;

    const double* MHD_RESTRICT rho = U.rho.data();
    const double* MHD_RESTRICT mx = U.mx.data();
    const double* MHD_RESTRICT my = U.my.data();
    const double* MHD_RESTRICT mz = U.mz.data();
    const double* MHD_RESTRICT energy = U.energy.data();
    const double* MHD_RESTRICT bx = U.bx.data();
    const double* MHD_RESTRICT by = U.by.data();
    const double* MHD_RESTRICT bz = U.bz.data();
    const double* MHD_RESTRICT psi = U.psi.data();

    double c_h = 0.0;

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        const int row = j * U.nx;
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const int idx = row + i;
            const MHDState state{
                rho[idx], mx[idx], my[idx], mz[idx], energy[idx], bx[idx], by[idx], bz[idx], psi[idx],
            };
            const MHDPrimitive W = to_primitive(state, gamma);
            const double cf_x = fast_magnetosonic_speed_x(W, gamma);
            const double cf_y = fast_magnetosonic_speed_y(W, gamma);
            c_h = std::max(c_h, std::abs(W.vx) + cf_x);
            c_h = std::max(c_h, std::abs(W.vy) + cf_y);
        }
    }

    CFLResult result{};
    result.c_h = c_h;

    if (c_h <= 0.0) {
        result.dt = std::numeric_limits<double>::infinity();
    } else {
        result.dt = cfl_number / (c_h * inv_dx + c_h * inv_dy);
    }

    const double eta_max = nonideal.max_diffusivity();
    if (eta_max > 0.0) {
        constexpr double ndim = 2.0;
        const double h = std::min(grid.dx, grid.dy);
        const double dt_diff = cfl_number * h * h / (2.0 * ndim * eta_max);
        result.dt = std::min(result.dt, dt_diff);
    }

    return result;
}
