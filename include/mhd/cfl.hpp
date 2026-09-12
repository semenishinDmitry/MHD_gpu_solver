#pragma once

#include "grid/grid.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/mhd_types.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

struct CFLResult {
    double dt = 0.0;
    double c_h = 0.0; // max_ij max(|v| + c_f)
};

// Multidimensional CFL including GLM cleaning speed:
//   c_h = max_ij max(|v_x| + c_{f,x}, |v_y| + c_{f,y})
//   dt  = cfl / max_ij( max(|v_x|+c_{f,x}, c_h)/dx + max(|v_y|+c_{f,y}, c_h)/dy )
//
// With a global c_h this reduces to dt = cfl / (c_h/dx + c_h/dy).
inline CFLResult compute_cfl_dt(const StateField& U,
                               const Grid2D& grid,
                               double gamma,
                               double cfl_number)
{
    if (cfl_number <= 0.0) {
        throw std::invalid_argument("CFL number must be positive");
    }
    if (U.nx != grid.get_size_x() || U.ny != grid.get_size_y()) {
        throw std::invalid_argument("StateField size does not match Grid2D");
    }

    const double inv_dx = 1.0 / grid.dx;
    const double inv_dy = 1.0 / grid.dy;

    const double* __restrict__ rho = U.rho.data();
    const double* __restrict__ mx = U.mx.data();
    const double* __restrict__ my = U.my.data();
    const double* __restrict__ mz = U.mz.data();
    const double* __restrict__ energy = U.energy.data();
    const double* __restrict__ bx = U.bx.data();
    const double* __restrict__ by = U.by.data();
    const double* __restrict__ bz = U.bz.data();
    const double* __restrict__ psi = U.psi.data();

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
        return result;
    }

    // Global cleaning waves at ±c_h dominate the signal estimate.
    result.dt = cfl_number / (c_h * inv_dx + c_h * inv_dy);
    return result;
}
