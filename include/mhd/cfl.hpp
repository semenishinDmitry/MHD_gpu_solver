#pragma once

#include "grid/grid.hpp"
#include "mhd/compat.hpp"
#include "mhd/kernels/cpu_rows.hpp"
#include "mhd/mask.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/mhd_types.hpp"
#include "mhd/parallel.hpp"
#include "physics_config/mhd_config.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <atomic>
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
inline CFLResult compute_cfl_dt(const StateField& U, const Grid2D& grid, double gamma,
                                double cfl_number,
                                const NonIdealConfig& nonideal = NonIdealConfig::ideal(),
                                const FluidMask* mask = nullptr)
{
    if (cfl_number <= 0.0) {
        throw std::invalid_argument("CFL number must be positive");
    }
    if (U.nx != grid.get_size_x() || U.ny != grid.get_size_y()) {
        throw std::invalid_argument("StateField size does not match Grid2D");
    }

    const double inv_dx = 1.0 / grid.dx;
    const double inv_dy = 1.0 / grid.dy;

    const bool use_simd = mhd::simd_enabled();
    const int ng = grid.ng;
    const int nx_tot = U.nx;
    const long long cells = static_cast<long long>(grid.nx) * grid.ny;

    // Max is exact and order-independent, so the threaded reduction is deterministic.
    std::atomic<double> c_h_max{0.0};
    mhd::parallel_rows(grid.j_begin(), grid.j_end(), cells, [&](int j0, int j1, int /*tid*/) {
        double local = 0.0;
        for (int j = j0; j < j1; ++j) {
            const int row = j * nx_tot;
            if (mask == nullptr) {
                local = std::max(local, mhd::kernels::max_signal_speed_row(
                                            U, row + grid.i_begin(), grid.nx, gamma, use_simd));
                continue;
            }
            for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
                if (!mask->is_fluid_global(i, j, ng)) {
                    continue;
                }
                local = std::max(local,
                                 mhd::kernels::max_signal_speed_row(U, row + i, 1, gamma, false));
            }
        }
        double cur = c_h_max.load(std::memory_order_relaxed);
        while (local > cur &&
               !c_h_max.compare_exchange_weak(cur, local, std::memory_order_relaxed)) {
        }
    });
    const double c_h = c_h_max.load();

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
