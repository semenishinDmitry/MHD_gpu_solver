#pragma once

#include "grid/grid.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/mhd_types.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Interior fluid/solid mask for staircase immersed boundaries.
// Layout: row-major over interior cells only, size nx * ny.
//   1 = fluid, 0 = solid
struct FluidMask {
    int nx = 0;
    int ny = 0;
    std::vector<std::uint8_t> fluid;

    FluidMask() = default;

    explicit FluidMask(int nx_, int ny_, bool all_fluid = true)
        : nx(nx_), ny(ny_),
          fluid(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_),
                static_cast<std::uint8_t>(all_fluid ? 1 : 0))
    {
    }

    std::size_t size() const { return fluid.size(); }

    bool empty() const { return fluid.empty(); }

    bool all_fluid() const
    {
        for (std::uint8_t v : fluid) {
            if (v == 0) {
                return false;
            }
        }
        return true;
    }

    int index(int i_int, int j_int) const { return j_int * nx + i_int; }

    bool is_fluid_interior(int i_int, int j_int) const
    {
        if (i_int < 0 || j_int < 0 || i_int >= nx || j_int >= ny) {
            return false;
        }
        return fluid[static_cast<std::size_t>(index(i_int, j_int))] != 0;
    }

    // Global indices include ghosts; only interior cells may be solid.
    bool is_fluid_global(int i, int j, int ng) const
    {
        const int i_int = i - ng;
        const int j_int = j - ng;
        if (i_int < 0 || j_int < 0 || i_int >= nx || j_int >= ny) {
            return true; // ghosts treated as fluid for stencil access after BC fill
        }
        return is_fluid_interior(i_int, j_int);
    }

    void clear_to_fluid()
    {
        std::fill(fluid.begin(), fluid.end(), static_cast<std::uint8_t>(1));
    }

    void set_from_interior(const std::uint8_t* data, std::size_t n, int nx_, int ny_)
    {
        if (n != static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_)) {
            throw std::invalid_argument("FluidMask: size does not match nx*ny");
        }
        nx = nx_;
        ny = ny_;
        fluid.assign(data, data + n);
    }
};

// Reflective (slip) fill of solid cells from neighboring fluid cells.
// For each solid cell, copy density/pressure/B/psi from a fluid neighbor and
// reflect the normal momentum component. If no fluid neighbor exists, leave benign state.
inline void apply_solid_wall_fill(StateField& U, const Grid2D& grid, const FluidMask& mask,
                                  double gamma)
{
    if (mask.empty() || mask.all_fluid()) {
        return;
    }
    if (mask.nx != grid.nx || mask.ny != grid.ny) {
        throw std::invalid_argument("FluidMask size does not match Grid2D interior");
    }

    const int ng = grid.ng;
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            if (mask.is_fluid_global(i, j, ng)) {
                continue;
            }

            // Prefer cardinal fluid neighbors: right, left, top, bottom.
            const int candidates[4][3] = {
                {i + 1, j, 0}, // reflect mx
                {i - 1, j, 0},
                {i, j + 1, 1}, // reflect my
                {i, j - 1, 1},
            };

            bool filled = false;
            for (const auto& c : candidates) {
                const int ni = c[0];
                const int nj = c[1];
                const int axis = c[2];
                if (ni < grid.i_begin() || ni >= grid.i_end() || nj < grid.j_begin() ||
                    nj >= grid.j_end()) {
                    continue;
                }
                if (!mask.is_fluid_global(ni, nj, ng)) {
                    continue;
                }

                MHDState S = U.get_state(ni, nj);
                MHDPrimitive W = to_primitive(S, gamma);
                if (axis == 0) {
                    W.vx = -W.vx;
                } else {
                    W.vy = -W.vy;
                }
                if (W.rho < 1e-12) {
                    W.rho = 1e-12;
                }
                if (W.pressure < 1e-12) {
                    W.pressure = 1e-12;
                }
                U.set_state(i, j, to_conservative(W, gamma));
                filled = true;
                break;
            }

            if (!filled) {
                MHDPrimitive W{};
                W.rho = 1.0;
                W.pressure = 1.0;
                U.set_state(i, j, to_conservative(W, gamma));
            }
        }
    }
}

inline void zero_rhs_on_solids(StateField& rhs, const Grid2D& grid, const FluidMask& mask)
{
    if (mask.empty() || mask.all_fluid()) {
        return;
    }
    const MHDState z{};
    const int ng = grid.ng;
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            if (!mask.is_fluid_global(i, j, ng)) {
                rhs.set_state(i, j, z);
            }
        }
    }
}
