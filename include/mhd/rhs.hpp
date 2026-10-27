#pragma once

#include "grid/grid.hpp"
#include "mhd/kernels/cpu_rows.hpp"
#include "mhd/mask.hpp"
#include "mhd/mhd_types.hpp"
#include "mhd/nonideal.hpp"
#include "mhd/parallel.hpp"
#include "mhd/simd.hpp"
#include "physics_config/mhd_config.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

// Scratch for MUSCL-GLM (+ optional non-ideal) RHS. Owned by TimeIntegratorWorkspace.
//
// Face fluxes are never stored for the whole grid: each thread keeps three flux rows
// (x-faces of row j, y-faces below and above row j) and assembles the RHS of row j directly.
struct RHSWorkspace {
    PrimitiveField prim;
    std::vector<double> Jx;
    std::vector<double> Jy;
    std::vector<double> Jz;

    int row_len = 0;
    // Per-thread scratch: 3 flux rows x 9 components x row_len.
    std::vector<std::vector<double>> row_scratch;

    RHSWorkspace() = default;

    RHSWorkspace(int nx_tot, int ny_tot)
        : prim(nx_tot, ny_tot),
          Jx(static_cast<std::size_t>(nx_tot) * static_cast<std::size_t>(ny_tot), 0.0),
          Jy(static_cast<std::size_t>(nx_tot) * static_cast<std::size_t>(ny_tot), 0.0),
          Jz(static_cast<std::size_t>(nx_tot) * static_cast<std::size_t>(ny_tot), 0.0),
          row_len(nx_tot)
    {
        ensure_threads(mhd::num_threads());
    }

    // Grows the per-thread scratch after mhd::set_num_threads(); no-op in steady state.
    void ensure_threads(int n)
    {
        const std::size_t want = static_cast<std::size_t>(std::max(1, n));
        const std::size_t len = static_cast<std::size_t>(27) * static_cast<std::size_t>(row_len);
        if (row_scratch.size() < want) {
            row_scratch.resize(want);
        }
        for (auto& s : row_scratch) {
            if (s.size() != len) {
                s.assign(len, 0.0);
            }
        }
    }
};

inline void fill_primitives(const StateField& U, PrimitiveField& W, double gamma)
{
    const bool use_simd = mhd::simd_enabled();
    mhd::parallel_range(U.rho.size(), mhd::simd::kWidth,
                        [&](std::size_t lo, std::size_t hi, int /*tid*/) {
                            mhd::kernels::primitive_range(U, W, lo, hi, gamma, use_simd);
                        });
}

namespace detail {

// Non-ideal electric-field contribution to the x-faces of row j (faces i_begin .. i_end).
inline void add_nonideal_x_row(const PrimitiveField& W, const RHSWorkspace& work,
                               const Grid2D& grid, int j, const mhd::kernels::FluxRow& fx,
                               const NonIdealConfig& cfg)
{
    const int nx = W.nx;
    for (int k = 0; k <= grid.nx; ++k) {
        const int i = grid.i_begin() + k;
        const int il = j * nx + (i - 1);
        const int ir = j * nx + i;
        const Vec3 J =
            avg3(work.Jx[il], work.Jy[il], work.Jz[il], work.Jx[ir], work.Jy[ir], work.Jz[ir]);
        const Vec3 B = avg3(W.bx[il], W.by[il], W.bz[il], W.bx[ir], W.by[ir], W.bz[ir]);
        const Vec3 E = nonideal_electric_field(J, B, cfg);
        // F_by += -Ez, F_bz += +Ey, F_E += (B x E)_x = By Ez - Bz Ey
        fx.c[6][k] += -E.z;
        fx.c[7][k] += E.y;
        fx.c[4][k] += B.y * E.z - B.z * E.y;
    }
}

// Non-ideal contribution to the y-faces j-1/2 of row j (cells i_begin .. i_end-1).
inline void add_nonideal_y_row(const PrimitiveField& W, const RHSWorkspace& work,
                               const Grid2D& grid, int j, const mhd::kernels::FluxRow& fy,
                               const NonIdealConfig& cfg)
{
    const int nx = W.nx;
    for (int k = 0; k < grid.nx; ++k) {
        const int i = grid.i_begin() + k;
        const int ib = (j - 1) * nx + i;
        const int it = j * nx + i;
        const Vec3 J =
            avg3(work.Jx[ib], work.Jy[ib], work.Jz[ib], work.Jx[it], work.Jy[it], work.Jz[it]);
        const Vec3 B = avg3(W.bx[ib], W.by[ib], W.bz[ib], W.bx[it], W.by[it], W.bz[it]);
        const Vec3 E = nonideal_electric_field(J, B, cfg);
        // F_bx += +Ez, F_bz += -Ex, F_E += (B x E)_y = Bz Ex - Bx Ez
        fy.c[5][k] += E.z;
        fy.c[7][k] += -E.x;
        fy.c[4][k] += B.z * E.x - B.x * E.z;
    }
}

inline void assemble_component(double* MHD_RESTRICT out, const double* MHD_RESTRICT fx,
                               const double* MHD_RESTRICT gb, const double* MHD_RESTRICT gt, int n,
                               double inv_dx, double inv_dy)
{
    for (int k = 0; k < n; ++k) {
        out[k] = -(fx[k + 1] - fx[k]) * inv_dx - (gt[k] - gb[k]) * inv_dy;
    }
}

using ComponentPtrs = std::array<double*, 9>;

inline ComponentPtrs state_ptrs(StateField& U)
{
    return {U.rho.data(), U.mx.data(), U.my.data(), U.mz.data(), U.energy.data(),
            U.bx.data(),  U.by.data(), U.bz.data(), U.psi.data()};
}

} // namespace detail

// Ideal / non-ideal GLM-MHD spatial operator (MUSCL + HLL + Dedner GLM).
// Non-ideal Ohmic / Hall / Ambipolar terms are added only when enabled in `nonideal`.
//
// Row-fused, SIMD-vectorized and multithreaded. compute_rhs_reference() in rhs_reference.hpp
// is the original scalar implementation and agrees with this one to round-off.
inline void compute_rhs(const StateField& U, StateField& rhs, const Grid2D& grid,
                        RHSWorkspace& work, double gamma, double c_h, double glm_alpha,
                        SlopeLimiter limiter,
                        const NonIdealConfig& nonideal = NonIdealConfig::ideal(),
                        const FluidMask* mask = nullptr)
{
#ifndef NDEBUG
    if (U.nx != grid.get_size_x() || U.ny != grid.get_size_y()) {
        throw std::invalid_argument("StateField size does not match Grid2D");
    }
    if (rhs.nx != U.nx || rhs.ny != U.ny) {
        throw std::invalid_argument("rhs StateField size does not match U");
    }
    if (grid.ng < 2) {
        throw std::invalid_argument("compute_rhs (MUSCL) requires grid.ng >= 2");
    }
    if (work.prim.nx != U.nx || work.row_len != U.nx) {
        throw std::invalid_argument("RHSWorkspace size does not match StateField");
    }
#endif

    fill_primitives(U, work.prim, gamma);
    const PrimitiveField& W = work.prim;

    const bool has_nonideal = nonideal.any();
    if (has_nonideal) {
        compute_current(W, grid, work.Jx.data(), work.Jy.data(), work.Jz.data());
    }

    work.ensure_threads(mhd::num_threads());

    const double inv_dx = 1.0 / grid.dx;
    const double inv_dy = 1.0 / grid.dy;
    const double h = std::min(grid.dx, grid.dy);
    const double psi_damp = (c_h > 0.0 && glm_alpha > 0.0) ? (c_h * glm_alpha / h) : 0.0;
    const bool use_simd = mhd::simd_enabled();

    const mhd::kernels::PrimPtrs P(W);
    const std::ptrdiff_t nx_tot = U.nx;
    const int nxi = grid.nx;
    const int ib = grid.i_begin();
    const int ng = grid.ng;
    const detail::ComponentPtrs R = detail::state_ptrs(rhs);
    const double* MHD_RESTRICT psi_now = U.psi.data();
    const long long cells = static_cast<long long>(grid.nx) * grid.ny;

    mhd::parallel_rows(grid.j_begin(), grid.j_end(), cells, [&](int j0, int j1, int tid) {
        double* scratch = work.row_scratch[static_cast<std::size_t>(tid)].data();
        const std::size_t len = static_cast<std::size_t>(work.row_len);
        mhd::kernels::FluxRow fx{};
        mhd::kernels::FluxRow fy_lo{};
        mhd::kernels::FluxRow fy_hi{};
        for (int c = 0; c < 9; ++c) {
            fx.c[c] = scratch + static_cast<std::size_t>(c) * len;
            fy_lo.c[c] = scratch + static_cast<std::size_t>(9 + c) * len;
            fy_hi.c[c] = scratch + static_cast<std::size_t>(18 + c) * len;
        }

        auto y_faces = [&](int j, const mhd::kernels::FluxRow& out) {
            mhd::kernels::face_row<true>(limiter, P, j * nx_tot + ib, nx_tot, nxi, gamma, c_h, out,
                                         use_simd);
            if (has_nonideal) {
                detail::add_nonideal_y_row(W, work, grid, j, out, nonideal);
            }
        };

        y_faces(j0, fy_lo);
        for (int j = j0; j < j1; ++j) {
            const std::ptrdiff_t row = j * nx_tot + ib;
            mhd::kernels::face_row<false>(limiter, P, row, 1, nxi + 1, gamma, c_h, fx, use_simd);
            if (has_nonideal) {
                detail::add_nonideal_x_row(W, work, grid, j, fx, nonideal);
            }
            y_faces(j + 1, fy_hi);

            for (int c = 0; c < 9; ++c) {
                detail::assemble_component(R[c] + row, fx.c[c], fy_lo.c[c], fy_hi.c[c], nxi, inv_dx,
                                           inv_dy);
            }
            if (psi_damp != 0.0) {
                double* MHD_RESTRICT rpsi = R[8] + row;
                const double* MHD_RESTRICT upsi = psi_now + row;
                for (int k = 0; k < nxi; ++k) {
                    rpsi[k] -= psi_damp * upsi[k];
                }
            }
            if (mask != nullptr) {
                const std::uint8_t* fluid =
                    mask->fluid.data() + static_cast<std::size_t>(j - ng) * mask->nx;
                for (int k = 0; k < nxi; ++k) {
                    if (fluid[k] == 0) {
                        for (int c = 0; c < 9; ++c) {
                            R[c][row + k] = 0.0;
                        }
                    }
                }
            }
            std::swap(fy_lo, fy_hi);
        }
    });
}
