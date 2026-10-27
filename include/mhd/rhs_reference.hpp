#pragma once

// Reference implementation of the MUSCL-GLM spatial operator (the original scalar code).
// The production operator is compute_rhs() in rhs.hpp; this file exists so the optimized
// SIMD / multithreaded path can be verified against an independent, simple implementation.

#include "grid/grid.hpp"
#include "mhd/hll.hpp"
#include "mhd/mask.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/mhd_types.hpp"
#include "mhd/nonideal.hpp"
#include "mhd/reconstruction.hpp"
#include "physics_config/mhd_config.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

inline void fill_primitives_reference(const StateField& U, PrimitiveField& W, double gamma)
{
    const std::size_t n = U.rho.size();
    for (std::size_t idx = 0; idx < n; ++idx) {
        const MHDState Ucell{
            U.rho[idx], U.mx[idx], U.my[idx], U.mz[idx],  U.energy[idx],
            U.bx[idx],  U.by[idx], U.bz[idx], U.psi[idx],
        };
        const MHDPrimitive P = to_primitive(Ucell, gamma);
        W.rho[idx] = P.rho;
        W.vx[idx] = P.vx;
        W.vy[idx] = P.vy;
        W.vz[idx] = P.vz;
        W.pressure[idx] = P.pressure;
        W.bx[idx] = P.bx;
        W.by[idx] = P.by;
        W.bz[idx] = P.bz;
        W.psi[idx] = P.psi;
    }
}

inline void store_flux_reference(StateField& F, int i, int j, const MHDFlux& flux)
{
    const int idx = F.index(i, j);
    F.rho[idx] = flux.rho;
    F.mx[idx] = flux.mx;
    F.my[idx] = flux.my;
    F.mz[idx] = flux.mz;
    F.energy[idx] = flux.energy;
    F.bx[idx] = flux.bx;
    F.by[idx] = flux.by;
    F.bz[idx] = flux.bz;
    F.psi[idx] = flux.psi;
}

inline MHDFlux load_flux_reference(const StateField& F, int i, int j)
{
    const int idx = F.index(i, j);
    return MHDFlux{
        F.rho[idx], F.mx[idx], F.my[idx], F.mz[idx],  F.energy[idx],
        F.bx[idx],  F.by[idx], F.bz[idx], F.psi[idx],
    };
}

// Preallocated buffers of the reference operator (same layout as the original RHSWorkspace).
struct RHSReferenceWorkspace {
    PrimitiveField prim;
    StateField fx;
    StateField fy;
    std::vector<double> Jx;
    std::vector<double> Jy;
    std::vector<double> Jz;

    RHSReferenceWorkspace(int nx_tot, int ny_tot)
        : prim(nx_tot, ny_tot), fx(nx_tot, ny_tot), fy(nx_tot, ny_tot),
          Jx(static_cast<std::size_t>(nx_tot) * static_cast<std::size_t>(ny_tot), 0.0),
          Jy(static_cast<std::size_t>(nx_tot) * static_cast<std::size_t>(ny_tot), 0.0),
          Jz(static_cast<std::size_t>(nx_tot) * static_cast<std::size_t>(ny_tot), 0.0)
    {
    }
};

// Original (pre-optimization) spatial operator: AoS-style per-face evaluation with full-size
// face-flux arrays, serial, no SIMD. Kept as the reference that the optimized compute_rhs()
// is tested against (tests/test_cpu_kernels.cpp) and as the "before" baseline in
// tools/mhd_backend_compare.cpp. Not used by the solver.
inline void compute_rhs_reference(const StateField& U, StateField& rhs, const Grid2D& grid,
                                  RHSReferenceWorkspace& work, double gamma, double c_h,
                                  double glm_alpha, SlopeLimiter limiter,
                                  const NonIdealConfig& nonideal = NonIdealConfig::ideal(),
                                  const FluidMask* mask = nullptr)
{
    PrimitiveField& prim = work.prim;
    StateField& fx = work.fx;
    StateField& fy = work.fy;
    std::vector<double>& Jx = work.Jx;
    std::vector<double>& Jy = work.Jy;
    std::vector<double>& Jz = work.Jz;
    fill_primitives_reference(U, prim, gamma);
    const PrimitiveField& W = prim;

    const double inv_dx = 1.0 / grid.dx;
    const double inv_dy = 1.0 / grid.dy;

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i <= grid.i_end(); ++i) {
            const MHDPrimitive Wmm = W.get(i - 2, j);
            const MHDPrimitive Wm = W.get(i - 1, j);
            const MHDPrimitive W0 = W.get(i, j);
            const MHDPrimitive Wp = W.get(i + 1, j);

            MHDPrimitive WL{};
            MHDPrimitive WR{};
            muscl_interface_x(Wmm, Wm, W0, Wp, limiter, WL, WR);
            store_flux_reference(fx, i, j, hll_flux_x(WL, WR, gamma, c_h));
        }
    }

    for (int j = grid.j_begin(); j <= grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const MHDPrimitive Wmm = W.get(i, j - 2);
            const MHDPrimitive Wm = W.get(i, j - 1);
            const MHDPrimitive W0 = W.get(i, j);
            const MHDPrimitive Wp = W.get(i, j + 1);

            MHDPrimitive WB{};
            MHDPrimitive WT{};
            muscl_interface_y(Wmm, Wm, W0, Wp, limiter, WB, WT);
            store_flux_reference(fy, i, j, hll_flux_y(WB, WT, gamma, c_h));
        }
    }

    // Ideal path: skip all non-ideal work.
    if (nonideal.any()) {
        compute_current(W, grid, Jx.data(), Jy.data(), Jz.data());
        add_nonideal_face_fluxes(W, fx, fy, grid, Jx.data(), Jy.data(), Jz.data(), nonideal);
    }

    const double h = std::min(grid.dx, grid.dy);
    const double psi_damp = (c_h > 0.0 && glm_alpha > 0.0) ? (c_h * glm_alpha / h) : 0.0;

    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const MHDFlux F_L = load_flux_reference(fx, i, j);
            const MHDFlux F_R = load_flux_reference(fx, i + 1, j);
            const MHDFlux G_B = load_flux_reference(fy, i, j);
            const MHDFlux G_T = load_flux_reference(fy, i, j + 1);

            MHDState L{};
            L.rho = -(F_R.rho - F_L.rho) * inv_dx - (G_T.rho - G_B.rho) * inv_dy;
            L.mx = -(F_R.mx - F_L.mx) * inv_dx - (G_T.mx - G_B.mx) * inv_dy;
            L.my = -(F_R.my - F_L.my) * inv_dx - (G_T.my - G_B.my) * inv_dy;
            L.mz = -(F_R.mz - F_L.mz) * inv_dx - (G_T.mz - G_B.mz) * inv_dy;
            L.energy = -(F_R.energy - F_L.energy) * inv_dx - (G_T.energy - G_B.energy) * inv_dy;
            L.bx = -(F_R.bx - F_L.bx) * inv_dx - (G_T.bx - G_B.bx) * inv_dy;
            L.by = -(F_R.by - F_L.by) * inv_dx - (G_T.by - G_B.by) * inv_dy;
            L.bz = -(F_R.bz - F_L.bz) * inv_dx - (G_T.bz - G_B.bz) * inv_dy;
            L.psi = -(F_R.psi - F_L.psi) * inv_dx - (G_T.psi - G_B.psi) * inv_dy;
            L.psi -= psi_damp * U.psi[U.index(i, j)];

            rhs.set_state(i, j, L);
        }
    }

    if (mask != nullptr) {
        zero_rhs_on_solids(rhs, grid, *mask);
    }
}

inline void compute_rhs_reference(const StateField& U, StateField& rhs, const Grid2D& grid,
                                  double gamma, double c_h, double glm_alpha, SlopeLimiter limiter,
                                  const NonIdealConfig& nonideal = NonIdealConfig::ideal(),
                                  const FluidMask* mask = nullptr)
{
    RHSReferenceWorkspace work(U.nx, U.ny);
    compute_rhs_reference(U, rhs, grid, work, gamma, c_h, glm_alpha, limiter, nonideal, mask);
}
