#pragma once

#include "grid/grid.hpp"
#include "mhd/compat.hpp"
#include "physics_config/mhd_config.hpp"
#include "state/state_field.hpp"

#include <cmath>

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline Vec3 cross(const Vec3& a, const Vec3& b)
{
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

// Cell-centered current J = ∇ × B (2.5D: ∂/∂z = 0).
//   Jx =  ∂Bz/∂y
//   Jy = -∂Bz/∂x
//   Jz =  ∂By/∂x - ∂Bx/∂y
inline void compute_current(const PrimitiveField& W,
                            const Grid2D& grid,
                            double* MHD_RESTRICT Jx,
                            double* MHD_RESTRICT Jy,
                            double* MHD_RESTRICT Jz)
{
    const double inv_2dx = 0.5 / grid.dx;
    const double inv_2dy = 0.5 / grid.dy;
    const int nx = W.nx;

    const double* MHD_RESTRICT bx = W.bx.data();
    const double* MHD_RESTRICT by = W.by.data();
    const double* MHD_RESTRICT bz = W.bz.data();

    // Need one layer of ghosts around interior for face averaging; fill all cells
    // with accessible centered neighbors (ng >= 2 guarantees this for interior faces).
    for (int j = 1; j < grid.get_size_y() - 1; ++j) {
        const int row = j * nx;
        for (int i = 1; i < grid.get_size_x() - 1; ++i) {
            const int idx = row + i;
            const double dbz_dy = (bz[idx + nx] - bz[idx - nx]) * inv_2dy;
            const double dbz_dx = (bz[idx + 1] - bz[idx - 1]) * inv_2dx;
            const double dby_dx = (by[idx + 1] - by[idx - 1]) * inv_2dx;
            const double dbx_dy = (bx[idx + nx] - bx[idx - nx]) * inv_2dy;

            Jx[idx] = dbz_dy;
            Jy[idx] = -dbz_dx;
            Jz[idx] = dby_dx - dbx_dy;
        }
    }
}

// Non-ideal electric field at a point from J and B.
inline Vec3 nonideal_electric_field(const Vec3& J, const Vec3& B, const NonIdealConfig& cfg)
{
    Vec3 E{};

    if (cfg.ohmic) {
        E.x += cfg.eta_ohm * J.x;
        E.y += cfg.eta_ohm * J.y;
        E.z += cfg.eta_ohm * J.z;
    }

    const bool need_B = cfg.hall || cfg.ambipolar;
    if (!need_B) {
        return E;
    }

    constexpr double eps = 1.0e-14;
    const double B2 = B.x * B.x + B.y * B.y + B.z * B.z;
    if (B2 < eps) {
        return E;
    }

    const Vec3 JxB = cross(J, B);

    if (cfg.hall) {
        const double invB = 1.0 / std::sqrt(B2);
        E.x += cfg.eta_hall * JxB.x * invB;
        E.y += cfg.eta_hall * JxB.y * invB;
        E.z += cfg.eta_hall * JxB.z * invB;
    }

    if (cfg.ambipolar) {
        const Vec3 JxBxB = cross(JxB, B);
        const double invB2 = 1.0 / B2;
        E.x += cfg.eta_ambipolar * JxBxB.x * invB2;
        E.y += cfg.eta_ambipolar * JxBxB.y * invB2;
        E.z += cfg.eta_ambipolar * JxBxB.z * invB2;
    }

    return E;
}

inline Vec3 avg3(double ax, double ay, double az, double bx, double by, double bz)
{
    return Vec3{0.5 * (ax + bx), 0.5 * (ay + by), 0.5 * (az + bz)};
}

// Add -∇·F_NI to magnetic and energy RHS via face electric fields.
// Hyperbolic fluxes are already stored in fx/fy; we accumulate resistive fluxes there.
// Induction: Faraday ∂t B = -∇×E (face fluxes of E).
// Energy: Poynting contribution written as B×E_ni (= -E_ni×B), consistent with
// ∂t E + ∇·(... + B×E_ni) = 0 in the model equations.
inline void add_nonideal_face_fluxes(const PrimitiveField& W,
                                     StateField& fx,
                                     StateField& fy,
                                     const Grid2D& grid,
                                     const double* MHD_RESTRICT Jx,
                                     const double* MHD_RESTRICT Jy,
                                     const double* MHD_RESTRICT Jz,
                                     const NonIdealConfig& cfg)
{
    const int nx = W.nx;

    // X-faces i-1/2, j
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i <= grid.i_end(); ++i) {
            const int il = (j * nx) + (i - 1);
            const int ir = (j * nx) + i;

            const Vec3 J = avg3(Jx[il], Jy[il], Jz[il], Jx[ir], Jy[ir], Jz[ir]);
            const Vec3 B = avg3(W.bx[il], W.by[il], W.bz[il], W.bx[ir], W.by[ir], W.bz[ir]);
            const Vec3 E = nonideal_electric_field(J, B, cfg);

            const int fidx = fx.index(i, j);
            // F_by += -Ez, F_bz += +Ey, F_E += (B × E)_x = By Ez - Bz Ey
            fx.by[fidx] += -E.z;
            fx.bz[fidx] += E.y;
            fx.energy[fidx] += B.y * E.z - B.z * E.y;
        }
    }

    // Y-faces i, j-1/2
    for (int j = grid.j_begin(); j <= grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            const int ib = ((j - 1) * nx) + i;
            const int it = (j * nx) + i;

            const Vec3 J = avg3(Jx[ib], Jy[ib], Jz[ib], Jx[it], Jy[it], Jz[it]);
            const Vec3 B = avg3(W.bx[ib], W.by[ib], W.bz[ib], W.bx[it], W.by[it], W.bz[it]);
            const Vec3 E = nonideal_electric_field(J, B, cfg);

            const int fidx = fy.index(i, j);
            // F_bx += +Ez, F_bz += -Ex, F_E += (B × E)_y = Bz Ex - Bx Ez
            fy.bx[fidx] += E.z;
            fy.bz[fidx] += -E.x;
            fy.energy[fidx] += B.z * E.x - B.x * E.z;
        }
    }
}
