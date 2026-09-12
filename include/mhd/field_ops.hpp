#pragma once

#include "state/state_field.hpp"

#include <cstddef>
#include <cstring>
#include <stdexcept>

// Shape checks are debug-only: hot RK kernels must stay allocation- and branch-light.
inline void ensure_same_shape(const StateField& a, const StateField& b)
{
#ifndef NDEBUG
    if (a.nx != b.nx || a.ny != b.ny) {
        throw std::invalid_argument("StateField shape mismatch");
    }
#else
    (void)a;
    (void)b;
#endif
}

inline std::size_t field_size(const StateField& U)
{
    return U.rho.size();
}

inline void field_copy(StateField& dst, const StateField& src)
{
    ensure_same_shape(dst, src);
    const std::size_t n = field_size(src);
    std::memcpy(dst.rho.data(), src.rho.data(), n * sizeof(double));
    std::memcpy(dst.mx.data(), src.mx.data(), n * sizeof(double));
    std::memcpy(dst.my.data(), src.my.data(), n * sizeof(double));
    std::memcpy(dst.mz.data(), src.mz.data(), n * sizeof(double));
    std::memcpy(dst.energy.data(), src.energy.data(), n * sizeof(double));
    std::memcpy(dst.bx.data(), src.bx.data(), n * sizeof(double));
    std::memcpy(dst.by.data(), src.by.data(), n * sizeof(double));
    std::memcpy(dst.bz.data(), src.bz.data(), n * sizeof(double));
    std::memcpy(dst.psi.data(), src.psi.data(), n * sizeof(double));
}

inline void field_xpay(StateField& out, const StateField& x, double a, const StateField& y)
{
    ensure_same_shape(out, x);
    ensure_same_shape(out, y);

    const std::size_t n = field_size(out);

    double* __restrict__ o_rho = out.rho.data();
    double* __restrict__ o_mx = out.mx.data();
    double* __restrict__ o_my = out.my.data();
    double* __restrict__ o_mz = out.mz.data();
    double* __restrict__ o_e = out.energy.data();
    double* __restrict__ o_bx = out.bx.data();
    double* __restrict__ o_by = out.by.data();
    double* __restrict__ o_bz = out.bz.data();
    double* __restrict__ o_psi = out.psi.data();

    const double* __restrict__ x_rho = x.rho.data();
    const double* __restrict__ x_mx = x.mx.data();
    const double* __restrict__ x_my = x.my.data();
    const double* __restrict__ x_mz = x.mz.data();
    const double* __restrict__ x_e = x.energy.data();
    const double* __restrict__ x_bx = x.bx.data();
    const double* __restrict__ x_by = x.by.data();
    const double* __restrict__ x_bz = x.bz.data();
    const double* __restrict__ x_psi = x.psi.data();

    const double* __restrict__ y_rho = y.rho.data();
    const double* __restrict__ y_mx = y.mx.data();
    const double* __restrict__ y_my = y.my.data();
    const double* __restrict__ y_mz = y.mz.data();
    const double* __restrict__ y_e = y.energy.data();
    const double* __restrict__ y_bx = y.bx.data();
    const double* __restrict__ y_by = y.by.data();
    const double* __restrict__ y_bz = y.bz.data();
    const double* __restrict__ y_psi = y.psi.data();

    for (std::size_t i = 0; i < n; ++i) {
        o_rho[i] = x_rho[i] + a * y_rho[i];
        o_mx[i] = x_mx[i] + a * y_mx[i];
        o_my[i] = x_my[i] + a * y_my[i];
        o_mz[i] = x_mz[i] + a * y_mz[i];
        o_e[i] = x_e[i] + a * y_e[i];
        o_bx[i] = x_bx[i] + a * y_bx[i];
        o_by[i] = x_by[i] + a * y_by[i];
        o_bz[i] = x_bz[i] + a * y_bz[i];
        o_psi[i] = x_psi[i] + a * y_psi[i];
    }
}

inline void field_ssp_rk2_combine(StateField& U,
                                 const StateField& U_star,
                                 const StateField& rhs,
                                 double dt)
{
    ensure_same_shape(U, U_star);
    ensure_same_shape(U, rhs);

    const std::size_t n = field_size(U);
    const double half = 0.5;
    const double half_dt = 0.5 * dt;

    double* __restrict__ u_rho = U.rho.data();
    double* __restrict__ u_mx = U.mx.data();
    double* __restrict__ u_my = U.my.data();
    double* __restrict__ u_mz = U.mz.data();
    double* __restrict__ u_e = U.energy.data();
    double* __restrict__ u_bx = U.bx.data();
    double* __restrict__ u_by = U.by.data();
    double* __restrict__ u_bz = U.bz.data();
    double* __restrict__ u_psi = U.psi.data();

    const double* __restrict__ s_rho = U_star.rho.data();
    const double* __restrict__ s_mx = U_star.mx.data();
    const double* __restrict__ s_my = U_star.my.data();
    const double* __restrict__ s_mz = U_star.mz.data();
    const double* __restrict__ s_e = U_star.energy.data();
    const double* __restrict__ s_bx = U_star.bx.data();
    const double* __restrict__ s_by = U_star.by.data();
    const double* __restrict__ s_bz = U_star.bz.data();
    const double* __restrict__ s_psi = U_star.psi.data();

    const double* __restrict__ r_rho = rhs.rho.data();
    const double* __restrict__ r_mx = rhs.mx.data();
    const double* __restrict__ r_my = rhs.my.data();
    const double* __restrict__ r_mz = rhs.mz.data();
    const double* __restrict__ r_e = rhs.energy.data();
    const double* __restrict__ r_bx = rhs.bx.data();
    const double* __restrict__ r_by = rhs.by.data();
    const double* __restrict__ r_bz = rhs.bz.data();
    const double* __restrict__ r_psi = rhs.psi.data();

    for (std::size_t i = 0; i < n; ++i) {
        u_rho[i] = half * (u_rho[i] + s_rho[i]) + half_dt * r_rho[i];
        u_mx[i] = half * (u_mx[i] + s_mx[i]) + half_dt * r_mx[i];
        u_my[i] = half * (u_my[i] + s_my[i]) + half_dt * r_my[i];
        u_mz[i] = half * (u_mz[i] + s_mz[i]) + half_dt * r_mz[i];
        u_e[i] = half * (u_e[i] + s_e[i]) + half_dt * r_e[i];
        u_bx[i] = half * (u_bx[i] + s_bx[i]) + half_dt * r_bx[i];
        u_by[i] = half * (u_by[i] + s_by[i]) + half_dt * r_by[i];
        u_bz[i] = half * (u_bz[i] + s_bz[i]) + half_dt * r_bz[i];
        u_psi[i] = half * (u_psi[i] + s_psi[i]) + half_dt * r_psi[i];
    }
}
