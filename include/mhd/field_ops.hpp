#pragma once

#include "mhd/compat.hpp"
#include "mhd/parallel.hpp"
#include "state/state_field.hpp"

#include <array>
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

namespace detail {

inline void xpay_span(double* MHD_RESTRICT o, const double* MHD_RESTRICT x, double a,
                      const double* MHD_RESTRICT y, std::size_t lo, std::size_t hi)
{
    for (std::size_t i = lo; i < hi; ++i) {
        o[i] = x[i] + a * y[i];
    }
}

inline void ssp_rk2_combine_span(double* MHD_RESTRICT u, const double* MHD_RESTRICT s,
                                 const double* MHD_RESTRICT r, double half_dt, std::size_t lo,
                                 std::size_t hi)
{
    const double half = 0.5;
    for (std::size_t i = lo; i < hi; ++i) {
        u[i] = half * (u[i] + s[i]) + half_dt * r[i];
    }
}

inline std::array<double*, 9> components(StateField& U)
{
    return {U.rho.data(), U.mx.data(), U.my.data(), U.mz.data(), U.energy.data(),
            U.bx.data(),  U.by.data(), U.bz.data(), U.psi.data()};
}

inline std::array<const double*, 9> components(const StateField& U)
{
    return {U.rho.data(), U.mx.data(), U.my.data(), U.mz.data(), U.energy.data(),
            U.bx.data(),  U.by.data(), U.bz.data(), U.psi.data()};
}

} // namespace detail

// out = x + a * y (all 9 components), multithreaded over cell blocks.
inline void field_xpay(StateField& out, const StateField& x, double a, const StateField& y)
{
    ensure_same_shape(out, x);
    ensure_same_shape(out, y);

    const auto o = detail::components(out);
    const auto xs = detail::components(x);
    const auto ys = detail::components(y);
    mhd::parallel_range(field_size(out), 8, [&](std::size_t lo, std::size_t hi, int /*tid*/) {
        for (int c = 0; c < 9; ++c) {
            detail::xpay_span(o[c], xs[c], a, ys[c], lo, hi);
        }
    });
}

// U = 1/2 (U + U*) + dt/2 * L(U*) (second SSP-RK2 stage), multithreaded over cell blocks.
inline void field_ssp_rk2_combine(StateField& U, const StateField& U_star, const StateField& rhs,
                                  double dt)
{
    ensure_same_shape(U, U_star);
    ensure_same_shape(U, rhs);

    const double half_dt = 0.5 * dt;
    const auto u = detail::components(U);
    const auto s = detail::components(U_star);
    const auto r = detail::components(rhs);
    mhd::parallel_range(field_size(U), 8, [&](std::size_t lo, std::size_t hi, int /*tid*/) {
        for (int c = 0; c < 9; ++c) {
            detail::ssp_rk2_combine_span(u[c], s[c], r[c], half_dt, lo, hi);
        }
    });
}
