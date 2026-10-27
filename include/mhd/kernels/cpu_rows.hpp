#pragma once

// CPU row kernels: SoA loads/stores around the generic kernels in muscl_hll.hpp.
// Every row is processed as [vector body | scalar tail]; the split depends only on the row
// length, never on the thread partition, so results are independent of the thread count.

#include "mhd/compat.hpp"
#include "mhd/kernels/muscl_hll.hpp"
#include "mhd/mhd_types.hpp"
#include "mhd/simd.hpp"
#include "state/state_field.hpp"

#include <cstddef>

namespace mhd::kernels {

struct PrimPtrs {
    const double* MHD_RESTRICT rho;
    const double* MHD_RESTRICT vx;
    const double* MHD_RESTRICT vy;
    const double* MHD_RESTRICT vz;
    const double* MHD_RESTRICT p;
    const double* MHD_RESTRICT bx;
    const double* MHD_RESTRICT by;
    const double* MHD_RESTRICT bz;
    const double* MHD_RESTRICT psi;

    explicit PrimPtrs(const PrimitiveField& W)
        : rho(W.rho.data()), vx(W.vx.data()), vy(W.vy.data()), vz(W.vz.data()),
          p(W.pressure.data()), bx(W.bx.data()), by(W.by.data()), bz(W.bz.data()), psi(W.psi.data())
    {
    }
};

// Nine flux rows in physical component order (rho, mx, my, mz, E, bx, by, bz, psi).
struct FluxRow {
    double* c[9];
};

template <class T, bool YDir>
MHD_INLINE PrimN<T> load_prim(const PrimPtrs& P, std::ptrdiff_t idx)
{
    PrimN<T> W;
    W.rho = simd::load<T>(P.rho + idx);
    W.vz = simd::load<T>(P.vz + idx);
    W.p = simd::load<T>(P.p + idx);
    W.bz = simd::load<T>(P.bz + idx);
    W.psi = simd::load<T>(P.psi + idx);
    if constexpr (!YDir) {
        W.vn = simd::load<T>(P.vx + idx);
        W.vt = simd::load<T>(P.vy + idx);
        W.bn = simd::load<T>(P.bx + idx);
        W.bt = simd::load<T>(P.by + idx);
    } else {
        W.vn = simd::load<T>(P.vy + idx);
        W.vt = simd::load<T>(P.vx + idx);
        W.bn = simd::load<T>(P.by + idx);
        W.bt = simd::load<T>(P.bx + idx);
    }
    return W;
}

template <class T, bool YDir>
MHD_INLINE void store_flux(const FluxRow& out, int k, const FluxN<T>& F)
{
    simd::store(out.c[0] + k, F.rho);
    simd::store(out.c[3] + k, F.mz);
    simd::store(out.c[4] + k, F.e);
    simd::store(out.c[7] + k, F.bz);
    simd::store(out.c[8] + k, F.psi);
    if constexpr (!YDir) {
        simd::store(out.c[1] + k, F.mn);
        simd::store(out.c[2] + k, F.mt);
        simd::store(out.c[5] + k, F.bn);
        simd::store(out.c[6] + k, F.bt);
    } else {
        simd::store(out.c[1] + k, F.mt);
        simd::store(out.c[2] + k, F.mn);
        simd::store(out.c[5] + k, F.bt);
        simd::store(out.c[6] + k, F.bn);
    }
}

template <SlopeLimiter L, bool YDir, class T>
MHD_INLINE void face_block(const PrimPtrs& P, std::ptrdiff_t idx, std::ptrdiff_t stride,
                           double gamma, double inv_gm1, double c_h, const FluxRow& out, int k)
{
    const PrimN<T> Wmm = load_prim<T, YDir>(P, idx - 2 * stride);
    const PrimN<T> Wm = load_prim<T, YDir>(P, idx - stride);
    const PrimN<T> W0 = load_prim<T, YDir>(P, idx);
    const PrimN<T> Wp = load_prim<T, YDir>(P, idx + stride);
    const FluxN<T> F = muscl_hll_face<L>(Wmm, Wm, W0, Wp, T(gamma), T(inv_gm1), T(c_h));
    store_flux<T, YDir>(out, k, F);
}

// Fluxes at `count` consecutive faces. Face k sits between cells (base + k - stride) and
// (base + k); stride = 1 for x-faces, stride = nx_tot for y-faces.
template <SlopeLimiter L, bool YDir>
void face_row(const PrimPtrs& P, std::ptrdiff_t base, std::ptrdiff_t stride, int count,
              double gamma, double c_h, const FluxRow& out, bool use_simd)
{
    const double inv_gm1 = 1.0 / (gamma - 1.0);
    int k = 0;
    if (use_simd) {
        constexpr int w = simd::kWidth;
        for (; k + w <= count; k += w) {
            face_block<L, YDir, simd::vdouble>(P, base + k, stride, gamma, inv_gm1, c_h, out, k);
        }
    }
    for (; k < count; ++k) {
        face_block<L, YDir, double>(P, base + k, stride, gamma, inv_gm1, c_h, out, k);
    }
}

template <bool YDir>
void face_row(SlopeLimiter limiter, const PrimPtrs& P, std::ptrdiff_t base, std::ptrdiff_t stride,
              int count, double gamma, double c_h, const FluxRow& out, bool use_simd)
{
    switch (limiter) {
    case SlopeLimiter::Minmod:
        face_row<SlopeLimiter::Minmod, YDir>(P, base, stride, count, gamma, c_h, out, use_simd);
        return;
    case SlopeLimiter::MC:
        face_row<SlopeLimiter::MC, YDir>(P, base, stride, count, gamma, c_h, out, use_simd);
        return;
    }
}

// Conserved -> primitive for cells [lo, hi).
template <class T>
MHD_INLINE void primitive_block(const StateField& U, PrimitiveField& W, std::size_t idx,
                                double gamma)
{
    const T half(0.5);
    const T rho = simd::load<T>(U.rho.data() + idx);
    const T mx = simd::load<T>(U.mx.data() + idx);
    const T my = simd::load<T>(U.my.data() + idx);
    const T mz = simd::load<T>(U.mz.data() + idx);
    const T e = simd::load<T>(U.energy.data() + idx);
    const T bx = simd::load<T>(U.bx.data() + idx);
    const T by = simd::load<T>(U.by.data() + idx);
    const T bz = simd::load<T>(U.bz.data() + idx);
    const T inv_rho = T(1.0) / rho;
    const T ekin = half * (mx * mx + my * my + mz * mz) * inv_rho;
    const T emag = half * (bx * bx + by * by + bz * bz);
    simd::store(W.rho.data() + idx, rho);
    simd::store(W.vx.data() + idx, mx * inv_rho);
    simd::store(W.vy.data() + idx, my * inv_rho);
    simd::store(W.vz.data() + idx, mz * inv_rho);
    simd::store(W.pressure.data() + idx, T(gamma - 1.0) * (e - ekin - emag));
    simd::store(W.bx.data() + idx, bx);
    simd::store(W.by.data() + idx, by);
    simd::store(W.bz.data() + idx, bz);
    simd::store(W.psi.data() + idx, simd::load<T>(U.psi.data() + idx));
}

inline void primitive_range(const StateField& U, PrimitiveField& W, std::size_t lo, std::size_t hi,
                            double gamma, bool use_simd)
{
    std::size_t idx = lo;
    if (use_simd) {
        constexpr std::size_t w = static_cast<std::size_t>(simd::kWidth);
        for (; idx + w <= hi; idx += w) {
            primitive_block<simd::vdouble>(U, W, idx, gamma);
        }
    }
    for (; idx < hi; ++idx) {
        primitive_block<double>(U, W, idx, gamma);
    }
}

// max signal speed over cells [base, base + count) of one row.
inline double max_signal_speed_row(const StateField& U, std::ptrdiff_t base, int count,
                                   double gamma, bool use_simd)
{
    auto block = [&](auto tag, std::ptrdiff_t idx) {
        using T = decltype(tag);
        return cell_max_signal_speed<T>(
            simd::load<T>(U.rho.data() + idx), simd::load<T>(U.mx.data() + idx),
            simd::load<T>(U.my.data() + idx), simd::load<T>(U.mz.data() + idx),
            simd::load<T>(U.energy.data() + idx), simd::load<T>(U.bx.data() + idx),
            simd::load<T>(U.by.data() + idx), simd::load<T>(U.bz.data() + idx), T(gamma));
    };

    double vmax_scalar = 0.0;
    int k = 0;
    if (use_simd && simd::kWidth > 1) {
        constexpr int w = simd::kWidth;
        simd::vdouble acc(0.0);
        for (; k + w <= count; k += w) {
            acc = simd::vmax(acc, block(simd::vdouble{}, base + k));
        }
        vmax_scalar = simd::hmax(acc);
    }
    for (; k < count; ++k) {
        vmax_scalar = simd::vmax(vmax_scalar, block(double{}, base + k));
    }
    return vmax_scalar;
}

} // namespace mhd::kernels
