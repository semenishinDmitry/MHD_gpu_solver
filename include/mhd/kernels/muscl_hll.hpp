#pragma once

// Branchless MUSCL + HLL face-flux kernel for GLM-MHD, generic over the number type T.
//
// The same source is instantiated for
//   * double                — CPU scalar path / reference for the vector path
//   * mhd::simd::vdouble    — CPU SIMD path (NEON, SSE2, AVX, AVX-512)
//   * double, float         — CUDA device code (MHD_HD)
// The Metal shaders in src/metal/mhd_kernels.metal mirror this file line by line.
//
// Work is done in a rotated frame: "n" is the face-normal direction and "t" the in-plane
// tangential direction. For x-faces (n, t) = (x, y), for y-faces (n, t) = (y, x).
// This is algebraically identical to physical_flux_x / physical_flux_y in mhd_physics.hpp.

#include "mhd/compat.hpp"
#include "mhd/mhd_types.hpp"
#include "mhd/simd.hpp"

#include <cmath>

namespace mhd::kernels {

template <class T>
struct PrimN {
    T rho, vn, vt, vz, p, bn, bt, bz, psi;
};

template <class T>
struct FluxN {
    T rho, mn, mt, mz, e, bn, bt, bz, psi;
};

template <class T>
MHD_HD MHD_INLINE T minmod(T a, T b)
{
    const T zero(0.0);
    return simd::select(a * b <= zero, zero, simd::select(simd::vabs(a) < simd::vabs(b), a, b));
}

// Monotonized-central limiter on consecutive differences a = W_i - W_{i-1}, b = W_{i+1} - W_i.
template <class T>
MHD_HD MHD_INLINE T mc_limiter(T a, T b)
{
    const T half(0.5);
    const T two(2.0);
    return minmod(half * (a + b), minmod(two * a, two * b));
}

template <SlopeLimiter L, class T>
MHD_HD MHD_INLINE T limit_slope(T a, T b)
{
    if constexpr (L == SlopeLimiter::Minmod) {
        return minmod(a, b);
    } else {
        return mc_limiter(a, b);
    }
}

// Face i-1/2 from the stencil (i-2, i-1, i, i+1): left state from cell i-1, right from cell i.
template <SlopeLimiter L, class T>
MHD_HD MHD_INLINE void reconstruct(T wmm, T wm, T w0, T wp, T& left, T& right)
{
    const T half(0.5);
    const T dm = limit_slope<L>(wm - wmm, w0 - wm);
    const T d0 = limit_slope<L>(w0 - wm, wp - w0);
    left = wm + half * dm;
    right = w0 - half * d0;
}

// Fast magnetosonic speed along n.
template <class T>
MHD_HD MHD_INLINE T fast_speed_n(const PrimN<T>& W, T gamma)
{
    const T zero(0.0);
    const T half(0.5);
    const T four(4.0);
    const T inv_rho = T(1.0) / W.rho;
    const T cs2 = gamma * W.p * inv_rho;
    const T bn2 = W.bn * W.bn;
    const T b2 = bn2 + W.bt * W.bt + W.bz * W.bz;
    const T va2 = b2 * inv_rho;
    const T van2 = bn2 * inv_rho;
    const T sum = cs2 + va2;
    const T disc = simd::vmax(sum * sum - four * cs2 * van2, zero);
    const T cf2 = half * (sum + simd::vsqrt(disc));
    return simd::vsqrt(simd::vmax(cf2, zero));
}

// Conserved state and physical GLM-MHD flux along n.
template <class T>
MHD_HD MHD_INLINE void conserved_and_flux(const PrimN<T>& W, T inv_gm1, T ch2, FluxN<T>& U,
                                          FluxN<T>& F)
{
    const T half(0.5);
    const T mn = W.rho * W.vn;
    const T mt = W.rho * W.vt;
    const T mz = W.rho * W.vz;
    const T v2 = W.vn * W.vn + W.vt * W.vt + W.vz * W.vz;
    const T b2 = W.bn * W.bn + W.bt * W.bt + W.bz * W.bz;
    const T pmag = half * b2;
    const T energy = W.p * inv_gm1 + half * W.rho * v2 + pmag;
    const T pstar = W.p + pmag;
    const T vb = W.vn * W.bn + W.vt * W.bt + W.vz * W.bz;

    U.rho = W.rho;
    U.mn = mn;
    U.mt = mt;
    U.mz = mz;
    U.e = energy;
    U.bn = W.bn;
    U.bt = W.bt;
    U.bz = W.bz;
    U.psi = W.psi;

    F.rho = mn;
    F.mn = mn * W.vn + pstar - W.bn * W.bn;
    F.mt = mn * W.vt - W.bn * W.bt;
    F.mz = mn * W.vz - W.bn * W.bz;
    F.e = (energy + pstar) * W.vn - W.bn * vb + W.psi * W.bn;
    F.bn = W.psi;
    F.bt = W.vn * W.bt - W.vt * W.bn;
    F.bz = W.vn * W.bz - W.vz * W.bn;
    F.psi = ch2 * W.bn;
}

// MUSCL reconstruction + HLL flux with Dedner GLM wave-speed bounds (-c_h, +c_h).
// Falls back to first order on the face when the reconstructed state is unphysical.
template <SlopeLimiter L, class T>
MHD_HD MHD_INLINE FluxN<T> muscl_hll_face(const PrimN<T>& Wmm, const PrimN<T>& Wm,
                                          const PrimN<T>& W0, const PrimN<T>& Wp, T gamma,
                                          T inv_gm1, T c_h)
{
    PrimN<T> WL;
    PrimN<T> WR;
    reconstruct<L>(Wmm.rho, Wm.rho, W0.rho, Wp.rho, WL.rho, WR.rho);
    reconstruct<L>(Wmm.vn, Wm.vn, W0.vn, Wp.vn, WL.vn, WR.vn);
    reconstruct<L>(Wmm.vt, Wm.vt, W0.vt, Wp.vt, WL.vt, WR.vt);
    reconstruct<L>(Wmm.vz, Wm.vz, W0.vz, Wp.vz, WL.vz, WR.vz);
    reconstruct<L>(Wmm.p, Wm.p, W0.p, Wp.p, WL.p, WR.p);
    reconstruct<L>(Wmm.bn, Wm.bn, W0.bn, Wp.bn, WL.bn, WR.bn);
    reconstruct<L>(Wmm.bt, Wm.bt, W0.bt, Wp.bt, WL.bt, WR.bt);
    reconstruct<L>(Wmm.bz, Wm.bz, W0.bz, Wp.bz, WL.bz, WR.bz);
    reconstruct<L>(Wmm.psi, Wm.psi, W0.psi, Wp.psi, WL.psi, WR.psi);

    // Positivity + finiteness (x < inf is false for NaN and inf) of both face states.
    const T zero(0.0);
    const T inf(HUGE_VAL);
    const auto ok = simd::land(
        simd::land(simd::land(WL.rho > zero, WL.p > zero), simd::land(WL.rho < inf, WL.p < inf)),
        simd::land(simd::land(WR.rho > zero, WR.p > zero), simd::land(WR.rho < inf, WR.p < inf)));
    WL.rho = simd::select(ok, WL.rho, Wm.rho);
    WL.vn = simd::select(ok, WL.vn, Wm.vn);
    WL.vt = simd::select(ok, WL.vt, Wm.vt);
    WL.vz = simd::select(ok, WL.vz, Wm.vz);
    WL.p = simd::select(ok, WL.p, Wm.p);
    WL.bn = simd::select(ok, WL.bn, Wm.bn);
    WL.bt = simd::select(ok, WL.bt, Wm.bt);
    WL.bz = simd::select(ok, WL.bz, Wm.bz);
    WL.psi = simd::select(ok, WL.psi, Wm.psi);
    WR.rho = simd::select(ok, WR.rho, W0.rho);
    WR.vn = simd::select(ok, WR.vn, W0.vn);
    WR.vt = simd::select(ok, WR.vt, W0.vt);
    WR.vz = simd::select(ok, WR.vz, W0.vz);
    WR.p = simd::select(ok, WR.p, W0.p);
    WR.bn = simd::select(ok, WR.bn, W0.bn);
    WR.bt = simd::select(ok, WR.bt, W0.bt);
    WR.bz = simd::select(ok, WR.bz, W0.bz);
    WR.psi = simd::select(ok, WR.psi, W0.psi);

    const T cfL = fast_speed_n(WL, gamma);
    const T cfR = fast_speed_n(WR, gamma);
    const T sL = simd::vmin(simd::vmin(WL.vn - cfL, WR.vn - cfR), -c_h);
    const T sR = simd::vmax(simd::vmax(WL.vn + cfL, WR.vn + cfR), c_h);

    const T ch2 = c_h * c_h;
    FluxN<T> UL;
    FluxN<T> UR;
    FluxN<T> FL;
    FluxN<T> FR;
    conserved_and_flux(WL, inv_gm1, ch2, UL, FL);
    conserved_and_flux(WR, inv_gm1, ch2, UR, FR);

    const T denom = sR - sL;
    const T inv_denom = T(1.0) / denom;
    const T sLsR = sL * sR;
    const T half(0.5);
    const auto use_left = sL >= zero;
    const auto use_right = sR <= zero;
    const auto degenerate = simd::vabs(denom) < T(1.0e-14);

    // F_hll = (sR F_L - sL F_R + sL sR (U_R - U_L)) / (sR - sL), with upwind/degenerate cases.
    auto combine = [&](T fl, T fr, T ul, T ur) {
        const T hll = (sR * fl - sL * fr + sLsR * (ur - ul)) * inv_denom;
        const T mid = simd::select(degenerate, half * (fl + fr), hll);
        return simd::select(use_left, fl, simd::select(use_right, fr, mid));
    };

    FluxN<T> F;
    F.rho = combine(FL.rho, FR.rho, UL.rho, UR.rho);
    F.mn = combine(FL.mn, FR.mn, UL.mn, UR.mn);
    F.mt = combine(FL.mt, FR.mt, UL.mt, UR.mt);
    F.mz = combine(FL.mz, FR.mz, UL.mz, UR.mz);
    F.e = combine(FL.e, FR.e, UL.e, UR.e);
    F.bn = combine(FL.bn, FR.bn, UL.bn, UR.bn);
    F.bt = combine(FL.bt, FR.bt, UL.bt, UR.bt);
    F.bz = combine(FL.bz, FR.bz, UL.bz, UR.bz);
    F.psi = combine(FL.psi, FR.psi, UL.psi, UR.psi);
    return F;
}

// max(|v_x| + c_f,x, |v_y| + c_f,y) for one cell given conserved variables.
template <class T>
MHD_HD MHD_INLINE T cell_max_signal_speed(T rho, T mx, T my, T mz, T energy, T bx, T by, T bz,
                                          T gamma)
{
    const T half(0.5);
    const T inv_rho = T(1.0) / rho;
    PrimN<T> W;
    W.rho = rho;
    W.vn = mx * inv_rho;
    W.vt = my * inv_rho;
    W.vz = mz * inv_rho;
    const T ekin = half * (mx * mx + my * my + mz * mz) * inv_rho;
    const T emag = half * (bx * bx + by * by + bz * bz);
    W.p = (gamma - T(1.0)) * (energy - ekin - emag);
    W.bn = bx;
    W.bt = by;
    W.bz = bz;
    W.psi = T(0.0);
    const T sx = simd::vabs(W.vn) + fast_speed_n(W, gamma);

    PrimN<T> Wy = W;
    Wy.vn = W.vt;
    Wy.vt = W.vn;
    Wy.bn = by;
    Wy.bt = bx;
    const T sy = simd::vabs(Wy.vn) + fast_speed_n(Wy, gamma);
    return simd::vmax(sx, sy);
}

} // namespace mhd::kernels
