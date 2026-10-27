#pragma once

// Minimal portable SIMD layer for the hot MHD kernels.
//
// Kernels in include/mhd/kernels/ are templates over a number type T and only use the
// operations below (arithmetic operators, comparisons, vmin/vmax/vsqrt/vabs, select,
// land/lor, load/store). T can be:
//   * double / float          — scalar CPU path, CUDA device code (MHD_HD)
//   * mhd::simd::vdouble      — CPU vector path, backend chosen at compile time:
//       AVX-512F (8 lanes) > AVX/AVX2 (4) > SSE2 (2) on x86-64, NEON (2) on AArch64,
//       scalar fallback (1) elsewhere.
//
// The backend follows the compiler's target flags (-march=native, /arch:AVX2, ...),
// see MHD_NATIVE_ARCH / MHD_SIMD_ARCH in CMake and docs/performance.md.

#include "mhd/compat.hpp"

#include <cmath>

#if !defined(__CUDACC__) && !defined(MHD_SIMD_DISABLE)
#if defined(__AVX512F__)
#include <immintrin.h>
#define MHD_SIMD_AVX512 1
#elif defined(__AVX__)
#include <immintrin.h>
#define MHD_SIMD_AVX 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define MHD_SIMD_SSE2 1
#elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
#include <arm_neon.h>
#define MHD_SIMD_NEON 1
#endif
#endif

namespace mhd::simd {

// --- Scalar overloads (host + device) ------------------------------------------------

MHD_HD MHD_INLINE double vmin(double a, double b)
{
    return (b < a) ? b : a;
}
MHD_HD MHD_INLINE double vmax(double a, double b)
{
    return (a < b) ? b : a;
}
MHD_HD MHD_INLINE double vsqrt(double a)
{
    return sqrt(a);
}
MHD_HD MHD_INLINE double vabs(double a)
{
    return fabs(a);
}
MHD_HD MHD_INLINE double select(bool m, double a, double b)
{
    return m ? a : b;
}

MHD_HD MHD_INLINE float vmin(float a, float b)
{
    return (b < a) ? b : a;
}
MHD_HD MHD_INLINE float vmax(float a, float b)
{
    return (a < b) ? b : a;
}
MHD_HD MHD_INLINE float vsqrt(float a)
{
    return sqrtf(a);
}
MHD_HD MHD_INLINE float vabs(float a)
{
    return fabsf(a);
}
MHD_HD MHD_INLINE float select(bool m, float a, float b)
{
    return m ? a : b;
}

MHD_HD MHD_INLINE bool land(bool a, bool b)
{
    return a && b;
}
MHD_HD MHD_INLINE bool lor(bool a, bool b)
{
    return a || b;
}

template <class T>
struct lanes {
    static constexpr int value = 1;
};

// Scalar load/store from double storage (CPU SoA arrays are always double).
template <class T>
MHD_HD MHD_INLINE T load(const double* p);
template <>
MHD_HD MHD_INLINE double load<double>(const double* p)
{
    return *p;
}
MHD_HD MHD_INLINE void store(double* p, double v)
{
    *p = v;
}

#if !defined(__CUDACC__)

// --- Vector type ----------------------------------------------------------------------

#if defined(MHD_SIMD_AVX512)

inline constexpr int kWidth = 8;
inline constexpr const char* kBackendName = "AVX-512F";
struct vmask {
    __mmask8 m;
};
struct vdouble {
    __m512d v;
    vdouble() = default;
    MHD_INLINE vdouble(__m512d x) : v(x) {}
    MHD_INLINE vdouble(double x) : v(_mm512_set1_pd(x)) {}
};
MHD_INLINE vdouble operator+(vdouble a, vdouble b)
{
    return _mm512_add_pd(a.v, b.v);
}
MHD_INLINE vdouble operator-(vdouble a, vdouble b)
{
    return _mm512_sub_pd(a.v, b.v);
}
MHD_INLINE vdouble operator*(vdouble a, vdouble b)
{
    return _mm512_mul_pd(a.v, b.v);
}
MHD_INLINE vdouble operator/(vdouble a, vdouble b)
{
    return _mm512_div_pd(a.v, b.v);
}
MHD_INLINE vdouble operator-(vdouble a)
{
    return _mm512_sub_pd(_mm512_setzero_pd(), a.v);
}
MHD_INLINE vmask operator<(vdouble a, vdouble b)
{
    return {_mm512_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ)};
}
MHD_INLINE vmask operator<=(vdouble a, vdouble b)
{
    return {_mm512_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ)};
}
MHD_INLINE vmask operator>(vdouble a, vdouble b)
{
    return {_mm512_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ)};
}
MHD_INLINE vmask operator>=(vdouble a, vdouble b)
{
    return {_mm512_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ)};
}
MHD_INLINE vdouble vmin(vdouble a, vdouble b)
{
    return _mm512_min_pd(b.v, a.v);
}
MHD_INLINE vdouble vmax(vdouble a, vdouble b)
{
    return _mm512_max_pd(b.v, a.v);
}
MHD_INLINE vdouble vsqrt(vdouble a)
{
    return _mm512_sqrt_pd(a.v);
}
MHD_INLINE vdouble vabs(vdouble a)
{
    return _mm512_abs_pd(a.v);
}
MHD_INLINE vdouble select(vmask m, vdouble a, vdouble b)
{
    return _mm512_mask_blend_pd(m.m, b.v, a.v);
}
MHD_INLINE vmask land(vmask a, vmask b)
{
    return {static_cast<__mmask8>(a.m & b.m)};
}
MHD_INLINE vmask lor(vmask a, vmask b)
{
    return {static_cast<__mmask8>(a.m | b.m)};
}
template <>
MHD_INLINE vdouble load<vdouble>(const double* p)
{
    return _mm512_loadu_pd(p);
}
MHD_INLINE void store(double* p, vdouble a)
{
    _mm512_storeu_pd(p, a.v);
}

#elif defined(MHD_SIMD_AVX)

inline constexpr int kWidth = 4;
#if defined(__AVX2__)
inline constexpr const char* kBackendName = "AVX2";
#else
inline constexpr const char* kBackendName = "AVX";
#endif
struct vmask {
    __m256d m;
};
struct vdouble {
    __m256d v;
    vdouble() = default;
    MHD_INLINE vdouble(__m256d x) : v(x) {}
    MHD_INLINE vdouble(double x) : v(_mm256_set1_pd(x)) {}
};
MHD_INLINE vdouble operator+(vdouble a, vdouble b)
{
    return _mm256_add_pd(a.v, b.v);
}
MHD_INLINE vdouble operator-(vdouble a, vdouble b)
{
    return _mm256_sub_pd(a.v, b.v);
}
MHD_INLINE vdouble operator*(vdouble a, vdouble b)
{
    return _mm256_mul_pd(a.v, b.v);
}
MHD_INLINE vdouble operator/(vdouble a, vdouble b)
{
    return _mm256_div_pd(a.v, b.v);
}
MHD_INLINE vdouble operator-(vdouble a)
{
    return _mm256_sub_pd(_mm256_setzero_pd(), a.v);
}
MHD_INLINE vmask operator<(vdouble a, vdouble b)
{
    return {_mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ)};
}
MHD_INLINE vmask operator<=(vdouble a, vdouble b)
{
    return {_mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ)};
}
MHD_INLINE vmask operator>(vdouble a, vdouble b)
{
    return {_mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ)};
}
MHD_INLINE vmask operator>=(vdouble a, vdouble b)
{
    return {_mm256_cmp_pd(a.v, b.v, _CMP_GE_OQ)};
}
MHD_INLINE vdouble vmin(vdouble a, vdouble b)
{
    return _mm256_min_pd(b.v, a.v);
}
MHD_INLINE vdouble vmax(vdouble a, vdouble b)
{
    return _mm256_max_pd(b.v, a.v);
}
MHD_INLINE vdouble vsqrt(vdouble a)
{
    return _mm256_sqrt_pd(a.v);
}
MHD_INLINE vdouble vabs(vdouble a)
{
    return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a.v);
}
MHD_INLINE vdouble select(vmask m, vdouble a, vdouble b)
{
    return _mm256_blendv_pd(b.v, a.v, m.m);
}
MHD_INLINE vmask land(vmask a, vmask b)
{
    return {_mm256_and_pd(a.m, b.m)};
}
MHD_INLINE vmask lor(vmask a, vmask b)
{
    return {_mm256_or_pd(a.m, b.m)};
}
template <>
MHD_INLINE vdouble load<vdouble>(const double* p)
{
    return _mm256_loadu_pd(p);
}
MHD_INLINE void store(double* p, vdouble a)
{
    _mm256_storeu_pd(p, a.v);
}

#elif defined(MHD_SIMD_SSE2)

inline constexpr int kWidth = 2;
inline constexpr const char* kBackendName = "SSE2";
struct vmask {
    __m128d m;
};
struct vdouble {
    __m128d v;
    vdouble() = default;
    MHD_INLINE vdouble(__m128d x) : v(x) {}
    MHD_INLINE vdouble(double x) : v(_mm_set1_pd(x)) {}
};
MHD_INLINE vdouble operator+(vdouble a, vdouble b)
{
    return _mm_add_pd(a.v, b.v);
}
MHD_INLINE vdouble operator-(vdouble a, vdouble b)
{
    return _mm_sub_pd(a.v, b.v);
}
MHD_INLINE vdouble operator*(vdouble a, vdouble b)
{
    return _mm_mul_pd(a.v, b.v);
}
MHD_INLINE vdouble operator/(vdouble a, vdouble b)
{
    return _mm_div_pd(a.v, b.v);
}
MHD_INLINE vdouble operator-(vdouble a)
{
    return _mm_sub_pd(_mm_setzero_pd(), a.v);
}
MHD_INLINE vmask operator<(vdouble a, vdouble b)
{
    return {_mm_cmplt_pd(a.v, b.v)};
}
MHD_INLINE vmask operator<=(vdouble a, vdouble b)
{
    return {_mm_cmple_pd(a.v, b.v)};
}
MHD_INLINE vmask operator>(vdouble a, vdouble b)
{
    return {_mm_cmpgt_pd(a.v, b.v)};
}
MHD_INLINE vmask operator>=(vdouble a, vdouble b)
{
    return {_mm_cmpge_pd(a.v, b.v)};
}
MHD_INLINE vdouble vmin(vdouble a, vdouble b)
{
    return _mm_min_pd(b.v, a.v);
}
MHD_INLINE vdouble vmax(vdouble a, vdouble b)
{
    return _mm_max_pd(b.v, a.v);
}
MHD_INLINE vdouble vsqrt(vdouble a)
{
    return _mm_sqrt_pd(a.v);
}
MHD_INLINE vdouble vabs(vdouble a)
{
    return _mm_andnot_pd(_mm_set1_pd(-0.0), a.v);
}
MHD_INLINE vdouble select(vmask m, vdouble a, vdouble b)
{
    return _mm_or_pd(_mm_and_pd(m.m, a.v), _mm_andnot_pd(m.m, b.v));
}
MHD_INLINE vmask land(vmask a, vmask b)
{
    return {_mm_and_pd(a.m, b.m)};
}
MHD_INLINE vmask lor(vmask a, vmask b)
{
    return {_mm_or_pd(a.m, b.m)};
}
template <>
MHD_INLINE vdouble load<vdouble>(const double* p)
{
    return _mm_loadu_pd(p);
}
MHD_INLINE void store(double* p, vdouble a)
{
    _mm_storeu_pd(p, a.v);
}

#elif defined(MHD_SIMD_NEON)

inline constexpr int kWidth = 2;
inline constexpr const char* kBackendName = "NEON";
struct vmask {
    uint64x2_t m;
};
struct vdouble {
    float64x2_t v;
    vdouble() = default;
    MHD_INLINE vdouble(float64x2_t x) : v(x) {}
    MHD_INLINE vdouble(double x) : v(vdupq_n_f64(x)) {}
};
MHD_INLINE vdouble operator+(vdouble a, vdouble b)
{
    return vaddq_f64(a.v, b.v);
}
MHD_INLINE vdouble operator-(vdouble a, vdouble b)
{
    return vsubq_f64(a.v, b.v);
}
MHD_INLINE vdouble operator*(vdouble a, vdouble b)
{
    return vmulq_f64(a.v, b.v);
}
MHD_INLINE vdouble operator/(vdouble a, vdouble b)
{
    return vdivq_f64(a.v, b.v);
}
MHD_INLINE vdouble operator-(vdouble a)
{
    return vnegq_f64(a.v);
}
MHD_INLINE vmask operator<(vdouble a, vdouble b)
{
    return {vcltq_f64(a.v, b.v)};
}
MHD_INLINE vmask operator<=(vdouble a, vdouble b)
{
    return {vcleq_f64(a.v, b.v)};
}
MHD_INLINE vmask operator>(vdouble a, vdouble b)
{
    return {vcgtq_f64(a.v, b.v)};
}
MHD_INLINE vmask operator>=(vdouble a, vdouble b)
{
    return {vcgeq_f64(a.v, b.v)};
}
MHD_INLINE vdouble vmin(vdouble a, vdouble b)
{
    return vbslq_f64(vcltq_f64(b.v, a.v), b.v, a.v);
}
MHD_INLINE vdouble vmax(vdouble a, vdouble b)
{
    return vbslq_f64(vcltq_f64(a.v, b.v), b.v, a.v);
}
MHD_INLINE vdouble vsqrt(vdouble a)
{
    return vsqrtq_f64(a.v);
}
MHD_INLINE vdouble vabs(vdouble a)
{
    return vabsq_f64(a.v);
}
MHD_INLINE vdouble select(vmask m, vdouble a, vdouble b)
{
    return vbslq_f64(m.m, a.v, b.v);
}
MHD_INLINE vmask land(vmask a, vmask b)
{
    return {vandq_u64(a.m, b.m)};
}
MHD_INLINE vmask lor(vmask a, vmask b)
{
    return {vorrq_u64(a.m, b.m)};
}
template <>
MHD_INLINE vdouble load<vdouble>(const double* p)
{
    return vld1q_f64(p);
}
MHD_INLINE void store(double* p, vdouble a)
{
    vst1q_f64(p, a.v);
}

#else

// Scalar fallback: the "vector" is a plain double.
inline constexpr int kWidth = 1;
inline constexpr const char* kBackendName = "scalar";
using vdouble = double;

#endif

#if defined(MHD_SIMD_AVX512) || defined(MHD_SIMD_AVX) || defined(MHD_SIMD_SSE2) ||                 \
    defined(MHD_SIMD_NEON)
template <>
struct lanes<vdouble> {
    static constexpr int value = kWidth;
};
#endif

// Horizontal max (used by CFL reductions).
#if defined(MHD_SIMD_AVX512) || defined(MHD_SIMD_AVX) || defined(MHD_SIMD_SSE2) ||                 \
    defined(MHD_SIMD_NEON)
inline double hmax(vdouble a)
{
    alignas(64) double tmp[kWidth];
    store(tmp, a);
    double m = tmp[0];
    for (int k = 1; k < kWidth; ++k) {
        m = vmax(m, tmp[k]);
    }
    return m;
}
#else
inline double hmax(double a)
{
    return a;
}
#endif

#endif // !__CUDACC__

} // namespace mhd::simd
