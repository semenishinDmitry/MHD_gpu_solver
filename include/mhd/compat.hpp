#pragma once

// Portable restrict qualifier (MSVC uses __restrict, GCC/Clang use __restrict__).
#if defined(_MSC_VER)
#define MHD_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define MHD_RESTRICT __restrict__
#else
#define MHD_RESTRICT
#endif

// Host/device qualifier for kernels shared between the CPU build and CUDA (nvcc).
#if defined(__CUDACC__)
#define MHD_HD __host__ __device__
#else
#define MHD_HD
#endif

// Force inlining of small hot kernels (the generic SIMD kernels rely on it).
#if defined(__CUDACC__)
#define MHD_INLINE __forceinline__
#elif defined(_MSC_VER)
#define MHD_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define MHD_INLINE inline __attribute__((always_inline))
#else
#define MHD_INLINE inline
#endif
