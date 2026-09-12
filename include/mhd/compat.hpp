#pragma once

// Portable restrict qualifier (MSVC uses __restrict, GCC/Clang use __restrict__).
#if defined(_MSC_VER)
#  define MHD_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#  define MHD_RESTRICT __restrict__
#else
#  define MHD_RESTRICT
#endif
