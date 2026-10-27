# Build

## Requirements

- CMake ≥ 3.20
- C++20 compiler (Clang/LLVM preferred; GCC and MSVC also used in CI)
- Python 3.x + development headers (only if `-DMHD_BUILD_PYTHON=ON`)
- Git (optional; for commit stamp in build info)

## Configure & build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMHD_NATIVE_ARCH=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/mhd_verify --build-info
```

### Important options

| Option | Default | Meaning |
|--------|---------|---------|
| `CMAKE_BUILD_TYPE` | Release | Debug / Release / RelWithDebInfo |
| `MHD_NATIVE_ARCH` | ON | `-march=native` (turn **OFF** for portable/CI/refs) |
| `MHD_ENABLE_SANITIZERS` | OFF | ASan + UBSan |
| `MHD_BUILD_TESTS` | ON | GoogleTest suite |
| `MHD_BUILD_PYTHON` | ON | pybind11 module |
| `MHD_BUILD_BENCHMARKS` | OFF | Google Benchmark |
| `MHD_ENABLE_METAL` | ON on macOS | Metal GPU backend (fp32), see [gpu.md](gpu.md) |
| `MHD_ENABLE_CUDA` | OFF | CUDA GPU backend (fp64/fp32), needs the CUDA toolkit |
| `MHD_CUDA_ARCHITECTURES` | `70;80;86;89;90` | `CMAKE_CUDA_ARCHITECTURES` for the CUDA backend |

### SIMD instruction set

The CPU kernels pick their SIMD backend from the compiler target (see
[performance.md](performance.md#2-simd)). With `MHD_NATIVE_ARCH=ON` this is the best set for the
build machine. Portable builds get SSE2 (x86-64) or NEON (ARM64). To target AVX2 explicitly:

```bash
cmake -S . -B build -DMHD_NATIVE_ARCH=OFF -DCMAKE_CXX_FLAGS="-mavx2 -mfma"   # GCC / Clang
cmake -S . -B build -DMHD_NATIVE_ARCH=OFF -DCMAKE_CXX_FLAGS="/arch:AVX2"     # MSVC
```

Threads are controlled at run time (`MHD_NUM_THREADS`, `mhd::set_num_threads`), not at
build time.

### Sanitizer build

```bash
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMHD_ENABLE_SANITIZERS=ON -DMHD_NATIVE_ARCH=OFF -DMHD_BUILD_PYTHON=OFF
cmake --build build-san -j
ctest --test-dir build-san --output-on-failure
```

### Formatting / tidy

```bash
./scripts/format.sh check          # pinned clang-format 19.1.7
cmake --build build --target lint  # clang-tidy if installed
```

## Reproducibility

`mhd_verify --build-info` prints compiler, C++ standard, CMake version, build type,
git commit, platform, and key options. Prefer this over informal “it worked on my machine”
notes when comparing numerical results.
