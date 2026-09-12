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
