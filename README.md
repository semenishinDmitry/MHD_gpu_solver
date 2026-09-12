# MHD Solver

[![CI](https://github.com/semenishinDmitry/MHD_gpu_solver/actions/workflows/ci.yml/badge.svg)](https://github.com/semenishinDmitry/MHD_gpu_solver/actions/workflows/ci.yml)

2D ideal / non-ideal GLM-MHD finite-volume solver (MUSCL + HLL + SSP-RK2), with optional Python bindings.

## Quick start (recommended)

**macOS / Linux**
```bash
chmod +x scripts/setup_and_build.sh
./scripts/setup_and_build.sh
```

**Windows**
```bat
scripts\setup_and_build.bat
```
or
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_and_build.ps1
```

The scripts install missing tools when possible (Homebrew / apt / winget), prefer **LLVM clang++**, configure **Release** with native CPU flags, build, and run tests.

## Manual build (Clang/LLVM)

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DMHD_BUILD_TESTS=ON \
  -DMHD_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/mhd_solver
```

Useful options:
- `-DMHD_NATIVE_ARCH=ON` (default) — `-march=native`
- `-DMHD_ENABLE_LTO=ON` — Clang thin LTO
- `-DMHD_BUILD_PYTHON=OFF` — skip pybind11
- `-DMHD_BUILD_TESTS=OFF` — skip GoogleTest

## Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR to `main`:

| Job | What it checks |
|-----|----------------|
| clang-format | `./scripts/format.sh check` against repo `.clang-format` |
| Ubuntu Clang / GCC | Release build, unit + **regression goldens**, Python smoke |
| macOS AppleClang | Same |
| Windows MSVC | Same |
| Ubuntu Clang + benchmarks | Google Benchmark (`mhd_bench`) quick run |
| Ubuntu Clang Debug + ASan/UBSan | Tests under AddressSanitizer + UndefinedBehaviorSanitizer |

CI always sets `-DMHD_NATIVE_ARCH=OFF` (portable codegen). To reproduce locally:

```bash
chmod +x scripts/ci_build.sh scripts/format.sh
./scripts/format.sh check
./scripts/ci_build.sh
```

### Formatting

Single style: **clang-format 19.1.7** (pinned via PyPI; same in CI and locally).

```bash
./scripts/format.sh          # rewrite
./scripts/format.sh check    # CI gate
```

Override pin with `MHD_CLANG_FORMAT_VERSION=19.1.7` if needed. Do not use Homebrew/apt `clang-format` for checks — versions disagree (e.g. 23 vs 14).

### Regression goldens

Reference fields live in `tests/goldens/*.golden` (Orszag–Tang 32² and Ohmic Bz decay).
Compared with relative L² / L∞ tolerances in `tests/test_regression_golden.cpp`.

Regenerate after intentional physics/numerics changes (Release, `MHD_NATIVE_ARCH=OFF`):

```bash
cmake -S . -B build-ci -DMHD_NATIVE_ARCH=OFF -DMHD_BUILD_TESTS=ON
cmake --build build-ci -j
MHD_REGEN_GOLDENS=1 ./build-ci/tests/mhd_tests --gtest_filter='Regression.*'
```

### Benchmarks

```bash
cmake -S . -B build -DMHD_BUILD_BENCHMARKS=ON -DMHD_NATIVE_ARCH=OFF
cmake --build build -j --target mhd_bench
./build/benchmarks/mhd_bench --benchmark_filter=BM_OrszagTang_Run
```

## Desktop GUI (Windows / macOS / Linux)

After a successful build:

```bash
./scripts/run_gui.sh          # macOS / Linux
scripts\run_gui.bat           # Windows
```

The GUI lets you set grid/time/CFL/IC/limiter/BC and Ohmic/Hall/Ambipolar, then:
- run the solver,
- plot any conserved field on the interior mesh,
- scrub time with a slider,
- play an animation over all snapshots.

Browser UI is used by default (`gui/web_app.py`); set `MHD_GUI=tk` only if you need the legacy tkinter window.

## Python

```bash
export PYTHONPATH="$PWD/build/python:$PYTHONPATH"   # Windows: set PYTHONPATH=...\build\python;%PYTHONPATH%
python examples/python/orszag_tang.py
```

```python
import mhd_solver as mhd
s = mhd.MHDSolver(64, 64)
s.enable_ohmic(1e-3)          # optional; omit for ideal MHD
s.initialize("orszag_tang")
s.run(0.05)
print(s.max_div_b(), s.field("rho").shape)
```

## Performance notes

Hot paths (`compute_rhs`, SSP-RK2, field SoA kernels, non-ideal fluxes) do **not** allocate:
- all scratch lives in `TimeIntegratorWorkspace` / `RHSWorkspace`
- ideal MHD skips non-ideal `J`/`E` work entirely (`NonIdealConfig::any() == false`)
- Release builds use `-O3`, `-fno-math-errno`, `-ffp-contract=fast`, vectorize (Clang), `-march=native`
