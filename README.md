# MHD Solver

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
