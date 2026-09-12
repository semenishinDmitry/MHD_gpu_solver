#!/usr/bin/env bash
# Portable CI-style build (no -march=native). Useful locally and as a reference
# for what GitHub Actions runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build-ci}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CXX_COMPILER="${CXX_COMPILER:-$(command -v clang++ || command -v g++)}"
JOBS="${JOBS:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
BUILD_BENCHMARKS="${BUILD_BENCHMARKS:-ON}"
GENERATOR=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR=(-G Ninja)
fi

echo "==> Configure ($BUILD_TYPE, CXX=$CXX_COMPILER, dir=$BUILD_DIR)"
cmake -S . -B "$BUILD_DIR" "${GENERATOR[@]}" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
  -DMHD_BUILD_TESTS=ON \
  -DMHD_BUILD_PYTHON=ON \
  -DMHD_BUILD_BENCHMARKS="$BUILD_BENCHMARKS" \
  -DMHD_NATIVE_ARCH=OFF \
  -DMHD_ENABLE_LTO=OFF

echo "==> Build"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j "$JOBS"

echo "==> Test (unit + regression)"
ctest --test-dir "$BUILD_DIR" --output-on-failure --build-config "$BUILD_TYPE" --parallel "$JOBS" -LE benchmark

if [[ "$BUILD_BENCHMARKS" == ON ]]; then
  echo "==> Benchmarks (quick)"
  ctest --test-dir "$BUILD_DIR" --output-on-failure -L benchmark || true
  "$BUILD_DIR/benchmarks/mhd_bench" --benchmark_filter='BM_OrszagTang_Run/32' --benchmark_min_time=0.05s || true
fi

echo "==> Python smoke"
export PYTHONPATH="$ROOT/$BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<'PY'
import mhd_solver as mhd
s = mhd.MHDSolver(16, 16)
s.initialize("orszag_tang")
s.run(0.01)
print("ok", s.time, s.steps)
PY

echo "==> Example"
python3 examples/python/orszag_tang.py

echo "Done."
