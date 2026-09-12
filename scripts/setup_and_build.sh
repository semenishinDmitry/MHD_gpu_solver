#!/usr/bin/env bash
# Setup + build for macOS / Linux (LLVM/Clang preferred).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_PYTHON="${BUILD_PYTHON:-ON}"
BUILD_TESTS="${BUILD_TESTS:-ON}"
JOBS="${JOBS:-$(command -v nproc >/dev/null && nproc || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

info() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

detect_os() {
  case "$(uname -s)" in
    Darwin) echo mac ;;
    Linux) echo linux ;;
    *) echo other ;;
  esac
}

OS="$(detect_os)"
info "OS: $OS"

# --- dependencies ------------------------------------------------------------
need_install=()

if ! have cmake; then need_install+=(cmake); fi
if ! have git; then need_install+=(git); fi
if ! have python3; then need_install+=(python3); fi

CXX_COMPILER=""
if have clang++; then
  CXX_COMPILER="$(command -v clang++)"
elif have clang++-18; then
  CXX_COMPILER="$(command -v clang++-18)"
elif have clang++-17; then
  CXX_COMPILER="$(command -v clang++-17)"
elif have g++; then
  warn "clang++ not found; falling back to g++"
  CXX_COMPILER="$(command -v g++)"
else
  need_install+=(clang)
fi

if ((${#need_install[@]})); then
  info "Missing tools: ${need_install[*]}"
  if [[ "$OS" == mac ]]; then
    if have brew; then
      brew install "${need_install[@]}"
    else
      die "Install Homebrew (https://brew.sh) then re-run, or install: ${need_install[*]}"
    fi
  elif [[ "$OS" == linux ]]; then
    if have apt-get; then
      sudo apt-get update
      # Map generic names to apt packages
      pkgs=()
      for p in "${need_install[@]}"; do
        case "$p" in
          clang) pkgs+=(clang lld);;
          python3) pkgs+=(python3 python3-dev python3-pip python3-tk);;
          *) pkgs+=("$p");;
        esac
      done
      sudo apt-get install -y "${pkgs[@]}"
    elif have dnf; then
      sudo dnf install -y "${need_install[@]}"
    elif have pacman; then
      sudo pacman -S --needed --noconfirm "${need_install[@]}"
    else
      die "Install manually: ${need_install[*]}"
    fi
  else
    die "Unsupported OS. Install manually: ${need_install[*]}"
  fi

  # Re-detect clang after install
  if [[ -z "$CXX_COMPILER" ]]; then
    if have clang++; then CXX_COMPILER="$(command -v clang++)";
    elif have g++; then CXX_COMPILER="$(command -v g++)";
    else die "No C++ compiler found after install"; fi
  fi
fi

info "Using C++ compiler: $CXX_COMPILER"
"$CXX_COMPILER" --version | head -n 1 || true

if have python3; then
  python3 -m pip install --user -q numpy matplotlib || warn "pip/numpy/matplotlib optional install failed"
  # Ensure Tk bindings exist (needed by the GUI).
  python3 - <<'PY' 2>/dev/null || warn "tkinter missing (Linux: sudo apt install python3-tk)"
import tkinter
PY
fi

# --- configure & build -------------------------------------------------------
info "Configuring ($BUILD_TYPE) into $BUILD_DIR"
cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
  -DMHD_BUILD_TESTS="$BUILD_TESTS" \
  -DMHD_BUILD_PYTHON="$BUILD_PYTHON" \
  -DMHD_NATIVE_ARCH=ON

info "Building with $JOBS jobs"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j "$JOBS"

if [[ "$BUILD_TESTS" == ON ]]; then
  info "Running tests"
  ctest --test-dir "$BUILD_DIR" --output-on-failure --build-config "$BUILD_TYPE"
fi

info "Done"
echo "  binary : $BUILD_DIR/mhd_solver"
if [[ "$BUILD_PYTHON" == ON ]]; then
  echo "  python : export PYTHONPATH=\"$ROOT/$BUILD_DIR/python:\$PYTHONPATH\""
  echo "           python3 -c 'import mhd_solver; print(mhd_solver.MHDSolver)'"
fi
