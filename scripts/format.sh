#!/usr/bin/env bash
# Format or check C++ sources with a pinned clang-format (PyPI wheel).
# Same version locally and in CI: MHD_CLANG_FORMAT_VERSION (default 19.1.7).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PINNED_VERSION="${MHD_CLANG_FORMAT_VERSION:-19.1.7}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to install/run pinned clang-format." >&2
  exit 1
fi

have_ver="$(python3 - <<'PY' 2>/dev/null || true
try:
    import importlib.metadata as m
    print(m.version("clang-format"))
except Exception:
    pass
PY
)"

if [[ "$have_ver" != "$PINNED_VERSION" ]]; then
  echo "Installing clang-format==$PINNED_VERSION (was: ${have_ver:-missing})"
  python3 -m pip install -q "clang-format==${PINNED_VERSION}"
fi

# Resolve the binary shipped by the clang-format wheel (portable across OS).
CLANG_FORMAT_BIN="$(python3 - <<'PY'
import glob
import os
import clang_format

root = os.path.dirname(clang_format.__file__)
patterns = [
    os.path.join(root, "data", "bin", "clang-format"),
    os.path.join(root, "data", "bin", "clang-format.exe"),
]
cands = []
for pat in patterns:
    cands.extend(glob.glob(pat))
if not cands:
    raise SystemExit(f"clang-format binary not found under {root}")
print(cands[0])
PY
)"

if [[ ! -x "$CLANG_FORMAT_BIN" ]]; then
  # Windows wheels may lack +x; still runnable
  if [[ ! -f "$CLANG_FORMAT_BIN" ]]; then
    echo "ERROR: clang-format binary missing: $CLANG_FORMAT_BIN" >&2
    exit 1
  fi
fi

echo "Using: $CLANG_FORMAT_BIN ($("$CLANG_FORMAT_BIN" --version | head -n 1))"

FILES=()
while IFS= read -r f; do
  FILES+=("$f")
done < <(
  find include tests benchmarks python main.cpp \
    \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) \
    -type f 2>/dev/null | sort
)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No C++ files found."
  exit 1
fi

MODE="${1:-format}"

case "$MODE" in
  format|fix)
    echo "clang-format: formatting ${#FILES[@]} files"
    "$CLANG_FORMAT_BIN" -i "${FILES[@]}"
    ;;
  check)
    echo "clang-format: checking ${#FILES[@]} files"
    bad=0
    for f in "${FILES[@]}"; do
      if ! "$CLANG_FORMAT_BIN" --dry-run -Werror "$f" >/dev/null 2>&1; then
        echo "Needs format: $f"
        bad=1
      fi
    done
    if [[ "$bad" -ne 0 ]]; then
      echo "clang-format check failed. Run: ./scripts/format.sh" >&2
      exit 1
    fi
    echo "clang-format: OK"
    ;;
  *)
    echo "Usage: $0 [format|check]" >&2
    exit 2
    ;;
esac
