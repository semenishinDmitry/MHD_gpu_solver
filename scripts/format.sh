#!/usr/bin/env bash
# Format or check C++ sources with clang-format (repo .clang-format).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v clang-format >/dev/null 2>&1; then
  echo "ERROR: clang-format not found. Install it (e.g. brew install clang-format / apt install clang-format)." >&2
  exit 1
fi

# Portable file list (no mapfile — macOS /bin/bash is often 3.x)
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
    clang-format -i "${FILES[@]}"
    ;;
  check)
    echo "clang-format: checking ${#FILES[@]} files"
    bad=0
    for f in "${FILES[@]}"; do
      if ! clang-format --dry-run -Werror "$f" >/dev/null 2>&1; then
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
