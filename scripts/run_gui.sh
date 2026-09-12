#!/usr/bin/env bash
# Launch the MHD Solver GUI (macOS / Linux).
# Uses the browser UI by default — Apple's system Tk 8.5 cannot reliably
# render desktop tkinter/matplotlib windows.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d build/python ]]; then
  echo "Python module missing. Running setup_and_build.sh ..."
  "$ROOT/scripts/setup_and_build.sh"
fi

python3 - <<'PY' || python3 -m pip install --user -q matplotlib numpy
import matplotlib, numpy
PY

export PYTHONPATH="$ROOT/build/python${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

# Optional: MHD_GUI=tk ./scripts/run_gui.sh  → old tkinter UI
if [[ "${MHD_GUI:-web}" == "tk" ]]; then
  export TK_SILENCE_DEPRECATION="${TK_SILENCE_DEPRECATION:-1}"
  exec python3 "$ROOT/gui/app.py"
fi

exec python3 "$ROOT/gui/web_app.py"
