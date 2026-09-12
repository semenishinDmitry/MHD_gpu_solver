#!/usr/bin/env python3
"""
Browser-based MHD Solver GUI (Windows / macOS / Linux).

Avoids Apple's broken system Tk 8.5 by using only the stdlib http.server
and matplotlib's Agg backend.
"""

from __future__ import annotations

import html
import io
import json
import os
import sys
import threading
import traceback
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PY = os.path.join(_ROOT, "build", "python")
for path in (_PY, _ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mhd_solver as mhd


FIELDS = ("rho", "mx", "my", "mz", "energy", "bx", "by", "bz", "psi")
ICS = ("orszag_tang", "sine", "blast", "uniform")
LIMITERS = ("MC", "Minmod")
BCS = ("Periodic", "Outflow")

HOST = "127.0.0.1"
PORT = int(os.environ.get("MHD_GUI_PORT", "8765"))
PORT_SCAN = 32  # try PORT .. PORT+PORT_SCAN-1 if busy

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "busy": False,
    "status": "Ready. Press Run to start.",
    "error": "",
    "history": [],
    "times": [],
    "clim": {},
    "meta": {},
}


def interior(arr: np.ndarray, ng: int) -> np.ndarray:
    if ng <= 0:
        return arr
    return arr[ng:-ng, ng:-ng]


def run_simulation(params: dict[str, Any]) -> None:
    with _state_lock:
        _state["busy"] = True
        _state["error"] = ""
        _state["status"] = "Running…"
        _state["history"] = []
        _state["times"] = []
        _state["clim"] = {}
        _state["meta"] = {}

    try:
        solver = mhd.MHDSolver(params["nx"], params["ny"])
        solver.set_cfl(params["cfl"])
        solver.set_gamma(params["gamma"])
        solver.set_glm_alpha(params["glm"])
        if params["limiter"] == "Minmod":
            solver.set_limiter_minmod()
        else:
            solver.set_limiter_mc()
        if params["bc"] == "Outflow":
            solver.set_outflow_bc()
        else:
            solver.set_periodic_bc()

        solver.set_ideal()
        if params["ohmic"]:
            solver.enable_ohmic(params["eta_o"])
        if params["hall"]:
            solver.enable_hall(params["eta_h"])
        if params["amb"]:
            solver.enable_ambipolar(params["eta_a"])

        solver.initialize(params["ic"])
        ng = solver.ng
        nsnap = params["nsnap"]
        t_end = params["t_end"]
        dt_snap = t_end / (nsnap - 1)

        def snap() -> dict[str, np.ndarray]:
            return {name: interior(np.array(solver.field(name), copy=True), ng) for name in FIELDS}

        history = [snap()]
        times = [float(solver.time)]
        for k in range(1, nsnap):
            solver.advance_to(min(k * dt_snap, t_end))
            history.append(snap())
            times.append(float(solver.time))
            with _state_lock:
                _state["status"] = f"Running… {k + 1}/{nsnap}"

        clim: dict[str, tuple[float, float]] = {}
        for name in FIELDS:
            stack = np.stack([frame[name] for frame in history], axis=0)
            vmin = float(np.min(stack))
            vmax = float(np.max(stack))
            if vmin == vmax:
                vmax = vmin + 1e-30
            clim[name] = (vmin, vmax)

        meta = {"steps": int(solver.steps), "c_h": float(solver.c_h), "divb": float(solver.max_div_b())}
        with _state_lock:
            _state["history"] = history
            _state["times"] = times
            _state["clim"] = clim
            _state["meta"] = meta
            _state["status"] = (
                f"Done. steps={meta['steps']}  c_h={meta['c_h']:.4g}  max|div B|={meta['divb']:.4g}"
            )
    except Exception as exc:  # noqa: BLE001
        with _state_lock:
            _state["error"] = str(exc)
            _state["status"] = "Failed."
            _state["history"] = []
            _state["times"] = []
        traceback.print_exc()
    finally:
        with _state_lock:
            _state["busy"] = False


def render_plot(field: str, frame: int, fixed_clim: bool) -> bytes:
    with _state_lock:
        history = _state["history"]
        times = _state["times"]
        clim = _state["clim"]

    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=110)
    if not history:
        ax.text(0.5, 0.5, "No data yet.\nPress Run.", ha="center", va="center", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("MHD Solver")
    else:
        idx = int(np.clip(frame, 0, len(history) - 1))
        if field not in history[idx]:
            field = "rho"
        data = history[idx][field]
        t = times[idx]
        im = ax.imshow(data, origin="lower", aspect="equal", cmap="inferno", interpolation="nearest")
        if fixed_clim and field in clim:
            im.set_clim(*clim[field])
        else:
            im.set_clim(float(np.min(data)), float(np.max(data)) + 1e-30)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{field}   t = {t:.5g}")
        ax.set_xlabel("i")
        ax.set_ylabel("j")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MHD Solver</title>
<style>
  :root { --bg:#f4f5f7; --panel:#fff; --fg:#1a1a1a; --muted:#555; --accent:#1a5fb4; --border:#d0d4dc; }
  * { box-sizing: border-box; }
  body { margin:0; font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         color:var(--fg); background:var(--bg); }
  .wrap { display:grid; grid-template-columns: 320px 1fr; min-height:100vh; }
  aside { background:var(--panel); border-right:1px solid var(--border); padding:16px; overflow:auto; }
  main { padding:16px; display:flex; flex-direction:column; gap:12px; }
  h1 { font-size:20px; margin:0 0 12px; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:14px 0 8px; }
  label { display:block; font-size:12px; color:var(--muted); margin:6px 0 2px; }
  input[type=text], input[type=number], select {
    width:100%; padding:7px 8px; border:1px solid var(--border); border-radius:6px; background:#fff; }
  .row2 { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .chk { display:flex; align-items:center; gap:8px; margin:6px 0; }
  button {
    width:100%; margin-top:8px; padding:10px 12px; border:0; border-radius:8px;
    background:var(--accent); color:#fff; font-weight:600; cursor:pointer; }
  button.secondary { background:#5b616b; }
  button:disabled { opacity:.55; cursor:default; }
  #status { margin-top:12px; color:var(--accent); min-height:2.5em; white-space:pre-wrap; }
  #err { color:#b00020; white-space:pre-wrap; }
  #plot { width:100%; max-width:920px; background:#fff; border:1px solid var(--border); border-radius:8px; }
  .eq-box {
    max-width:920px; background:var(--panel); border:1px solid var(--border); border-radius:8px;
    padding:12px 16px; }
  .eq-box h2 { margin:0 0 8px; }
  .eq-box .mj { font-size:15px; overflow-x:auto; }
  .eq-box .note { font-size:12px; color:var(--muted); margin-top:8px; }
  .toolbar { display:flex; align-items:center; gap:10px; max-width:920px; }
  .toolbar input[type=range] { flex:1; }
  @media (max-width: 900px) { .wrap { grid-template-columns: 1fr; } }
</style>
<script>
MathJax = {
  tex: { inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre'] }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
<div class="wrap">
  <aside>
    <h1>MHD Solver</h1>
    <h2>Simulation</h2>
    <div class="row2">
      <div><label>nx</label><input id="nx" type="number" value="64" min="8"/></div>
      <div><label>ny</label><input id="ny" type="number" value="64" min="8"/></div>
    </div>
    <div class="row2">
      <div><label>t_end</label><input id="t_end" type="text" value="0.05"/></div>
      <div><label>Snapshots</label><input id="nsnap" type="number" value="40" min="2"/></div>
    </div>
    <div class="row2">
      <div><label>CFL</label><input id="cfl" type="text" value="0.4"/></div>
      <div><label>gamma</label><input id="gamma" type="text" value="1.6666666667"/></div>
    </div>
    <label>GLM alpha</label><input id="glm" type="text" value="0.1"/>
    <label>Initial condition</label>
    <select id="ic">__ICS__</select>
    <label>Limiter</label>
    <select id="limiter">__LIMS__</select>
    <label>Boundary conditions</label>
    <select id="bc">__BCS__</select>

    <h2>Non-ideal</h2>
    <label class="chk"><input id="ohmic" type="checkbox" onchange="updateEquations()"/> Ohmic</label>
    <label>eta_ohm</label><input id="eta_o" type="text" value="0.001"/>
    <label class="chk"><input id="hall" type="checkbox" onchange="updateEquations()"/> Hall</label>
    <label>eta_hall</label><input id="eta_h" type="text" value="0.001"/>
    <label class="chk"><input id="amb" type="checkbox" onchange="updateEquations()"/> Ambipolar</label>
    <label>eta_ambipolar</label><input id="eta_a" type="text" value="0.001"/>

    <h2>Visualization</h2>
    <label>Field</label>
    <select id="field">__FIELDS__</select>
    <label class="chk"><input id="fixed_clim" type="checkbox" checked/> Fixed color scale</label>

    <button id="runBtn" onclick="runSim()">Run</button>
    <button id="playBtn" class="secondary" onclick="togglePlay()" disabled>Play</button>
    <div id="status"></div>
    <div id="err"></div>
  </aside>
  <main>
    <div class="eq-box">
      <h2>Equations</h2>
      <div id="equations" class="mj"></div>
      <div class="note" id="eqNote"></div>
    </div>
    <img id="plot" alt="field plot" src="/plot.png?field=rho&frame=0&fixed=1"/>
    <div class="toolbar">
      <span>t</span>
      <input id="frame" type="range" min="0" max="0" value="0" oninput="onFrame()"/>
      <span id="tlabel">t = —</span>
    </div>
  </main>
</div>
<script>
let playing = false, playTimer = null, nFrames = 0, times = [];

function updateEquations() {
  const ohmic = document.getElementById('ohmic').checked;
  const hall = document.getElementById('hall').checked;
  const amb = document.getElementById('amb').checked;
  const ni = ohmic || hall || amb;
  const gamma = document.getElementById('gamma').value || '\\gamma';
  const alpha = document.getElementById('glm').value || '\\alpha';

  let faradayRhs = '';
  let eDef = '';
  if (ni) {
    faradayRhs = ' - \\\\nabla\\\\times\\\\mathbf{E}_{\\\\mathrm{ni}}';
    const terms = [];
    if (ohmic) terms.push('\\\\eta_{\\\\mathrm{ohm}}\\\\,\\\\mathbf{J}');
    if (hall) terms.push('\\\\eta_{\\\\mathrm{Hall}}\\\\,\\\\mathbf{J}\\\\times\\\\hat{\\\\mathbf{B}}');
    if (amb) terms.push('\\\\eta_{\\\\mathrm{A}}\\\\,(\\\\mathbf{J}\\\\times\\\\mathbf{B})\\\\times\\\\mathbf{B}/B^2');
    eDef = '\\\\[\\\\mathbf{E}_{\\\\mathrm{ni}} = ' + terms.join(' + ') + ',\\\\quad \\\\mathbf{J}=\\\\nabla\\\\times\\\\mathbf{B}.\\\\]';
  }

  const html = `
\\\\[
\\\\partial_t\\\\rho + \\\\nabla\\\\cdot(\\\\rho\\\\mathbf{v}) = 0
\\\\]
\\\\[
\\\\partial_t(\\\\rho\\\\mathbf{v})
  + \\\\nabla\\\\cdot\\\\!\\\\big(
      \\\\rho\\\\mathbf{v}\\\\mathbf{v}
      + (p + B^2/2)\\\\,\\\\mathbf{I}
      - \\\\mathbf{B}\\\\mathbf{B}
    \\\\big) = 0
\\\\]
\\\\[
\\\\partial_t E
  + \\\\nabla\\\\cdot\\\\!\\\\big(
      (E + p + B^2/2)\\\\mathbf{v}
      - (\\\\mathbf{v}\\\\cdot\\\\mathbf{B})\\\\mathbf{B}
      + \\\\psi\\\\mathbf{B}
    \\\\big)
  ${ni ? '+ \\\\nabla\\\\cdot(\\\\mathbf{E}_{\\\\mathrm{ni}}\\\\times\\\\mathbf{B})' : ''} = 0
\\\\]
\\\\[
\\\\partial_t\\\\mathbf{B}
  + \\\\nabla\\\\cdot(\\\\mathbf{v}\\\\mathbf{B} - \\\\mathbf{B}\\\\mathbf{v})
  + \\\\nabla\\\\psi
  ${faradayRhs} = 0
\\\\]
\\\\[
\\\\partial_t\\\\psi + c_h^2\\\\,\\\\nabla\\\\cdot\\\\mathbf{B}
  = -\\\\frac{\\\\alpha\\\\, c_h}{h}\\\\,\\\\psi
\\\\]
${eDef}
\\\\[
E = \\\\frac{p}{\\\\gamma-1} + \\\\tfrac12\\\\rho|\\\\mathbf{v}|^2 + \\\\tfrac12|\\\\mathbf{B}|^2,
\\\\quad \\\\gamma=${gamma},\\; \\\\alpha=${alpha}
\\\\]
`;
  const el = document.getElementById('equations');
  el.innerHTML = html;
  document.getElementById('eqNote').textContent = ni
    ? 'Ideal GLM-MHD + selected non-ideal terms (2D finite-volume, MUSCL + HLL, SSP-RK2).'
    : 'Ideal GLM-MHD with Dedner hyperbolic/parabolic divergence cleaning (2D finite-volume, MUSCL + HLL, SSP-RK2).';
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetClear([el]);
    MathJax.typesetPromise([el]).catch(() => {});
  }
}

function opts() {
  return {
    nx: +document.getElementById('nx').value,
    ny: +document.getElementById('ny').value,
    t_end: +document.getElementById('t_end').value,
    nsnap: +document.getElementById('nsnap').value,
    cfl: +document.getElementById('cfl').value,
    gamma: +document.getElementById('gamma').value,
    glm: +document.getElementById('glm').value,
    ic: document.getElementById('ic').value,
    limiter: document.getElementById('limiter').value,
    bc: document.getElementById('bc').value,
    ohmic: document.getElementById('ohmic').checked,
    hall: document.getElementById('hall').checked,
    amb: document.getElementById('amb').checked,
    eta_o: +document.getElementById('eta_o').value,
    eta_h: +document.getElementById('eta_h').value,
    eta_a: +document.getElementById('eta_a').value,
  };
}

function plotUrl() {
  const field = document.getElementById('field').value;
  const frame = document.getElementById('frame').value;
  const fixed = document.getElementById('fixed_clim').checked ? 1 : 0;
  return `/plot.png?field=${encodeURIComponent(field)}&frame=${frame}&fixed=${fixed}&_=${Date.now()}`;
}

function refreshPlot() {
  document.getElementById('plot').src = plotUrl();
  const i = +document.getElementById('frame').value;
  if (times.length) {
    document.getElementById('tlabel').textContent =
      `t = ${times[i].toPrecision(5)}  (${i+1}/${nFrames})`;
  }
}

async function poll() {
  const r = await fetch('/api/status');
  const s = await r.json();
  document.getElementById('status').textContent = s.status || '';
  document.getElementById('err').textContent = s.error || '';
  document.getElementById('runBtn').disabled = !!s.busy;
  if (!s.busy && s.nframes > 0) {
    nFrames = s.nframes;
    times = s.times || [];
    const fr = document.getElementById('frame');
    fr.max = Math.max(nFrames - 1, 0);
    document.getElementById('playBtn').disabled = false;
    refreshPlot();
  }
  if (s.busy) setTimeout(poll, 400);
}

async function runSim() {
  stopPlay();
  updateEquations();
  document.getElementById('runBtn').disabled = true;
  document.getElementById('playBtn').disabled = true;
  document.getElementById('err').textContent = '';
  document.getElementById('status').textContent = 'Running…';
  const r = await fetch('/api/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(opts())
  });
  const j = await r.json();
  if (!j.ok) {
    document.getElementById('err').textContent = j.error || 'Failed to start';
    document.getElementById('runBtn').disabled = false;
    return;
  }
  poll();
}

function onFrame() { if (!playing) refreshPlot(); }

function togglePlay() {
  if (playing) stopPlay();
  else {
    playing = true;
    document.getElementById('playBtn').textContent = 'Pause';
    playStep();
  }
}
function stopPlay() {
  playing = false;
  document.getElementById('playBtn').textContent = 'Play';
  if (playTimer) { clearTimeout(playTimer); playTimer = null; }
}
function playStep() {
  if (!playing || nFrames < 1) return;
  const fr = document.getElementById('frame');
  let next = (+fr.value) + 1;
  if (next >= nFrames) next = 0;
  fr.value = next;
  refreshPlot();
  playTimer = setTimeout(playStep, 80);
}

document.getElementById('field').addEventListener('change', () => refreshPlot());
document.getElementById('fixed_clim').addEventListener('change', () => refreshPlot());
document.getElementById('gamma').addEventListener('change', updateEquations);
document.getElementById('glm').addEventListener('change', updateEquations);
updateEquations();
poll();
</script>
</body>
</html>
"""


def page_html() -> str:
    def options(values: tuple[str, ...], selected: str) -> str:
        parts = []
        for v in values:
            sel = " selected" if v == selected else ""
            parts.append(f'<option value="{html.escape(v)}"{sel}>{html.escape(v)}</option>')
        return "\n".join(parts)

    return (
        PAGE.replace("__ICS__", options(ICS, "orszag_tang"))
        .replace("__LIMS__", options(LIMITERS, "MC"))
        .replace("__BCS__", options(BCS, "Periodic"))
        .replace("__FIELDS__", options(FIELDS, "rho"))
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _bytes(self, code: int, data: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._bytes(200, page_html().encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            with _state_lock:
                payload = {
                    "busy": _state["busy"],
                    "status": _state["status"],
                    "error": _state["error"],
                    "nframes": len(_state["history"]),
                    "times": list(_state["times"]),
                    "meta": _state["meta"],
                }
            self._json(200, payload)
            return
        if parsed.path == "/plot.png":
            q = urllib.parse.parse_qs(parsed.query)
            field = q.get("field", ["rho"])[0]
            frame = int(q.get("frame", ["0"])[0])
            fixed = q.get("fixed", ["1"])[0] != "0"
            png = render_plot(field, frame, fixed)
            self._bytes(200, png, "image/png")
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/run":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            params = json.loads(raw.decode("utf-8"))
            for key in ("nx", "ny", "nsnap"):
                params[key] = int(params[key])
            for key in ("t_end", "cfl", "gamma", "glm", "eta_o", "eta_h", "eta_a"):
                params[key] = float(params[key])
            if params["nx"] < 8 or params["ny"] < 8:
                raise ValueError("nx and ny must be >= 8")
            if params["nsnap"] < 2:
                raise ValueError("Snapshots must be >= 2")
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"ok": False, "error": str(exc)})
            return

        with _state_lock:
            if _state["busy"]:
                self._json(409, {"ok": False, "error": "Already running"})
                return

        threading.Thread(target=run_simulation, args=(params,), daemon=True).start()
        self._json(200, {"ok": True})


def _open_browser(url: str) -> None:
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:
        pass


def _probe_mhd_gui(port: int) -> bool:
    """Return True if an MHD Solver web GUI is already listening on port."""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/status", timeout=0.4) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read().decode("utf-8"))
            return isinstance(data, dict) and "busy" in data and "nframes" in data
    except Exception:
        return False


def _bind_server(port: int) -> ThreadingHTTPServer:
    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((HOST, port), Handler)


def main() -> None:
    # Touch import early for clearer errors
    _ = mhd

    # Prefer reusing an already-running GUI so extra launches just open another tab.
    if _probe_mhd_gui(PORT):
        url = f"http://{HOST}:{PORT}/"
        print(f"MHD Solver GUI already running: {url}")
        print("Opening another browser window/tab (same session).")
        _open_browser(url)
        return

    httpd: ThreadingHTTPServer | None = None
    bound_port = PORT
    last_err: Exception | None = None
    for port in range(PORT, PORT + PORT_SCAN):
        if port != PORT and _probe_mhd_gui(port):
            url = f"http://{HOST}:{port}/"
            print(f"MHD Solver GUI already running: {url}")
            _open_browser(url)
            return
        try:
            httpd = _bind_server(port)
            bound_port = port
            break
        except OSError as exc:
            last_err = exc
            continue

    if httpd is None:
        raise SystemExit(
            f"Could not bind any port in {PORT}..{PORT + PORT_SCAN - 1}: {last_err}"
        )

    url = f"http://{HOST}:{bound_port}/"
    print(f"MHD Solver GUI: {url}")
    print("Open more tabs/windows at the same URL — one server is enough.")
    print("Press Ctrl+C to stop.")
    _open_browser(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
