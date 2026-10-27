#!/usr/bin/env python3
"""
Browser-based MHD Solver setup GUI (Windows / macOS / Linux).

Geometry canvas, mesh/mask preview, IC regions, domain BCs, Run + heatmap.
Uses stdlib http.server and matplotlib Agg (avoids broken system Tk).
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
for path in (_PY, _ROOT, os.path.dirname(__file__)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import mhd_solver as mhd
from geometry import GeometryScene, rasterize_scene

FIELDS = ("rho", "mx", "my", "mz", "energy", "bx", "by", "bz", "psi")
ICS = ("orszag_tang", "sine", "blast", "uniform")
LIMITERS = ("MC", "Minmod")
BC_TYPES = ("Periodic", "Outflow")

HOST = "127.0.0.1"
PORT = int(os.environ.get("MHD_GUI_PORT", "8765"))
PORT_SCAN = 32

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "busy": False,
    "status": "Ready. Configure geometry and press Run.",
    "error": "",
    "history": [],
    "times": [],
    "clim": {},
    "meta": {},
    "mask": None,
}


def interior(arr: np.ndarray, ng: int) -> np.ndarray:
    if ng <= 0:
        return arr
    return arr[ng:-ng, ng:-ng]


def _bc_enum(name: str) -> Any:
    if name == "Outflow":
        return mhd.BoundaryConditionType.Outflow
    return mhd.BoundaryConditionType.Periodic


def apply_ic_regions(
    solver: mhd.MHDSolver,
    mask: np.ndarray,
    regions: list[dict[str, Any]],
    base: dict[str, float],
) -> None:
    """Paint constant primitives onto fluid cells; solids keep benign fill."""
    ny, nx = mask.shape
    rho = np.full((ny, nx), float(base.get("rho", 1.0)))
    vx = np.full((ny, nx), float(base.get("vx", 0.0)))
    vy = np.full((ny, nx), float(base.get("vy", 0.0)))
    vz = np.full((ny, nx), float(base.get("vz", 0.0)))
    p = np.full((ny, nx), float(base.get("p", 1.0)))
    bx = np.full((ny, nx), float(base.get("bx", 0.0)))
    by = np.full((ny, nx), float(base.get("by", 0.0)))
    bz = np.full((ny, nx), float(base.get("bz", 0.0)))

    x_min, x_max = solver.x_min, solver.x_max
    y_min, y_max = solver.y_min, solver.y_max
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    xs = x_min + (np.arange(nx) + 0.5) * dx
    ys = y_min + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    for reg in regions:
        rtype = reg.get("type", "rect")
        vals = reg.get("values", {})
        if rtype == "rect":
            xa, xb = sorted((float(reg["x0"]), float(reg["x1"])))
            ya, yb = sorted((float(reg["y0"]), float(reg["y1"])))
            hit = (X >= xa) & (X <= xb) & (Y >= ya) & (Y <= yb)
        elif rtype == "circle":
            hit = (X - float(reg["cx"])) ** 2 + (Y - float(reg["cy"])) ** 2 <= float(
                reg["radius"]
            ) ** 2
        else:
            continue
        hit &= mask.astype(bool)
        if "rho" in vals:
            rho[hit] = float(vals["rho"])
        if "vx" in vals:
            vx[hit] = float(vals["vx"])
        if "vy" in vals:
            vy[hit] = float(vals["vy"])
        if "vz" in vals:
            vz[hit] = float(vals["vz"])
        if "p" in vals:
            p[hit] = float(vals["p"])
        if "bx" in vals:
            bx[hit] = float(vals["bx"])
        if "by" in vals:
            by[hit] = float(vals["by"])
        if "bz" in vals:
            bz[hit] = float(vals["bz"])

    solver.set_primitives(rho, vx, vy, vz, p, bx, by, bz)


def run_simulation(params: dict[str, Any]) -> None:
    with _state_lock:
        _state["busy"] = True
        _state["error"] = ""
        _state["status"] = "Running…"
        _state["history"] = []
        _state["times"] = []
        _state["clim"] = {}
        _state["meta"] = {}
        _state["mask"] = None

    try:
        nx = int(params["nx"])
        ny = int(params["ny"])
        x_min = float(params.get("x_min", 0.0))
        x_max = float(params.get("x_max", 1.0))
        y_min = float(params.get("y_min", 0.0))
        y_max = float(params.get("y_max", 1.0))

        solver = mhd.MHDSolver(nx, ny, x_min, x_max, y_min, y_max)
        solver.set_cfl(float(params["cfl"]))
        solver.set_gamma(float(params["gamma"]))
        solver.set_glm_alpha(float(params["glm"]))
        if params.get("limiter") == "Minmod":
            solver.set_limiter_minmod()
        else:
            solver.set_limiter_mc()

        bc_x = _bc_enum(params.get("bc_x", params.get("bc", "Periodic")))
        bc_y = _bc_enum(params.get("bc_y", params.get("bc", "Periodic")))
        solver.set_bc(bc_x, bc_y)
        solver.set_walls_reflective(bool(params.get("walls_reflective", True)))

        solver.set_ideal()
        if params.get("ohmic"):
            solver.enable_ohmic(float(params["eta_o"]))
        if params.get("hall"):
            solver.enable_hall(float(params["eta_h"]))
        if params.get("amb"):
            solver.enable_ambipolar(float(params["eta_a"]))

        # Mask from shapes or explicit array
        if params.get("mask") is not None:
            mask = np.asarray(params["mask"], dtype=np.uint8)
        else:
            scene = params.get("scene") or {"shapes": params.get("shapes", [])}
            mask = rasterize_scene(scene, nx, ny, x_min, x_max, y_min, y_max)
        if mask.shape != (ny, nx):
            raise ValueError(f"mask shape {mask.shape} != ({ny}, {nx})")
        solver.set_mask(mask)

        ic_mode = params.get("ic_mode", "preset")
        if ic_mode == "regions":
            apply_ic_regions(
                solver,
                mask,
                list(params.get("ic_regions") or []),
                dict(params.get("ic_base") or {"rho": 1.0, "p": 1.0}),
            )
        else:
            solver.initialize(params.get("ic", "orszag_tang"))

        ng = solver.ng
        nsnap = int(params["nsnap"])
        t_end = float(params["t_end"])
        dt_snap = t_end / (nsnap - 1)

        def snap() -> dict[str, np.ndarray]:
            out = {name: interior(np.array(solver.field(name), copy=True), ng) for name in FIELDS}
            # Mask solids as NaN for plotting
            solid = mask == 0
            for name in FIELDS:
                arr = out[name].astype(float, copy=True)
                arr[solid] = np.nan
                out[name] = arr
            return out

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
            finite = stack[np.isfinite(stack)]
            if finite.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
                if vmin == vmax:
                    vmax = vmin + 1e-30
            clim[name] = (vmin, vmax)

        meta = {
            "steps": int(solver.steps),
            "c_h": float(solver.c_h),
            "divb": float(solver.max_div_b()),
            "n_solid": int(np.sum(mask == 0)),
        }
        with _state_lock:
            _state["history"] = history
            _state["times"] = times
            _state["clim"] = clim
            _state["meta"] = meta
            _state["mask"] = mask
            _state["status"] = (
                f"Done. steps={meta['steps']}  c_h={meta['c_h']:.4g}  "
                f"max|div B|={meta['divb']:.4g}  solids={meta['n_solid']}"
            )
    except Exception as exc:  # noqa: BLE001
        with _state_lock:
            _state["error"] = str(exc)
            _state["status"] = "Failed."
            _state["history"] = []
            _state["times"] = []
            _state["mask"] = None
        traceback.print_exc()
    finally:
        with _state_lock:
            _state["busy"] = False


def render_mask_preview(params: dict[str, Any]) -> bytes:
    nx = int(params["nx"])
    ny = int(params["ny"])
    x_min = float(params.get("x_min", 0.0))
    x_max = float(params.get("x_max", 1.0))
    y_min = float(params.get("y_min", 0.0))
    y_max = float(params.get("y_max", 1.0))
    if params.get("mask") is not None:
        mask = np.asarray(params["mask"], dtype=np.uint8)
    else:
        scene = params.get("scene") or {"shapes": params.get("shapes", [])}
        mask = rasterize_scene(scene, nx, ny, x_min, x_max, y_min, y_max)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=100)
    cmap = ListedColormap(["#6b7280", "#93c5fd"])  # solid gray, fluid blue
    ax.imshow(
        mask,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest",
    )
    ax.set_title(f"Mask  fluid={int(np.sum(mask))}  solid={int(np.sum(mask == 0))}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue(), mask


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
        cmap = plt.get_cmap("inferno").copy()
        cmap.set_bad(color="#9ca3af")
        im = ax.imshow(data, origin="lower", aspect="equal", cmap=cmap, interpolation="nearest")
        if fixed_clim and field in clim:
            im.set_clim(*clim[field])
        else:
            finite = data[np.isfinite(data)]
            if finite.size:
                im.set_clim(float(np.min(finite)), float(np.max(finite)) + 1e-30)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{field}   t = {t:.5g}")
        ax.set_xlabel("i")
        ax.set_ylabel("j")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MHD Solver Setup</title>
<style>
  :root { --bg:#eef1f5; --panel:#fff; --fg:#1a1a1a; --muted:#5a6270; --accent:#1a5fb4;
          --border:#c8ced8; --solid:#6b7280; --fluid:#93c5fd; }
  * { box-sizing: border-box; }
  body { margin:0; font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         color:var(--fg); background:var(--bg); }
  .wrap { display:grid; grid-template-columns: 340px 1fr 360px; min-height:100vh; }
  aside, .right { background:var(--panel); border-color:var(--border); padding:14px; overflow:auto; }
  aside { border-right:1px solid var(--border); }
  .right { border-left:1px solid var(--border); }
  main { padding:14px; display:flex; flex-direction:column; gap:10px; min-width:0; }
  h1 { font-size:18px; margin:0 0 10px; }
  h2 { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:12px 0 6px; }
  label { display:block; font-size:11px; color:var(--muted); margin:5px 0 2px; }
  input[type=text], input[type=number], select {
    width:100%; padding:6px 8px; border:1px solid var(--border); border-radius:5px; background:#fff; }
  .row2 { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
  .row4 { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:4px; }
  .chk { display:flex; align-items:center; gap:6px; margin:4px 0; font-size:12px; }
  button {
    width:100%; margin-top:6px; padding:9px 10px; border:0; border-radius:6px;
    background:var(--accent); color:#fff; font-weight:600; cursor:pointer; }
  button.secondary { background:#5b616b; }
  button.tool { width:auto; margin:2px; padding:6px 10px; font-size:12px; }
  button.tool.active { outline:2px solid #111; }
  button:disabled { opacity:.55; cursor:default; }
  #status { margin-top:8px; color:var(--accent); min-height:2.2em; white-space:pre-wrap; font-size:12px; }
  #err { color:#b00020; white-space:pre-wrap; font-size:12px; }
  #plot, #maskPrev { width:100%; background:#fff; border:1px solid var(--border); border-radius:6px; }
  #geoCanvas {
    width:100%; height:360px; background:#fafbfc; border:1px solid var(--border); border-radius:6px;
    cursor:crosshair; touch-action:none; }
  .tools { display:flex; flex-wrap:wrap; gap:2px; margin-bottom:6px; }
  .hint { font-size:11px; color:var(--muted); margin:4px 0 8px; }
  .eq-box { background:var(--panel); border:1px solid var(--border); border-radius:6px; padding:8px 12px; }
  .eq-box details summary { cursor:pointer; font-weight:600; color:var(--muted); }
  .toolbar { display:flex; align-items:center; gap:8px; }
  .toolbar input[type=range] { flex:1; }
  .region-list { max-height:120px; overflow:auto; font-size:11px; border:1px solid var(--border);
                 border-radius:4px; padding:4px; background:#fafbfc; }
  @media (max-width: 1100px) {
    .wrap { grid-template-columns: 1fr; }
    aside, .right { border:0; border-bottom:1px solid var(--border); max-height:none; }
  }
</style>
<script>
MathJax = {
  tex: { inlineMath: [['\\(','\\)']], displayMath: [['\\[','\\]']] },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre'] }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
<div class="wrap">
  <aside>
    <h1>MHD Solver</h1>
    <h2>Domain / Mesh</h2>
    <div class="row2">
      <div><label>x_min</label><input id="x_min" type="text" value="0"/></div>
      <div><label>x_max</label><input id="x_max" type="text" value="1"/></div>
    </div>
    <div class="row2">
      <div><label>y_min</label><input id="y_min" type="text" value="0"/></div>
      <div><label>y_max</label><input id="y_max" type="text" value="1"/></div>
    </div>
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
    <label>Limiter</label><select id="limiter">__LIMS__</select>

    <h2>Boundaries</h2>
    <div class="row2">
      <div><label>Left/Right (x)</label><select id="bc_x">__BCX__</select></div>
      <div><label>Bottom/Top (y)</label><select id="bc_y">__BCY__</select></div>
    </div>
    <label class="chk"><input id="walls_reflective" type="checkbox" checked/> Interior solids = reflective walls</label>
    <p class="hint">Domain-edge Reflective BC is not implemented; use solid mask for walls.</p>

    <h2>Initial conditions</h2>
    <label>Mode</label>
    <select id="ic_mode" onchange="toggleIcMode()">
      <option value="preset">Named preset</option>
      <option value="regions">Regional constants</option>
    </select>
    <div id="icPreset">
      <label>Preset</label><select id="ic">__ICS__</select>
    </div>
    <div id="icRegions" style="display:none">
      <div class="row2">
        <div><label>base ρ</label><input id="base_rho" type="text" value="1"/></div>
        <div><label>base p</label><input id="base_p" type="text" value="1"/></div>
      </div>
      <div class="row2">
        <div><label>base vx</label><input id="base_vx" type="text" value="0"/></div>
        <div><label>base vy</label><input id="base_vy" type="text" value="0"/></div>
      </div>
      <div class="row2">
        <div><label>base Bx</label><input id="base_bx" type="text" value="0"/></div>
        <div><label>base By</label><input id="base_by" type="text" value="0"/></div>
      </div>
      <p class="hint">Draw IC region on canvas with tool “IC rect”, then set values and Add.</p>
      <div class="row2">
        <div><label>ρ</label><input id="reg_rho" type="text" value="1"/></div>
        <div><label>p</label><input id="reg_p" type="text" value="1"/></div>
      </div>
      <div class="row2">
        <div><label>vx</label><input id="reg_vx" type="text" value="1"/></div>
        <div><label>vy</label><input id="reg_vy" type="text" value="0"/></div>
      </div>
      <button class="secondary" type="button" onclick="addIcRegion()">Add last IC rect</button>
      <div class="region-list" id="icList">(none)</div>
      <button class="secondary" type="button" onclick="clearIcRegions()">Clear IC regions</button>
    </div>

    <h2>Non-ideal</h2>
    <label class="chk"><input id="ohmic" type="checkbox" onchange="updateEquations()"/> Ohmic</label>
    <label>eta_ohm</label><input id="eta_o" type="text" value="0.001"/>
    <label class="chk"><input id="hall" type="checkbox" onchange="updateEquations()"/> Hall</label>
    <label>eta_hall</label><input id="eta_h" type="text" value="0.001"/>
    <label class="chk"><input id="amb" type="checkbox" onchange="updateEquations()"/> Ambipolar</label>
    <label>eta_ambipolar</label><input id="eta_a" type="text" value="0.001"/>

    <button id="runBtn" onclick="runSim()">Run</button>
    <button id="playBtn" class="secondary" onclick="togglePlay()" disabled>Play</button>
    <div id="status"></div>
    <div id="err"></div>
  </aside>

  <main>
    <h2 style="margin:0;text-transform:none;letter-spacing:0;font-size:14px;color:var(--fg)">Geometry</h2>
    <div class="tools">
      <button type="button" class="tool active" id="tool_select" onclick="setTool('select')">Select</button>
      <button type="button" class="tool" id="tool_rect" onclick="setTool('rect')">Rect</button>
      <button type="button" class="tool" id="tool_circle" onclick="setTool('circle')">Circle</button>
      <button type="button" class="tool" id="tool_poly" onclick="setTool('poly')">Polygon</button>
      <button type="button" class="tool" id="tool_ic" onclick="setTool('ic')">IC rect</button>
    </div>
    <div class="tools">
      <label class="chk" style="margin:0 8px 0 0"><input type="radio" name="paint" value="solid" checked onchange="paintMode='solid'"/> Solid</label>
      <label class="chk" style="margin:0"><input type="radio" name="paint" value="fluid" onchange="paintMode='fluid'"/> Fluid</label>
      <button type="button" class="tool secondary" onclick="undoShape()">Undo</button>
      <button type="button" class="tool secondary" onclick="clearShapes()">Clear</button>
      <button type="button" class="tool secondary" onclick="previewMask()">Preview mask</button>
    </div>
    <p class="hint" id="geoHint">Drag to draw solid/fluid shapes. Polygon: click vertices, double-click or Enter to close.</p>
    <canvas id="geoCanvas" width="640" height="360"></canvas>

    <div class="eq-box">
      <details>
        <summary>Equations</summary>
        <div id="equations" class="mj" style="margin-top:8px"></div>
        <div class="hint" id="eqNote"></div>
      </details>
    </div>

    <h2 style="margin:0;text-transform:none;letter-spacing:0;font-size:14px;color:var(--fg)">Solution</h2>
    <div class="row2" style="max-width:420px">
      <div><label>Field</label><select id="field">__FIELDS__</select></div>
      <div><label class="chk" style="margin-top:18px"><input id="fixed_clim" type="checkbox" checked/> Fixed color scale</label></div>
    </div>
    <img id="plot" alt="field plot" src="/plot.png?field=rho&frame=0&fixed=1"/>
    <div class="toolbar">
      <span>t</span>
      <input id="frame" type="range" min="0" max="0" value="0" oninput="onFrame()"/>
      <span id="tlabel">t = —</span>
    </div>
  </main>

  <div class="right">
    <h1 style="font-size:16px">Mesh / Mask</h1>
    <p class="hint">Staircase immersed boundary on uniform Cartesian cells. Gray = solid.</p>
    <img id="maskPrev" alt="mask preview" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"/>
    <button class="secondary" type="button" onclick="previewMask()">Refresh preview</button>
    <h2>Scene JSON</h2>
    <textarea id="sceneJson" rows="12" style="width:100%;font:11px/1.3 ui-monospace,monospace"></textarea>
    <button class="secondary" type="button" onclick="loadSceneJson()">Load JSON</button>
  </div>
</div>
<script>
let playing = false, playTimer = null, nFrames = 0, times = [];
let tool = 'select', paintMode = 'solid';
let shapes = [];
let draft = null; // in-progress shape in domain coords
let polyPts = [];
let icRegions = [];
let lastIcRect = null;
let dragging = false;

const canvas = document.getElementById('geoCanvas');
const ctx = canvas.getContext('2d');

function domain() {
  return {
    x_min: +document.getElementById('x_min').value,
    x_max: +document.getElementById('x_max').value,
    y_min: +document.getElementById('y_min').value,
    y_max: +document.getElementById('y_max').value,
    nx: +document.getElementById('nx').value,
    ny: +document.getElementById('ny').value,
  };
}

function toCanvas(x, y) {
  const d = domain();
  const w = canvas.width, h = canvas.height;
  const px = (x - d.x_min) / (d.x_max - d.x_min) * w;
  const py = (1 - (y - d.y_min) / (d.y_max - d.y_min)) * h;
  return [px, py];
}
function toDomain(px, py) {
  const d = domain();
  const w = canvas.width, h = canvas.height;
  const x = d.x_min + (px / w) * (d.x_max - d.x_min);
  const y = d.y_min + (1 - py / h) * (d.y_max - d.y_min);
  return [x, y];
}
function canvasPos(ev) {
  const r = canvas.getBoundingClientRect();
  const sx = canvas.width / r.width, sy = canvas.height / r.height;
  return [(ev.clientX - r.left) * sx, (ev.clientY - r.top) * sy];
}

function setTool(t) {
  tool = t;
  ['select','rect','circle','poly','ic'].forEach(name => {
    const el = document.getElementById('tool_' + (name === 'ic' ? 'ic' : name));
    if (el) el.classList.toggle('active', (name === tool) || (name === 'ic' && tool === 'ic'));
  });
  polyPts = [];
  draft = null;
  document.getElementById('geoHint').textContent =
    tool === 'poly' ? 'Click vertices; double-click or Enter to close polygon.'
    : tool === 'ic' ? 'Drag a rectangle for an IC paint region.'
    : tool === 'select' ? 'Select tool — use Undo/Clear to edit shapes.'
    : 'Drag to draw. Toggle Solid/Fluid paint mode above.';
  drawGeo();
}

function toggleIcMode() {
  const m = document.getElementById('ic_mode').value;
  document.getElementById('icPreset').style.display = m === 'preset' ? '' : 'none';
  document.getElementById('icRegions').style.display = m === 'regions' ? '' : 'none';
}

function drawShape(s, stroke) {
  ctx.strokeStyle = stroke || (s.mode === 'solid' ? '#374151' : '#1d4ed8');
  ctx.fillStyle = s.mode === 'solid' ? 'rgba(107,114,128,0.35)' : 'rgba(147,197,253,0.35)';
  ctx.lineWidth = 2;
  if (s.kind === 'rectangle') {
    const [x0,y0] = toCanvas(s.x0, s.y0);
    const [x1,y1] = toCanvas(s.x1, s.y1);
    const x = Math.min(x0,x1), y = Math.min(y0,y1), w = Math.abs(x1-x0), h = Math.abs(y1-y0);
    ctx.fillRect(x,y,w,h); ctx.strokeRect(x,y,w,h);
  } else if (s.kind === 'circle') {
    const [cx,cy] = toCanvas(s.cx, s.cy);
    const [rx,] = toCanvas(s.cx + s.radius, s.cy);
    const r = Math.abs(rx - cx);
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2); ctx.fill(); ctx.stroke();
  } else if (s.kind === 'polygon' && s.points && s.points.length) {
    ctx.beginPath();
    s.points.forEach((p,i) => {
      const [px,py] = toCanvas(p[0], p[1]);
      if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
    });
    ctx.closePath(); ctx.fill(); ctx.stroke();
  }
}

function drawGeo() {
  const d = domain();
  ctx.clearRect(0,0,canvas.width,canvas.height);
  // grid
  ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 1;
  const gx = Math.min(d.nx, 32), gy = Math.min(d.ny, 32);
  for (let i=0;i<=gx;i++) {
    const x = i/gx * canvas.width;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,canvas.height); ctx.stroke();
  }
  for (let j=0;j<=gy;j++) {
    const y = j/gy * canvas.height;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(canvas.width,y); ctx.stroke();
  }
  shapes.forEach(s => drawShape(s));
  if (draft) drawShape(draft, '#111');
  if (polyPts.length) {
    ctx.strokeStyle = '#111'; ctx.fillStyle = 'rgba(0,0,0,0.15)';
    ctx.beginPath();
    polyPts.forEach((p,i) => {
      const [px,py] = toCanvas(p[0], p[1]);
      if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
    });
    ctx.stroke();
    polyPts.forEach(p => {
      const [px,py] = toCanvas(p[0], p[1]);
      ctx.fillRect(px-3, py-3, 6, 6);
    });
  }
  document.getElementById('sceneJson').value = JSON.stringify({shapes}, null, 2);
}

function undoShape() { shapes.pop(); drawGeo(); }
function clearShapes() { shapes = []; polyPts = []; draft = null; drawGeo(); }
function loadSceneJson() {
  try {
    const data = JSON.parse(document.getElementById('sceneJson').value);
    shapes = data.shapes || [];
    drawGeo();
  } catch (e) {
    document.getElementById('err').textContent = 'Invalid scene JSON: ' + e;
  }
}

canvas.addEventListener('mousedown', (ev) => {
  const [px,py] = canvasPos(ev);
  const [x,y] = toDomain(px, py);
  if (tool === 'poly') {
    polyPts.push([x,y]);
    drawGeo();
    return;
  }
  if (tool === 'select') return;
  dragging = true;
  draft = { kind: tool === 'ic' ? 'rectangle' : tool, mode: paintMode, x0:x, y0:y, x1:x, y1:y, cx:x, cy:y, radius:0 };
});
canvas.addEventListener('mousemove', (ev) => {
  if (!dragging || !draft) return;
  const [px,py] = canvasPos(ev);
  const [x,y] = toDomain(px, py);
  if (draft.kind === 'rectangle') { draft.x1 = x; draft.y1 = y; }
  else if (draft.kind === 'circle') {
    draft.radius = Math.hypot(x - draft.cx, y - draft.cy);
  }
  drawGeo();
});
function finishDrag() {
  if (!dragging || !draft) { dragging = false; return; }
  dragging = false;
  if (tool === 'ic') {
    lastIcRect = { type:'rect', x0:draft.x0, y0:draft.y0, x1:draft.x1, y1:draft.y1 };
    draft = null; drawGeo();
    document.getElementById('geoHint').textContent = 'IC rect ready — set values and click Add last IC rect.';
    return;
  }
  if (draft.kind === 'circle' && draft.radius < 1e-12) { draft = null; drawGeo(); return; }
  if (draft.kind === 'rectangle' && Math.abs(draft.x1-draft.x0)<1e-12 && Math.abs(draft.y1-draft.y0)<1e-12) {
    draft = null; drawGeo(); return;
  }
  shapes.push(Object.assign({}, draft, { kind: draft.kind === 'rect' ? 'rectangle' : draft.kind }));
  // normalize kind
  const s = shapes[shapes.length-1];
  if (s.kind === 'rect') s.kind = 'rectangle';
  draft = null;
  drawGeo();
}
canvas.addEventListener('mouseup', finishDrag);
canvas.addEventListener('mouseleave', () => { if (dragging) finishDrag(); });
canvas.addEventListener('dblclick', (ev) => {
  if (tool !== 'poly' || polyPts.length < 3) return;
  shapes.push({ kind:'polygon', mode: paintMode, points: polyPts.slice() });
  polyPts = [];
  drawGeo();
  ev.preventDefault();
});
window.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && tool === 'poly' && polyPts.length >= 3) {
    shapes.push({ kind:'polygon', mode: paintMode, points: polyPts.slice() });
    polyPts = [];
    drawGeo();
  }
});

['x_min','x_max','y_min','y_max','nx','ny'].forEach(id => {
  document.getElementById(id).addEventListener('change', drawGeo);
});

function addIcRegion() {
  if (!lastIcRect) {
    document.getElementById('err').textContent = 'Draw an IC rect first.';
    return;
  }
  const values = {
    rho: +document.getElementById('reg_rho').value,
    p: +document.getElementById('reg_p').value,
    vx: +document.getElementById('reg_vx').value,
    vy: +document.getElementById('reg_vy').value,
  };
  icRegions.push(Object.assign({}, lastIcRect, { values }));
  renderIcList();
}
function clearIcRegions() { icRegions = []; renderIcList(); }
function renderIcList() {
  const el = document.getElementById('icList');
  if (!icRegions.length) { el.textContent = '(none)'; return; }
  el.textContent = icRegions.map((r,i) =>
    `${i+1}: rect [${r.x0.toFixed(2)},${r.y0.toFixed(2)}]→[${r.x1.toFixed(2)},${r.y1.toFixed(2)}] ρ=${r.values.rho}`
  ).join('\n');
}

function opts() {
  const d = domain();
  return {
    ...d,
    t_end: +document.getElementById('t_end').value,
    nsnap: +document.getElementById('nsnap').value,
    cfl: +document.getElementById('cfl').value,
    gamma: +document.getElementById('gamma').value,
    glm: +document.getElementById('glm').value,
    ic: document.getElementById('ic').value,
    ic_mode: document.getElementById('ic_mode').value,
    ic_base: {
      rho: +document.getElementById('base_rho').value,
      p: +document.getElementById('base_p').value,
      vx: +document.getElementById('base_vx').value,
      vy: +document.getElementById('base_vy').value,
      bx: +document.getElementById('base_bx').value,
      by: +document.getElementById('base_by').value,
    },
    ic_regions: icRegions,
    limiter: document.getElementById('limiter').value,
    bc_x: document.getElementById('bc_x').value,
    bc_y: document.getElementById('bc_y').value,
    walls_reflective: document.getElementById('walls_reflective').checked,
    scene: { shapes },
    ohmic: document.getElementById('ohmic').checked,
    hall: document.getElementById('hall').checked,
    amb: document.getElementById('amb').checked,
    eta_o: +document.getElementById('eta_o').value,
    eta_h: +document.getElementById('eta_h').value,
    eta_a: +document.getElementById('eta_a').value,
  };
}

async function previewMask() {
  const r = await fetch('/api/rasterize', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(opts())
  });
  const j = await r.json();
  if (!j.ok) {
    document.getElementById('err').textContent = j.error || 'rasterize failed';
    return;
  }
  document.getElementById('maskPrev').src = '/mask.png?_=' + Date.now();
  document.getElementById('err').textContent = '';
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
  await previewMask();
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
  else { playing = true; document.getElementById('playBtn').textContent = 'Pause'; playStep(); }
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

function formatGammaAlpha(gammaStr, alphaStr) {
  const g = parseFloat(gammaStr);
  const a = parseFloat(alphaStr);
  const gTex = (Number.isFinite(g) && Math.abs(g - 5/3) < 1e-6) ? '5/3 \\approx 1.6667'
    : (Number.isFinite(g) ? g.toPrecision(6) : gammaStr);
  const aTex = Number.isFinite(a) ? a.toPrecision(6) : alphaStr;
  return { gTex, aTex };
}
function updateEquations() {
  const ohmic = document.getElementById('ohmic').checked;
  const hall = document.getElementById('hall').checked;
  const amb = document.getElementById('amb').checked;
  const ni = ohmic || hall || amb;
  const { gTex, aTex } = formatGammaAlpha(
    document.getElementById('gamma').value,
    document.getElementById('glm').value
  );
  let eDef = '', energyNi = '', inductionNi = '';
  if (ni) {
    const terms = [];
    if (ohmic) terms.push('\\eta_{\\mathrm{ohm}}\\,\\mathbf{J}');
    if (hall) terms.push('\\eta_{\\mathrm{Hall}}\\,\\mathbf{J}\\times\\hat{\\mathbf{B}}');
    if (amb) terms.push('\\eta_{\\mathrm{A}}\\,(\\mathbf{J}\\times\\mathbf{B})\\times\\mathbf{B}/B^2');
    eDef = '\\[\\mathbf{E}_{\\mathrm{ni}} = ' + terms.join(' + ') + ',\\quad \\mathbf{J}=\\nabla\\times\\mathbf{B}.\\]';
    energyNi = ' + \\mathbf{B}\\times\\mathbf{E}_{\\mathrm{ni}}';
    inductionNi = ' - \\nabla\\times\\mathbf{E}_{\\mathrm{ni}}';
  }
  const parts = [];
  if (ohmic) parts.push('Resistive');
  if (hall) parts.push('Hall');
  if (amb) parts.push('Ambipolar');
  const noteFinal = ni
    ? ('Non-Ideal GLM-MHD (' + parts.join(' / ') + ') — 2D FV, MUSCL + HLL, SSP-RK2.')
    : 'Ideal GLM-MHD (2D FV, MUSCL + HLL, SSP-RK2). Staircase IB for solids.';
  const html = `
\\[\\partial_t\\rho + \\nabla\\cdot(\\rho\\mathbf{v}) = 0\\]
\\[\\partial_t(\\rho\\mathbf{v}) + \\nabla\\cdot\\!\\big(\\rho\\mathbf{v}\\mathbf{v} + (p + B^2/2)\\,\\mathbf{I} - \\mathbf{B}\\mathbf{B}\\big) = 0\\]
\\[\\partial_t E + \\nabla\\cdot\\!\\big((E + p + B^2/2)\\mathbf{v} - (\\mathbf{v}\\cdot\\mathbf{B})\\mathbf{B} + \\psi\\mathbf{B}${energyNi}\\big) = 0\\]
\\[\\partial_t\\mathbf{B} + \\nabla\\cdot(\\mathbf{v}\\mathbf{B} - \\mathbf{B}\\mathbf{v} + \\psi\\,\\mathbf{I})${inductionNi} = 0\\]
\\[\\partial_t\\psi + c_h^2\\,\\nabla\\cdot\\mathbf{B} = -\\frac{\\alpha\\, c_h}{h}\\,\\psi\\]
${eDef}
\\[E = \\frac{p}{\\gamma-1} + \\tfrac12\\rho|\\mathbf{v}|^2 + \\tfrac12|\\mathbf{B}|^2,\\quad \\gamma = ${gTex}, \\quad \\alpha = ${aTex}\\]
`;
  const el = document.getElementById('equations');
  el.innerHTML = html;
  document.getElementById('eqNote').textContent = noteFinal;
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetClear([el]);
    MathJax.typesetPromise([el]).catch(() => {});
  }
}

document.getElementById('field').addEventListener('change', () => refreshPlot());
document.getElementById('fixed_clim').addEventListener('change', () => refreshPlot());
document.getElementById('gamma').addEventListener('change', updateEquations);
document.getElementById('glm').addEventListener('change', updateEquations);
drawGeo();
updateEquations();
previewMask();
poll();
</script>
</body>
</html>
"""

_last_mask_png: bytes = b""


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
        .replace("__BCX__", options(BC_TYPES, "Periodic"))
        .replace("__BCY__", options(BC_TYPES, "Periodic"))
        .replace("__FIELDS__", options(FIELDS, "rho"))
    )


def _parse_run_params(params: dict[str, Any]) -> dict[str, Any]:
    for key in ("nx", "ny", "nsnap"):
        params[key] = int(params[key])
    for key in (
        "t_end",
        "cfl",
        "gamma",
        "glm",
        "eta_o",
        "eta_h",
        "eta_a",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
    ):
        if key in params:
            params[key] = float(params[key])
    params.setdefault("x_min", 0.0)
    params.setdefault("x_max", 1.0)
    params.setdefault("y_min", 0.0)
    params.setdefault("y_max", 1.0)
    params.setdefault("eta_o", 0.001)
    params.setdefault("eta_h", 0.001)
    params.setdefault("eta_a", 0.001)
    if params["nx"] < 8 or params["ny"] < 8:
        raise ValueError("nx and ny must be >= 8")
    if params["nsnap"] < 2:
        raise ValueError("Snapshots must be >= 2")
    if params["x_max"] <= params["x_min"] or params["y_max"] <= params["y_min"]:
        raise ValueError("Domain bounds must satisfy max > min")
    return params


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
        global _last_mask_png
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
        if parsed.path == "/mask.png":
            self._bytes(200, _last_mask_png or b"", "image/png")
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        global _last_mask_png
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            params = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"ok": False, "error": str(exc)})
            return

        if parsed.path == "/api/rasterize":
            try:
                params = _parse_run_params({**params, "nsnap": params.get("nsnap", 2), "t_end": params.get("t_end", 0.01),
                                            "cfl": params.get("cfl", 0.4), "gamma": params.get("gamma", 5/3),
                                            "glm": params.get("glm", 0.1)})
                png, mask = render_mask_preview(params)
                _last_mask_png = png
                self._json(
                    200,
                    {
                        "ok": True,
                        "nx": int(params["nx"]),
                        "ny": int(params["ny"]),
                        "n_fluid": int(np.sum(mask == 1)),
                        "n_solid": int(np.sum(mask == 0)),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self._json(400, {"ok": False, "error": str(exc)})
            return

        if parsed.path != "/api/run":
            self._json(404, {"error": "not found"})
            return

        try:
            params = _parse_run_params(params)
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
    _ = mhd

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
