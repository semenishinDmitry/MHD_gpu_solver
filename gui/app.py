#!/usr/bin/env python3
"""
Cross-platform MHD Solver desktop GUI (Windows / macOS / Linux).

Uses plain tkinter widgets only (no ttk, no Canvas-as-scroll) for reliable
rendering on Apple's ancient system Tk 8.5.
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PY = os.path.join(_ROOT, "build", "python")
for path in (_PY, _ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("numpy is required. pip install numpy") from exc

try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required. pip install matplotlib") from exc

try:
    import mhd_solver as mhd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "mhd_solver module not found. Build with MHD_BUILD_PYTHON=ON "
        "(./scripts/setup_and_build.sh) and set PYTHONPATH=build/python"
    ) from exc


FIELDS = ("rho", "mx", "my", "mz", "energy", "bx", "by", "bz", "psi")
ICS = ("orszag_tang", "sine", "blast", "uniform")
LIMITERS = ("MC", "Minmod")
BCS = ("Periodic", "Outflow")

# High-contrast colors that work even on broken Aqua Tk 8.5
BG = "#ffffff"
PANEL = "#f0f0f0"
FG = "#000000"
ENTRY_BG = "#ffffff"
BTN_BG = "#2060a0"
BTN_FG = "#ffffff"
ACCENT = "#003399"


def interior(arr: np.ndarray, ng: int) -> np.ndarray:
    if ng <= 0:
        return arr
    return arr[ng:-ng, ng:-ng]


class MHDGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MHD Solver")
        self.geometry("1280x820")
        self.minsize(1000, 700)
        self.configure(bg=BG)

        self._history: list[dict[str, np.ndarray]] = []
        self._times: list[float] = []
        self._playing = False
        self._play_job: str | None = None
        self._busy = False
        self._clim: dict[str, tuple[float, float]] = {}
        self.im = None
        self.cbar = None

        self._build_ui()
        self._draw_placeholder()
        self._set_status("Ready. Press Run to start.")
        self.after(50, self._force_refresh)

    def _lbl(self, parent: tk.Misc, text: str, **kw) -> tk.Label:
        opts = {"bg": PANEL, "fg": FG, "anchor": "w", "font": ("Helvetica", 11)}
        opts.update(kw)
        return tk.Label(parent, text=text, **opts)

    def _ent(self, parent: tk.Misc, default: str) -> tk.StringVar:
        var = tk.StringVar(value=default)
        e = tk.Entry(parent, textvariable=var, bg=ENTRY_BG, fg=FG, width=18)
        e.pack(fill="x", pady=(0, 4))
        return var

    def _opt(self, parent: tk.Misc, values: tuple[str, ...], default: str) -> tk.StringVar:
        var = tk.StringVar(value=default)
        om = tk.OptionMenu(parent, var, *values)
        om.configure(bg=ENTRY_BG, fg=FG, highlightthickness=0)
        om.pack(fill="x", pady=(0, 4))
        return var

    def _btn(self, parent: tk.Misc, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=BTN_BG,
            fg=BTN_FG,
            activebackground="#4080c0",
            activeforeground=BTN_FG,
            padx=8,
            pady=4,
        )

    def _row(self, parent: tk.Misc, title: str, default: str) -> tk.StringVar:
        self._lbl(parent, title).pack(anchor="w")
        return self._ent(parent, default)

    def _build_ui(self) -> None:
        # Two-pane layout via pack (no PanedWindow / no Canvas scroll — those break on Tk 8.5 macOS)
        left = tk.Frame(self, bg=PANEL, width=320, bd=1, relief="solid")
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        right = tk.Frame(self, bg=BG)
        right.pack(side="right", fill="both", expand=True)

        # Compact form: two columns inside left panel so everything fits without scrolling
        header = tk.Frame(left, bg=PANEL)
        header.pack(fill="x", padx=8, pady=8)
        self._lbl(header, "MHD Solver", font=("Helvetica", 16, "bold")).pack(anchor="w")

        form = tk.Frame(left, bg=PANEL)
        form.pack(fill="both", expand=True, padx=8, pady=0)

        col_a = tk.Frame(form, bg=PANEL)
        col_b = tk.Frame(form, bg=PANEL)
        col_a.pack(side="left", fill="both", expand=True, padx=(0, 4))
        col_b.pack(side="right", fill="both", expand=True, padx=(4, 0))

        self._lbl(col_a, "Grid / time", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 4))
        self.var_nx = self._row(col_a, "nx", "64")
        self.var_ny = self._row(col_a, "ny", "64")
        self.var_tend = self._row(col_a, "t_end", "0.05")
        self.var_nsnap = self._row(col_a, "Snapshots", "40")
        self.var_cfl = self._row(col_a, "CFL", "0.4")
        self.var_gamma = self._row(col_a, "gamma", "1.6666666667")
        self.var_glm = self._row(col_a, "GLM alpha", "0.1")

        self._lbl(col_a, "Setup", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(8, 4))
        self._lbl(col_a, "Initial condition").pack(anchor="w")
        self.var_ic = self._opt(col_a, ICS, "orszag_tang")
        self._lbl(col_a, "Limiter").pack(anchor="w")
        self.var_lim = self._opt(col_a, LIMITERS, "MC")
        self._lbl(col_a, "BC").pack(anchor="w")
        self.var_bc = self._opt(col_a, BCS, "Periodic")

        self._lbl(col_b, "Non-ideal", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 4))
        self.var_ohmic = tk.BooleanVar(value=False)
        self.var_hall = tk.BooleanVar(value=False)
        self.var_amb = tk.BooleanVar(value=False)
        tk.Checkbutton(col_b, text="Ohmic", variable=self.var_ohmic, bg=PANEL, fg=FG, anchor="w").pack(fill="x")
        self.var_eta_o = self._row(col_b, "eta_ohm", "0.001")
        tk.Checkbutton(col_b, text="Hall", variable=self.var_hall, bg=PANEL, fg=FG, anchor="w").pack(fill="x")
        self.var_eta_h = self._row(col_b, "eta_hall", "0.001")
        tk.Checkbutton(col_b, text="Ambipolar", variable=self.var_amb, bg=PANEL, fg=FG, anchor="w").pack(fill="x")
        self.var_eta_a = self._row(col_b, "eta_ambipolar", "0.001")

        self._lbl(col_b, "View", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(8, 4))
        self._lbl(col_b, "Field").pack(anchor="w")
        self.var_field = self._opt(col_b, FIELDS, "rho")
        self.var_field.trace_add("write", lambda *_: self._redraw())
        self.var_fixed_clim = tk.BooleanVar(value=True)
        tk.Checkbutton(
            col_b,
            text="Fixed color scale",
            variable=self.var_fixed_clim,
            command=self._redraw,
            bg=PANEL,
            fg=FG,
            anchor="w",
        ).pack(fill="x", pady=(0, 8))

        actions = tk.Frame(left, bg=PANEL)
        actions.pack(fill="x", padx=8, pady=8)
        self.btn_run = self._btn(actions, "Run", self._on_run)
        self.btn_run.pack(fill="x", pady=2)
        self.btn_play = self._btn(actions, "Play", self._toggle_play)
        self.btn_play.configure(state="disabled")
        self.btn_play.pack(fill="x", pady=2)

        self.status = tk.StringVar(value="")
        tk.Label(
            left,
            textvariable=self.status,
            bg=PANEL,
            fg=ACCENT,
            wraplength=290,
            justify="left",
            anchor="w",
            font=("Helvetica", 10),
        ).pack(fill="x", padx=8, pady=(0, 8))

        # Matplotlib
        self.fig = Figure(figsize=(7.5, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111)

        plot_host = tk.Frame(right, bg=BG)
        plot_host.pack(side="top", fill="both", expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_host)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg="white")
        widget.pack(side="top", fill="both", expand=True)

        toolbar_frame = tk.Frame(right, bg=BG)
        toolbar_frame.pack(side="top", fill="x")
        try:
            self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
            self.toolbar.pack(side="left", fill="x")
        except TypeError:
            self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        try:
            self.toolbar.update()
        except Exception:
            pass

        controls = tk.Frame(right, bg=BG)
        controls.pack(side="bottom", fill="x", padx=8, pady=6)
        tk.Label(controls, text="t", bg=BG, fg=FG).pack(side="left")
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_scale = tk.Scale(
            controls,
            from_=0,
            to=0,
            orient="horizontal",
            variable=self.time_var,
            showvalue=0,
            command=self._on_slider,
            state="disabled",
            bg=BG,
            fg=FG,
            highlightthickness=0,
        )
        self.time_scale.pack(side="left", fill="x", expand=True, padx=8)
        self.time_label = tk.Label(controls, text="t = —", bg=BG, fg=FG)
        self.time_label.pack(side="right")

    def _force_refresh(self) -> None:
        self.update_idletasks()
        try:
            self.canvas.draw()
        except Exception:
            pass

    def _draw_placeholder(self) -> None:
        self.ax.clear()
        self.ax.text(
            0.5,
            0.5,
            "MHD Solver\n\nSet parameters on the left,\nthen press Run.",
            ha="center",
            va="center",
            fontsize=14,
            transform=self.ax.transAxes,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("No data yet")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _set_status(self, text: str) -> None:
        self.status.set(text)

    def _parse_params(self) -> dict:
        nx = int(self.var_nx.get())
        ny = int(self.var_ny.get())
        if nx < 8 or ny < 8:
            raise ValueError("nx and ny must be >= 8")
        nsnap = int(self.var_nsnap.get())
        if nsnap < 2:
            raise ValueError("Snapshots must be >= 2")
        return {
            "nx": nx,
            "ny": ny,
            "t_end": float(self.var_tend.get()),
            "nsnap": nsnap,
            "cfl": float(self.var_cfl.get()),
            "gamma": float(self.var_gamma.get()),
            "glm": float(self.var_glm.get()),
            "ic": self.var_ic.get(),
            "limiter": self.var_lim.get(),
            "bc": self.var_bc.get(),
            "ohmic": bool(self.var_ohmic.get()),
            "hall": bool(self.var_hall.get()),
            "amb": bool(self.var_amb.get()),
            "eta_o": float(self.var_eta_o.get()),
            "eta_h": float(self.var_eta_h.get()),
            "eta_a": float(self.var_eta_a.get()),
        }

    def _on_run(self) -> None:
        if self._busy:
            return
        try:
            params = self._parse_params()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self._stop_play()
        self._busy = True
        self.btn_run.configure(state="disabled")
        self.btn_play.configure(state="disabled")
        self.time_scale.configure(state="disabled")
        self._set_status("Running…")

        def worker() -> None:
            err: Exception | None = None
            history: list[dict[str, np.ndarray]] = []
            times: list[float] = []
            meta: dict = {}
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
                    return {
                        name: interior(np.array(solver.field(name), copy=True), ng) for name in FIELDS
                    }

                history.append(snap())
                times.append(float(solver.time))
                for k in range(1, nsnap):
                    solver.advance_to(min(k * dt_snap, t_end))
                    history.append(snap())
                    times.append(float(solver.time))
                    self.after(0, lambda kk=k: self._set_status(f"Running… {kk + 1}/{nsnap}"))

                meta = {"steps": solver.steps, "c_h": solver.c_h, "divb": solver.max_div_b()}
            except Exception as exc:  # noqa: BLE001
                err = exc

            def done() -> None:
                self._busy = False
                self.btn_run.configure(state="normal")
                if err is not None:
                    self._set_status("Failed.")
                    messagebox.showerror("Solve failed", str(err))
                    return
                self._history = history
                self._times = times
                self._load_field_history(meta)

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _load_field_history(self, meta: dict) -> None:
        n = len(self._history)
        self.time_scale.configure(state="normal", from_=0, to=max(n - 1, 0))
        self.time_var.set(0)
        self.btn_play.configure(state="normal")
        self.im = None
        self.cbar = None
        self.ax.clear()
        self._recompute_clim()
        self._redraw()
        self._set_status(
            f"Done. steps={meta.get('steps')}  c_h={meta.get('c_h', 0):.4g}  "
            f"max|div B|={meta.get('divb', 0):.4g}"
        )

    def _recompute_clim(self) -> None:
        self._clim.clear()
        if not self._history:
            return
        for name in FIELDS:
            stack = np.stack([frame[name] for frame in self._history], axis=0)
            vmin = float(np.min(stack))
            vmax = float(np.max(stack))
            if vmin == vmax:
                vmax = vmin + 1e-30
            self._clim[name] = (vmin, vmax)

    def _frame_index(self) -> int:
        if not self._history:
            return 0
        return int(round(float(self.time_var.get())))

    def _on_slider(self, _value: str | None = None) -> None:
        if not self._playing:
            self._redraw()

    def _redraw(self) -> None:
        if not self._history:
            return
        idx = min(max(self._frame_index(), 0), len(self._history) - 1)
        field = self.var_field.get()
        data = self._history[idx][field]
        t = self._times[idx]

        if self.im is None:
            self.im = self.ax.imshow(
                data,
                origin="lower",
                aspect="equal",
                cmap="inferno",
                interpolation="nearest",
            )
            self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
        else:
            self.im.set_data(data)
            self.im.set_extent((-0.5, data.shape[1] - 0.5, -0.5, data.shape[0] - 0.5))

        if self.var_fixed_clim.get() and field in self._clim:
            self.im.set_clim(*self._clim[field])
        else:
            self.im.set_clim(float(np.min(data)), float(np.max(data)) + 1e-30)

        self.ax.set_title(f"{field}   t = {t:.5g}")
        self.ax.set_xlabel("i")
        self.ax.set_ylabel("j")
        self.time_label.configure(text=f"t = {t:.5g}  ({idx + 1}/{len(self._history)})")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _toggle_play(self) -> None:
        if not self._history:
            return
        if self._playing:
            self._stop_play()
        else:
            self._playing = True
            self.btn_play.configure(text="Pause")
            self._play_step()

    def _stop_play(self) -> None:
        self._playing = False
        self.btn_play.configure(text="Play")
        if self._play_job is not None:
            self.after_cancel(self._play_job)
            self._play_job = None

    def _play_step(self) -> None:
        if not self._playing or not self._history:
            return
        nxt = self._frame_index() + 1
        if nxt >= len(self._history):
            nxt = 0
        self.time_var.set(nxt)
        self._redraw()
        self._play_job = self.after(80, self._play_step)


def main() -> None:
    # Helpful warning for Apple CLT Python / Tk 8.5
    try:
        _r = tk.Tk()
        _r.withdraw()
        patch = str(_r.tk.call("info", "patchlevel"))
        _r.destroy()
        major_minor = tuple(int(x) for x in patch.split(".")[:2])
        if major_minor < (8, 6):
            print(
                f"WARNING: Tk {patch} (Apple system Tk) is buggy on modern macOS.\n"
                "If the GUI still looks blank, install Homebrew Python:\n"
                "  brew install python python-tk\n"
                "  /opt/homebrew/bin/python3 -m pip install matplotlib numpy\n"
                "  PYTHONPATH=build/python /opt/homebrew/bin/python3 gui/app.py\n",
                file=sys.stderr,
            )
    except Exception:
        pass

    app = MHDGui()
    app.mainloop()


if __name__ == "__main__":
    main()
