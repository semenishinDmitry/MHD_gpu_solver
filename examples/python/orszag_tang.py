"""Minimal Orszag-Tang example using the pybind module."""

from __future__ import annotations

import os
import sys

# Allow running from repo root without installing the wheel.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "python"))

import mhd_solver as mhd


def main() -> None:
    solver = mhd.MHDSolver(nx=64, ny=64)
    solver.set_cfl(0.4)
    solver.set_glm_alpha(0.1)
    solver.set_limiter_mc()
    # Ideal MHD by default. Uncomment for non-ideal:
    # solver.enable_ohmic(1e-3)

    solver.initialize("orszag_tang")
    result = solver.run(0.05)

    print(f"t={result.t:.4f} steps={result.steps} c_h={result.c_h:.4f}")
    print(f"max|div B|={solver.max_div_b():.6e}")
    rho = solver.field("rho")
    print(f"rho shape={rho.shape} center={rho[rho.shape[0]//2, rho.shape[1]//2]:.6f}")


if __name__ == "__main__":
    main()
