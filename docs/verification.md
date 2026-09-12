# Verification

## Verification vs validation

- **Verification:** does the *numerical implementation* solve the *intended equations*
  correctly? (manufactured/smooth solutions, shock-tube structure, conservation, GLM diagnostics)
- **Validation:** does the *physical model* agree with *experiments / nature*?

This suite provides **verification**. It does **not** claim experimental validation.

## Fast CI problems

Configs live in `include/verification/problems.hpp`. Scalar references:
`tests/verification/reference/*.ref` (see that directory’s README for regeneration).

| Problem | Grid (fast) | \(t_{\mathrm{end}}\) | Notes |
|---------|-------------|----------------------|-------|
| Sod | \(64\times1\) | 0.1 | Hydro; \(\gamma=1.4\); outflow–x |
| Brio–Wu | \(64\times1\) | 0.05 | MHD tube; \(\gamma=2\) |
| Orszag–Tang | \(32\times32\) | 0.05 | Periodic vortex |
| Rotor | \(32\times32\) | 0.05 | Positivity / robustness |
| Kelvin–Helmholtz | \(32\times32\) | 0.2 | Shear layer + seed |
| Alfvén wave | \(32\times1\) | 1.0 | Smooth; L2 vs exact |

Checks (all fast tests): completion, finite fields, \(\rho>0\), \(p>0\), steps \(>0\),
plus problem-specific scalar tolerances / Alfvén L2 error.

## Convergence (smooth Alfvén wave)

```bash
cmake --build build --target mhd_verify
./build/mhd_verify --convergence-alfven alfven_convergence.csv
```

Runs \(N=32,64,128,256\), reports L1/L2 errors of \((B_y,B_z,v_y,v_z)\) vs the exact
circularly polarized wave and observed orders
\(p=\log(E_N/E_{2N})/\log 2\).

**Do not** invent an order-of-accuracy claim for discontinuous Sod/Brio–Wu problems; those
are verified with structural/diagnostic checks, not formal EOCs.

## Full (manual) runs

Increase `nx,ny,t_end` in a local copy of the problem structs or via `mhd_verify --run …`
wrappers / Python API. Keep expensive grids out of mandatory CI.

## Command line

```bash
./build/mhd_verify --run sod
./build/mhd_verify --run brio_wu
./build/mhd_verify --run orszag_tang
./build/mhd_verify --run rotor
./build/mhd_verify --run kh
./build/mhd_verify --run alfven
./build/mhd_verify --dump-refs tests/verification/reference
```
