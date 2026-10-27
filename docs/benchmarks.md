# Benchmarks

Benchmarks measure **performance**, not correctness. Small timing fluctuations must not
fail pull requests.

## Methodology

- Tool: Google Benchmark target `mhd_bench` (`benchmarks/bench_mhd.cpp`)
- Build: Release, `-DMHD_NATIVE_ARCH=OFF` in CI (portable); local may enable native
- Metrics: wall time, iterations, items/s (cells processed for Orszag–Tang)

Representative cases:

| Benchmark | Problem | Notes |
|-----------|---------|-------|
| `BM_OrszagTang_Run/{32,64,128}` | Orszag–Tang to \(t=0.02\) | Primary scaling |
| `BM_Ohmic_OrszagTang64` | OT + Ohmic \(\eta=10^{-3}\) | Non-ideal cost |
| `BM_RhsOnly_OrszagTang` | Short advances | RHS-dominated |

## Reproduce

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMHD_BUILD_BENCHMARKS=ON -DMHD_NATIVE_ARCH=OFF
cmake --build build -j --target mhd_bench
./build/benchmarks/mhd_bench --benchmark_filter=BM_OrszagTang_Run
./build/mhd_verify --build-info   # record compiler / flags / commit
```

GitHub Actions workflow `.github/workflows/benchmark.yml` uploads JSON artifacts. It does
**not** gate merges on performance.

## Backend comparison tool

`mhd_backend_compare` (built by default) runs the same problems on every available backend
and writes Markdown/CSV tables:

```bash
./build/mhd_backend_compare --sizes 128,256,512,1024 --t-end 0.02 \
    --accuracy-n 256 --accuracy-t 0.5 --alfven-max 1024 --markdown compare.md --csv compare.csv
```

| Variant | Meaning |
|---------|---------|
| CPU reference (original code), 1T | pre-optimization operator `compute_rhs_reference()`, serial |
| CPU scalar, 1T | optimized row-fused kernels, SIMD disabled |
| CPU <SIMD>, 1T / NT | optimized kernels with explicit SIMD, 1 thread / all threads |
| Metal fp32 | Apple GPU |
| CUDA fp64 / fp32 | NVIDIA GPU (if built with `-DMHD_ENABLE_CUDA=ON`) |

GPU times include host↔device transfers of the state and exclude one-time device setup
(shader compilation), which runs in a warm-up call. Accuracy compares each variant with the
fp64 CPU solution of the same problem and, for the Alfvén wave, with the exact solution.

<a id="cpu-results"></a>
## Results: Apple M5 (10 cores: 4P + 6E), macOS, AppleClang 21, Release, `-march=native`

Measured 2026-10-27. CUDA was not available on this machine.

- Build: AppleClang 21.0.0.21000101, Release, native=ON, commit c151c69
- CPU SIMD backend: NEON (2 doubles/vector), hardware threads: 10
- Metal fp32: Metal: Apple M5

### Speed: Orszag-Tang, MC limiter, t_end = 0.02

Throughput in Mcell-updates/s (one update = one full SSP-RK2 step of one cell). GPU times include host<->device transfers of the state.

| Variant | 128² | 256² | 512² | 1024² |
|---|---|---|---|---|
| CPU reference (original code), 1T | 3.7 (×1.0) | 3.8 (×1.0) | 3.4 (×1.0) | 3.2 (×1.0) |
| CPU scalar, 1T | 6.3 (×1.7) | 6.3 (×1.7) | 6.1 (×1.8) | 6.1 (×1.9) |
| CPU NEON, 1T | 9.5 (×2.6) | 9.3 (×2.5) | 9.2 (×2.7) | 9.0 (×2.8) |
| CPU NEON, 10T | 29.3 (×7.9) | 33.0 (×8.7) | 37.5 (×11.0) | 38.0 (×11.7) |
| Metal fp32 | 47.9 (×12.9) | 150.2 (×39.6) | 122.0 (×35.9) | 103.5 (×31.9) |

(×k) = speed-up relative to the first row.

### Accuracy: Orszag-Tang (ideal), 256², t = 0.5

Differences from the fp64 CPU solution (relative L1 / Linf norms). Mass and energy drift are |Q(t) - Q(0)| / Q(0) (both are conserved exactly by the scheme with periodic boundaries, so the drift is pure round-off).

| Variant | steps | ρ L1 | p L1 | B L1 | ρ L∞ | mass drift | energy drift | max ∇·B |
|---|---|---|---|---|---|---|---|---|
| CPU scalar, 1T | 3452 | 1.52e-15 | 1.84e-15 | 1.26e-15 | 2.90e-15 | 2.4e-13 | 1.1e-15 | 2.70e+00 |
| CPU NEON, 1T | 3452 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.3e-13 | 8.2e-16 | 2.70e+00 |
| CPU NEON, 10T | 3452 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.3e-13 | 8.2e-16 | 2.70e+00 |
| Metal fp32 | 3452 | 6.31e-07 | 7.41e-07 | 5.16e-07 | 1.83e-06 | 8.3e-08 | 5.2e-08 | 2.70e+00 |

### Accuracy: Orszag-Tang (Ohmic η=1e-3 + Hall η_H=5e-4), 256², t = 0.5

Differences from the fp64 CPU solution (relative L1 / Linf norms). Mass and energy drift are |Q(t) - Q(0)| / Q(0) (both are conserved exactly by the scheme with periodic boundaries, so the drift is pure round-off).

| Variant | steps | ρ L1 | p L1 | B L1 | ρ L∞ | mass drift | energy drift | max ∇·B |
|---|---|---|---|---|---|---|---|---|
| CPU scalar, 1T | 3415 | 1.85e-15 | 2.54e-15 | 1.67e-15 | 6.12e-15 | 2.4e-13 | 4.4e-15 | 1.43e+00 |
| CPU NEON, 1T | 3415 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.4e-13 | 4.4e-15 | 1.43e+00 |
| CPU NEON, 10T | 3415 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.4e-13 | 4.4e-15 | 1.43e+00 |
| Metal fp32 | 3415 | 6.25e-07 | 6.13e-07 | 4.63e-07 | 1.32e-06 | 1.3e-07 | 7.3e-08 | 1.43e+00 |

### Accuracy vs exact solution: circular Alfvén wave, t = 1 (one period)

Relative L1 error of (v⊥, B⊥) and observed order between successive resolutions. Second order is expected; an fp32 backend levels off once the truncation error approaches single-precision round-off.

| Variant | N=32 | N=64 | N=128 | N=256 | N=512 | N=1024 |
|---|---|---|---|---|---|---|
| CPU NEON, 10T | 3.15e-02 | 9.44e-03 (1.74) | 2.56e-03 (1.88) | 6.57e-04 (1.96) | 1.68e-04 (1.97) | 4.26e-05 (1.98) |
| Metal fp32 | 3.22e-02 | 9.95e-03 (1.70) | 2.69e-03 (1.89) | 6.94e-04 (1.95) | 1.82e-04 (1.93) | 5.11e-05 (1.83) |

### Observations

- **CPU.** Restructuring the code gives ×1.7–1.9, NEON SIMD (2 lanes) another ×1.5, and 10
  threads another ×3.2–4.2 (4 performance cores + 6 slower efficiency cores). In total that is
  ×8–12 over the original code. Small grids scale worse because there is less work per
  thread-pool wake-up.
- **CPU results do not depend on SIMD or thread count:** they match the original
  implementation to ~1e-15, and the SIMD and multithreaded runs are bitwise identical to each
  other.
- **Metal.** ×32–40 over the original code and ×3–4.5 over the 10-thread CPU for 256² and up.
  At 128² the GPU is underused and per-batch launch overhead dominates. Throughput drops from
  256² to 1024² because the GPU path stores full-grid face fluxes (it is not row-fused) and
  becomes memory-bound. Fusing the GPU kernels is the next optimization step.
- **fp32 accuracy (Metal).** On the nonlinear Orszag–Tang problem the fp32 solution differs
  from fp64 by ~6e-7 (relative L1) and ~2e-6 (L∞) after 3452 steps, with the same step count.
  Mass and energy drift stays at 1e-7, and ∇·B is identical to three digits. On the
  small-amplitude (δ = 1e-3) Alfvén wave, fp32 converges at second order but is 2–20 % less
  accurate than fp64, and the observed order drops to 1.83 at N = 1024. See
  [gpu.md](gpu.md#accuracy) for why.

## Reporting template

When publishing timings, record: CPU model, OS, compiler+version, CMake build type,
`MHD_NATIVE_ARCH`, grid, CFL, `t_end`, steps, wall time, cells/s.
