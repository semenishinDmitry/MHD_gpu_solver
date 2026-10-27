# CPU performance: SIMD and multithreading

This page describes how the CPU solver is organized for speed, how to control it, and what it
guarantees numerically. GPU backends are covered in [gpu.md](gpu.md); measured numbers for all
backends are in [benchmarks.md](benchmarks.md).

## Summary

| Layer | What it does | Where |
|-------|--------------|-------|
| Row-fused spatial operator | Face fluxes are computed per grid row into small per-thread buffers and differenced immediately; no full-grid face-flux arrays | `include/mhd/rhs.hpp` |
| Branchless generic kernel | MUSCL + HLL + GLM written once as a template over the number type, with `select` instead of branches | `include/mhd/kernels/muscl_hll.hpp` |
| Explicit SIMD | The same template instantiated with a vector type: NEON (ARM64), SSE2 / AVX / AVX2 / AVX-512F (x86-64) | `include/mhd/simd.hpp`, `include/mhd/kernels/cpu_rows.hpp` |
| Multithreading | Persistent thread pool with dynamic row scheduling; no OpenMP dependency | `include/mhd/parallel.hpp` |

On an Apple M5 (4 performance + 6 efficiency cores, NEON) the time per cell update dropped
from ~270–310 ns to ~26 ns (×8–12). See [benchmarks.md](benchmarks.md#cpu-results) for the full table.

## 1. Restructuring the original code (scalar)

The original operator evaluated each face independently on `MHDPrimitive` structs, stored all
x- and y-face fluxes into two full-size `StateField`s, then read them back to form the RHS.
Changes:

- **Row fusion.** For a row `j` the solver computes the `nx+1` x-face fluxes and the `nx`
  y-face fluxes above the row into per-thread scratch rows, assembles `L(U)` for row `j`, and
  reuses the upper y-face row as the lower one for `j+1`. Memory traffic per RHS drops by about
  36 doubles per cell (18 stores + 18 loads of face fluxes).
- **Rotated-frame flux.** x- and y-faces use one function in a rotated frame
  (normal `n`, tangential `t`); the y-direction just swaps `(vx, vy)` and `(bx, by)` on load
  and store. This is algebraically identical to `physical_flux_x/y`.
- **Fewer divisions.** `1/ρ` and `1/(s_R − s_L)` are computed once and multiplied.
- **Compile-time limiter.** Minmod / MC is a template parameter, so the slope loop has no
  `switch` per component.
- **Branchless HLL and positivity fallback.** The upwind / degenerate cases of HLL and the
  first-order fallback for unphysical reconstructed states are evaluated with `select`. This is
  what makes the kernel vectorizable.

The original implementation is kept as `compute_rhs_reference()` in
`include/mhd/rhs_reference.hpp`. Tests compare against it, and `mhd_backend_compare` reports it
as the "before" baseline.

## 2. SIMD

`mhd::simd::vdouble` is a thin wrapper over the native double-precision vector. The backend is
selected at **compile time** from the target flags:

| Target flags | Backend | Lanes |
|--------------|---------|-------|
| `-mavx512f` (or `-march=native` on AVX-512 CPUs) | AVX-512F | 8 |
| `-mavx` / `-mavx2`, MSVC `/arch:AVX2` | AVX / AVX2 | 4 |
| x86-64 default | SSE2 | 2 |
| AArch64 (Apple Silicon, Graviton, …) | NEON | 2 |
| anything else, or `-DMHD_SIMD_DISABLE` | scalar | 1 |

The kernel template is instantiated twice per row: with `vdouble` for the vector body and with
`double` for the remainder (`nx mod lanes` cells). Both instantiations use the same source.

Portable builds (`-DMHD_NATIVE_ARCH=OFF`) get SSE2 on x86-64 and NEON on ARM64. For AVX2 or
AVX-512 without `-march=native`, pass flags explicitly, e.g.
`-DCMAKE_CXX_FLAGS="-mavx2 -mfma"`. The backend in use is reported by
`mhd_backend_compare` and by `mhd_solver.simd_backend()` in Python.

On NEON (2 lanes) explicit SIMD gives ×1.5 over the scalar build of the same code; wider x86
vectors should give more. The kernel is limited by divisions and square roots: each face needs
2 `sqrt` per side plus 3 divisions.

## 3. Multithreading

`mhd::ThreadPool` is a persistent pool. The caller participates as thread 0, so
`set_num_threads(1)` runs everything inline.

- **Work split.** The RHS, CFL reduction and current density are split by grid rows. The RK
  updates are split by blocks of cells. Chunks are handed out dynamically, about 8 chunks per
  thread, so fast and slow cores (P/E cores on Apple and Intel) finish together.
- **Small grids** (fewer than 128×128 interior cells by default) run single-threaded; the cost
  of waking threads would outweigh the work.
- **No OpenMP.** Apple Clang ships without `libomp`, and MSVC supports only OpenMP 2.0. The pool
  uses only `<thread>` and works the same with Clang, GCC and MSVC.
- **Thread safety.** Several solvers can run from different threads (for example Python threads,
  since `run()` releases the GIL); their parallel loops take turns on the shared pool.

### Determinism

Results are **bitwise identical for any thread count**:

- every output value is computed by exactly one thread, with the same arithmetic;
- the vector body / scalar tail split depends only on the row length, never on the chunking;
- the only reduction (max signal speed for the CFL step) is a `max`, which is exact.

`CpuKernels.ThreadCountDoesNotChangeResultsBitwise` checks this (1, 3 and 8 threads, with
non-ideal terms enabled).

The **SIMD and scalar paths agree to round-off, not bitwise.** They contract `a*b + c` into FMA
differently. The tests require a relative difference below 1e-12 per RHS evaluation against the
reference, and below 1e-10 after a full solve.

## Controls

| C++ | Python | Environment | Default |
|-----|--------|-------------|---------|
| `mhd::set_num_threads(n)` | `mhd_solver.set_num_threads(n)` | `MHD_NUM_THREADS=n` | all hardware threads |
| `mhd::set_simd_enabled(b)` | `mhd_solver.set_simd_enabled(b)` | — | on |
| `mhd::set_parallel_min_cells(c)` | — | — | 16384 |
| compile with `-DMHD_SIMD_DISABLE` | — | — | off |

`n = 0` restores the default. All settings are process-wide.

## Allocation-free time loop

All buffers are allocated once: `TimeIntegratorWorkspace` and the per-thread scratch rows in
`RHSWorkspace`. The scratch grows only after `set_num_threads()` increases the thread count.
`tests/test_performance_invariants.cpp` checks this.

## Profiling tips

- macOS: `xcrun xctrace record --template 'Time Profiler' --launch -- ./build/mhd_backend_compare --sizes 512`
- Linux: `perf record -g ./build/mhd_backend_compare --sizes 512 --no-reference`
- Check vectorization: build with `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize`
  (Clang). The flux kernel uses explicit SIMD, so these remarks matter only for the simple
  loops (RK updates, RHS assembly).
