# GPU backends: CUDA and Metal

Both backends run the same scheme as the CPU solver: MUSCL (minmod / MC) + HLL + Dedner GLM,
SSP-RK2, CFL time step, periodic / outflow boundaries, Ohmic / Hall / ambipolar terms, and
staircase solid masks with reflective walls. Speed and accuracy against the CPU are measured in
[benchmarks.md](benchmarks.md).

| | CUDA | Metal |
|---|---|---|
| Hardware | NVIDIA GPUs (sm_70+) | Apple GPUs (M1 and later) |
| Precision | fp64 **or** fp32 | **fp32 only** (Apple GPUs have no fp64) |
| Kernel source | shared C++ template `include/mhd/kernels/muscl_hll.hpp` | MSL port `src/metal/mhd_kernels.metal` |
| Host code | `src/cuda/cuda_solver.cu` | `src/metal/metal_solver.mm` (Objective-C++) |
| CMake option | `-DMHD_ENABLE_CUDA=ON` (default OFF) | `-DMHD_ENABLE_METAL=ON` (default ON on macOS) |
| Toolchain | CUDA toolkit ≥ 12 (C++20 host code) | Xcode Command Line Tools only |

## Usage

C++:

```cpp
MHDSolver s(1024, 1024);
s.initialize("orszag_tang");
s.set_backend(mhd::Backend::Metal);                        // or Backend::CUDA
// s.set_backend(mhd::Backend::CUDA, mhd::Precision::FP32);
s.run(0.5);              // upload -> solve on device -> download
double divb = s.max_div_b();   // host-side diagnostics keep working
```

Python:

```python
import mhd_solver as mhd
print(mhd.available_backends())        # ['cpu', 'metal'] on a Mac
s = mhd.MHDSolver(1024, 1024)
s.initialize("orszag_tang")
s.set_backend("metal")                 # "cuda", precision="fp64" | "fp32"
s.run(0.5)                             # releases the GIL while running
rho = s.field("rho")
```

`set_backend()` throws if the backend was not compiled in or no device is present.
`mhd::metal_available()` / `mhd::cuda_available()` (`mhd_solver.metal_available()` …) check
this beforehand.

## Build

```bash
# macOS: Metal is on by default
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Linux / Windows with CUDA
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMHD_ENABLE_CUDA=ON \
      -DMHD_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build build -j
```

Metal shaders are embedded into the binary at configure time and compiled when the first
Metal solver is created (about 0.1–0.3 s once per process). No offline `metal` compiler or full
Xcode install is needed. Shaders are compiled in IEEE-safe math mode (`MTLMathModeSafe`); fast
math is off so the comparison with fp64 is meaningful.

## Design

**The state stays on the device.** A `DeviceSolver` keeps U, U*, L(U), primitives, x/y face
fluxes and J in device memory (SoA, one array per component, same layout as the CPU
`StateField`). `MHDSolver::run()` uploads the state, solves, and downloads it, so diagnostics,
`field()`, masks and initial conditions work as before. For long runs, call `run()` with a
large `t_end` rather than many short calls: each call transfers the state twice.

**The time step is computed on the device.** One SSP-RK2 step is about 20 kernel launches:

```
BC(U) [+ wall fill] → CFL partial max (per block) → CFL finalize (sets dt, c_h)
→ primitives → [J = ∇×B] → x-face fluxes → y-face fluxes → assemble L(U) → U* = U + dt L
→ BC(U*) [+ wall fill] → primitives → [J] → fluxes → L(U*) → U = ½(U + U*) + ½ dt L
→ advance t
```

`dt` and `c_h` go into a small control block in device memory, and the kernels read them from
there. The host never waits for the CFL reduction. It encodes **batches** of steps (up to 256)
and synchronizes only after each batch. The batch size follows the expected number of remaining
steps. After `t_end` is reached the extra steps in a batch run with `dt = 0`, which leaves the
state exactly unchanged (`½(U + U) = U`).

**Time accumulation.** CUDA keeps `t` in fp64 even in fp32 mode. Metal has no fp64, so `t` and
`t_end` are stored as compensated (hi + lo) float pairs, updated with an exact two-sum. The
final time matches `t_end` to about 1e-12 rather than 1e-7.

**Shared kernel (CUDA).** `muscl_hll_face<Limiter>()` and `cell_max_signal_speed()` are the
same templates the CPU SIMD path uses, compiled with `__host__ __device__`. A fix to the scheme
applies to CPU and CUDA at once. The Metal shader is a port of the same file (MSL cannot include
host C++ headers), checked by `tests/test_gpu_backends.cpp`.

## Accuracy

Measured with `mhd_backend_compare` (full tables in [benchmarks.md](benchmarks.md)):

- **CPU variants** (scalar / SIMD / any thread count) agree with the reference to ~1e-15
  (round-off). Results do not depend on the thread count.
- **fp32 on Orszag–Tang** (256², t = 0.5, 3452 steps). The relative L1 difference from fp64 is
  ~6e-7 for ρ, p and B, and L∞ is ~2e-6. The step count is the same, and mass and energy drift
  stays at ~1e-7 (fp32 round-off). With Ohmic + Hall terms the numbers are the same.
- **fp32 on small-amplitude waves.** The circular Alfvén wave has amplitude δ = 1e-3 on a
  background with E ≈ 0.65 and |B| = 1. Pressure is recovered as p = (γ−1)(E − ½ρv² − ½B²),
  which cancels the background, so the perturbation keeps only ~4 significant digits in fp32.
  The fp32 error is still second order but 2–20 % above fp64, and the gap grows with resolution
  (observed order 1.83 instead of 1.98 at N = 1024). **Use fp64 (CPU or CUDA fp64) for linear-wave or convergence studies with
  relative amplitudes ≲ 1e-3**; fp32 is fine for nonlinear problems.
- ∇·B is controlled by the same GLM cleaning on all backends; max |∇·B| differs only at
  round-off level.

## Tests

`tests/test_gpu_backends.cpp` compares every available backend with the fp64 CPU solver after a
short run: Orszag–Tang (MC), Sod with outflow (minmod), all three non-ideal terms, and a solid
obstacle mask. It checks equal step counts, matching final time, relative L∞ of ρ, E and B_y
(fp32: 2e-5, CUDA fp64: 1e-11), repeated `run()` calls, and `max_steps`. These tests are
**skipped** on machines without a device, so CI stays green on hosted runners. The CUDA CI job
(`cuda-build` in `.github/workflows/ci.yml`) compiles the CUDA backend with `nvcc` but cannot
run it.

## CI gate

`mhd_backend_compare --gate` checks every available GPU backend against the fp64 CPU solver and
exits non-zero if a check fails:

| Check | fp32 threshold | fp64 threshold |
|-------|----------------|----------------|
| Orszag–Tang 128², t = 0.25 (ideal and Ohmic + Hall): same number of time steps | equal | equal |
| relative L1 of ρ, p, B vs fp64 CPU | ≤ 1e-5 | ≤ 1e-11 |
| relative L∞ of ρ vs fp64 CPU | ≤ 5e-5 | ≤ 1e-9 |
| mass and energy drift | ≤ 2e-6 | ≤ 1e-11 |
| Alfvén wave: observed order N = 128 → 256 | ≥ 1.8 | ≥ 1.8 |
| Alfvén wave N = 256: error / CPU error (vs exact solution) | ≤ 1.25 | ≤ 1.001 |
| speed: GPU / best CPU (all threads + SIMD), 512², median of 3 | ≥ `--min-speedup` (1.5) | same |

On the Apple M5 the measured values are ~5e-7, ~7e-7, ~5e-8, 1.95, 1.06 and ×3.5. The fp32
thresholds leave about 10× headroom for other GPUs and drivers.

```bash
./build/mhd_backend_compare --gate                       # accuracy + speed
./build/mhd_backend_compare --gate --no-speed            # accuracy only
./build/mhd_backend_compare --gate --require-gpu --min-speedup 2 --markdown gate.md
```

Exit codes: 0 = passed (or no GPU and `--require-gpu` not given), 1 = a check failed,
3 = `--require-gpu` but no GPU.

Where the gate runs:

- **ctest**: `GpuGate.Accuracy` (label `gpu`) runs the accuracy part. It passes as skipped on
  machines without a GPU.
- **`.github/workflows/gpu.yml`**:
  - `metal-hosted` runs the accuracy gate on a GitHub-hosted macOS runner. If the VM exposes
    no Metal device, the job reports "skipped" and passes. Timings in a VM say nothing about
    real GPU performance, so the speed check is off there.
  - `metal-self-hosted` / `cuda-self-hosted` run the full gate (accuracy + speed, GPU
    required) on your own machines. Register a runner with labels `[self-hosted, macOS, ARM64]`
    or `[self-hosted, Linux, X64, gpu]`, then set the repository variable
    `MHD_METAL_RUNNER=true` / `MHD_CUDA_RUNNER=true`. Optional `MHD_GPU_MIN_SPEEDUP` overrides
    the 1.5× threshold.
  - Each job writes the result table to the GitHub job summary.

GitHub-hosted runners have no NVIDIA GPU. Without a self-hosted runner, CUDA is only
**compiled** in CI (`cuda-build` in `ci.yml`), never run.

## Limitations

- Reflective domain boundaries are not implemented on any backend (CPU included).
- `max_div_b()` and the other diagnostics run on the host after download.
- No multi-GPU or MPI decomposition.
- The Metal backend uses shared (unified) memory buffers. On Apple Silicon this is the fastest
  option for frequent host access; discrete GPUs in Intel Macs work, but transfers go over PCIe.
