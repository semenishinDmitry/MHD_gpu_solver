# MHD Solver

[![CI](https://github.com/semenishinDmitry/MHD_gpu_solver/actions/workflows/ci.yml/badge.svg)](https://github.com/semenishinDmitry/MHD_gpu_solver/actions/workflows/ci.yml)
[![Sanitizers](https://github.com/semenishinDmitry/MHD_gpu_solver/actions/workflows/sanitizers.yml/badge.svg)](https://github.com/semenishinDmitry/MHD_gpu_solver/actions/workflows/sanitizers.yml)

2D **ideal / non-ideal GLM-MHD** finite-volume solver (MUSCL + HLL + SSP-RK2) with optional
Python bindings, a verification suite, and GitHub CI.

> This repository provides **mathematical/numerical verification** of the implementation.
> It does **not** claim experimental validation or production readiness.

## Physical model & methods

- Ideal MHD + Dedner GLM divergence cleaning; optional Ohmic / Hall / Ambipolar terms
- Uniform Cartesian FV mesh, SSP-RK2 time integration
- Details: [docs/equations.md](docs/equations.md), [docs/numerical-methods.md](docs/numerical-methods.md)

## Supported test problems

| Problem | IC name | Fast CI |
|---------|---------|---------|
| Sod shock tube | `sod` | yes (1D-in-2D) |
| Brio–Wu | `brio_wu` | yes |
| Orszag–Tang | `orszag_tang` | yes |
| MHD rotor | `rotor` | yes |
| Kelvin–Helmholtz | `kelvin_helmholtz` | yes |
| Alfvén wave | `alfven_wave` | yes (+ convergence tool) |

See [docs/verification.md](docs/verification.md).

## Build (C++20)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMHD_NATIVE_ARCH=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/mhd_verify --build-info
```

Convenience scripts: `./scripts/setup_and_build.sh` (local Release + native),
`./scripts/ci_build.sh` (CI-like portable build).

More: [docs/build.md](docs/build.md).

## Tests, verification, convergence

```bash
ctest --test-dir build --output-on-failure          # unit + regression + fast verification
./build/mhd_verify --run sod
./build/mhd_verify --convergence-alfven out.csv     # smooth-wave EOC table
```

## Benchmarks

```bash
cmake -S . -B build -DMHD_BUILD_BENCHMARKS=ON -DMHD_NATIVE_ARCH=OFF
cmake --build build -j --target mhd_bench
./build/benchmarks/mhd_bench --benchmark_filter=BM_OrszagTang_Run
```

See [docs/benchmarks.md](docs/benchmarks.md). Workflow: `.github/workflows/benchmark.yml`
(artifacts only; not a merge gate).

## Code quality

```bash
./scripts/format.sh check     # clang-format 19.1.7 (pinned)
cmake --build build --target lint   # clang-tidy (if available)
```

Sanitizers: `-DMHD_ENABLE_SANITIZERS=ON` or `.github/workflows/sanitizers.yml`.

## Python & GUI

```bash
export PYTHONPATH="$PWD/build/python:$PYTHONPATH"
python examples/python/orszag_tang.py
./scripts/run_gui.sh          # browser UI by default
```

## Documentation index

| Doc | Topic |
|-----|-------|
| [equations.md](docs/equations.md) | Conserved form, EOS, non-ideal |
| [numerical-methods.md](docs/numerical-methods.md) | FV / RK / CFL |
| [reconstruction.md](docs/reconstruction.md) | MUSCL / limiters |
| [riemann-solvers.md](docs/riemann-solvers.md) | HLL |
| [glm.md](docs/glm.md) | Divergence cleaning |
| [boundary-conditions.md](docs/boundary-conditions.md) | Periodic / outflow |
| [verification.md](docs/verification.md) | Test problems & EOC |
| [benchmarks.md](docs/benchmarks.md) | Timing methodology |
| [build.md](docs/build.md) | Toolchain & options |

## License

**No LICENSE file is present yet.** The project owner must choose a license; see `CITATION.cff`
(`license: TBD`). Do not assume open-source terms until a LICENSE is added.

## Citation

See [`CITATION.cff`](CITATION.cff).
