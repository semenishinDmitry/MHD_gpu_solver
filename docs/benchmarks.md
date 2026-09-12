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

## Reporting template

When publishing timings, record: CPU model, OS, compiler+version, CMake build type,
`MHD_NATIVE_ARCH`, grid, CFL, `t_end`, steps, wall time, cells/s.
