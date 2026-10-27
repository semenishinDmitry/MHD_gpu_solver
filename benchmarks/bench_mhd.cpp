#include <benchmark/benchmark.h>

#include "mhd/gpu/device_solver.hpp"
#include "mhd/parallel.hpp"
#include "mhd/solver_api.hpp"

#include <string>

namespace {

void configure_orszag(MHDSolver& solver)
{
    solver.set_cfl(0.4);
    solver.set_gamma(5.0 / 3.0);
    solver.set_glm_alpha(0.1);
    solver.set_limiter_mc();
    solver.set_periodic_bc();
    solver.set_ideal();
}

void BM_OrszagTang_Run(benchmark::State& state)
{
    const int n = static_cast<int>(state.range(0));
    for (auto _ : state) {
        state.PauseTiming();
        MHDSolver solver(n, n);
        configure_orszag(solver);
        solver.initialize("orszag_tang");
        state.ResumeTiming();

        const SolveResult r = solver.run(0.02);
        int steps = r.steps;
        benchmark::DoNotOptimize(steps);
        benchmark::DoNotOptimize(solver.state().rho.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) * n);
}
BENCHMARK(BM_OrszagTang_Run)->Arg(32)->Arg(64)->Arg(128)->Unit(benchmark::kMillisecond);

void BM_RhsOnly_OrszagTang(benchmark::State& state)
{
    const int n = static_cast<int>(state.range(0));
    MHDSolver solver(n, n);
    configure_orszag(solver);
    solver.initialize("orszag_tang");

    // Warm one step so BC/ghosts are consistent, then time ssp_rk2_step via advance.
    for (auto _ : state) {
        // Re-init each iter so work stays comparable (short advance).
        state.PauseTiming();
        solver.initialize("orszag_tang");
        state.ResumeTiming();
        solver.advance_to(solver.time() + 1e-3);
        benchmark::DoNotOptimize(solver.state().rho.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_RhsOnly_OrszagTang)->Arg(64)->Arg(128)->Unit(benchmark::kMillisecond);

void BM_Ohmic_OrszagTang64(benchmark::State& state)
{
    for (auto _ : state) {
        state.PauseTiming();
        MHDSolver solver(64, 64);
        configure_orszag(solver);
        solver.enable_ohmic(1e-3);
        solver.initialize("orszag_tang");
        state.ResumeTiming();
        const SolveResult r = solver.run(0.02);
        int steps = r.steps;
        benchmark::DoNotOptimize(steps);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Ohmic_OrszagTang64)->Unit(benchmark::kMillisecond);

// CPU execution matrix: args = {n, threads (0 = all cores), simd (0/1)}.
void BM_OrszagTang_Cpu(benchmark::State& state)
{
    const int n = static_cast<int>(state.range(0));
    mhd::set_num_threads(static_cast<int>(state.range(1)));
    mhd::set_simd_enabled(state.range(2) != 0);
    for (auto _ : state) {
        state.PauseTiming();
        MHDSolver solver(n, n);
        configure_orszag(solver);
        solver.initialize("orszag_tang");
        state.ResumeTiming();
        const SolveResult r = solver.run(0.01);
        state.counters["steps"] = r.steps;
        state.counters["Mcell_updates/s"] = benchmark::Counter(
            static_cast<double>(n) * n * r.steps * 1e-6, benchmark::Counter::kIsRate);
        benchmark::ClobberMemory();
    }
    mhd::set_num_threads(0);
    mhd::set_simd_enabled(true);
}
BENCHMARK(BM_OrszagTang_Cpu)
    ->ArgNames({"n", "threads", "simd"})
    ->Args({256, 1, 0})
    ->Args({256, 1, 1})
    ->Args({256, 0, 1})
    ->Args({512, 1, 1})
    ->Args({512, 0, 1})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// GPU backends (skipped when unavailable). Includes host<->device transfers per run().
void BM_OrszagTang_Metal(benchmark::State& state)
{
    if (!mhd::metal_available()) {
        state.SkipWithError("Metal backend not available");
        return;
    }
    const int n = static_cast<int>(state.range(0));
    for (auto _ : state) {
        state.PauseTiming();
        MHDSolver solver(n, n);
        configure_orszag(solver);
        solver.initialize("orszag_tang");
        solver.set_backend(mhd::Backend::Metal);
        solver.backend_device(); // create device + compile shaders outside the timed region
        state.ResumeTiming();
        const SolveResult r = solver.run(0.01);
        state.counters["Mcell_updates/s"] = benchmark::Counter(
            static_cast<double>(n) * n * r.steps * 1e-6, benchmark::Counter::kIsRate);
    }
}
BENCHMARK(BM_OrszagTang_Metal)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

} // namespace

BENCHMARK_MAIN();
