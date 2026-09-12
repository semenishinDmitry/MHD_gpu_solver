#include <benchmark/benchmark.h>

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

} // namespace

BENCHMARK_MAIN();
