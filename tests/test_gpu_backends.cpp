#include <gtest/gtest.h>

#include "mhd/gpu/device_solver.hpp"
#include "mhd/solver_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// GPU backends vs the fp64 CPU solver. Skipped when no device / backend is available.

namespace {

struct BackendCase {
    mhd::Backend backend;
    mhd::Precision precision;
    double tol; // relative Linf tolerance on rho after a short run
};

std::vector<BackendCase> available_backends()
{
    std::vector<BackendCase> out;
    if (mhd::metal_available()) {
        out.push_back({mhd::Backend::Metal, mhd::Precision::FP32, 2e-5});
    }
    if (mhd::cuda_available()) {
        out.push_back({mhd::Backend::CUDA, mhd::Precision::FP64, 1e-11});
        out.push_back({mhd::Backend::CUDA, mhd::Precision::FP32, 2e-5});
    }
    return out;
}

double rel_linf(const std::vector<double>& a, const std::vector<double>& b)
{
    double d = 0.0;
    double s = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        d = std::max(d, std::abs(a[k] - b[k]));
        s = std::max(s, std::abs(b[k]));
    }
    return s > 0.0 ? d / s : d;
}

template <class Setup>
void compare_with_cpu(int nx, int ny, double t_end, Setup setup)
{
    const auto backends = available_backends();
    if (backends.empty()) {
        GTEST_SKIP() << "no GPU backend available";
    }
    MHDSolver cpu(nx, ny, 0.0, 1.0, 0.0, ny == 1 ? 1.0 / nx : 1.0);
    setup(cpu);
    const SolveResult rc = cpu.run(t_end);

    for (const auto& b : backends) {
        MHDSolver gpu(nx, ny, 0.0, 1.0, 0.0, ny == 1 ? 1.0 / nx : 1.0);
        setup(gpu);
        gpu.set_backend(b.backend, b.precision);
        const SolveResult rg = gpu.run(t_end);
        const std::string label = gpu.backend_device();
        EXPECT_EQ(rg.steps, rc.steps) << label;
        EXPECT_NEAR(rg.t, rc.t, 1e-6 * std::max(1.0, t_end)) << label;
        EXPECT_LT(rel_linf(gpu.copy_field("rho"), cpu.copy_field("rho")), b.tol) << label;
        EXPECT_LT(rel_linf(gpu.copy_field("energy"), cpu.copy_field("energy")), b.tol) << label;
        EXPECT_LT(rel_linf(gpu.copy_field("by"), cpu.copy_field("by")), 10 * b.tol) << label;
    }
}

} // namespace

TEST(GpuBackends, StubsReportUnavailableBackendsConsistently)
{
    EXPECT_TRUE(mhd::backend_available(mhd::Backend::CPU));
    if (!mhd::metal_available()) {
        EXPECT_THROW(mhd::make_metal_solver(Grid2D(8, 8, 0.0, 1.0, 0.0, 1.0, 2)),
                     std::runtime_error);
        MHDSolver s(8, 8);
        EXPECT_THROW(s.set_backend("metal"), std::runtime_error);
    }
    EXPECT_THROW(mhd::parse_backend("tpu"), std::invalid_argument);
}

TEST(GpuBackends, OrszagTangMatchesCpu)
{
    compare_with_cpu(48, 40, 0.05, [](MHDSolver& s) {
        s.set_limiter_mc();
        s.initialize("orszag_tang");
    });
}

TEST(GpuBackends, MinmodOutflowSodMatchesCpu)
{
    compare_with_cpu(64, 1, 0.05, [](MHDSolver& s) {
        s.set_limiter_minmod();
        s.set_bc(BoundaryConditionType::Outflow, BoundaryConditionType::Periodic);
        s.initialize("sod");
    });
}

TEST(GpuBackends, NonIdealMatchesCpu)
{
    compare_with_cpu(32, 32, 0.03, [](MHDSolver& s) {
        s.initialize("orszag_tang");
        s.enable_ohmic(2e-3);
        s.enable_hall(1e-3);
        s.enable_ambipolar(5e-4);
    });
}

TEST(GpuBackends, SolidMaskMatchesCpu)
{
    compare_with_cpu(40, 40, 0.03, [](MHDSolver& s) {
        s.initialize("orszag_tang");
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(40 * 40), 1);
        for (int j = 15; j < 25; ++j) {
            for (int i = 12; i < 20; ++i) {
                mask[static_cast<std::size_t>(j * 40 + i)] = 0;
            }
        }
        s.set_mask(mask.data(), mask.size());
    });
}

TEST(GpuBackends, RepeatedRunsAccumulateTime)
{
    const auto backends = available_backends();
    if (backends.empty()) {
        GTEST_SKIP() << "no GPU backend available";
    }
    for (const auto& b : backends) {
        MHDSolver s(32, 32);
        s.initialize("orszag_tang");
        s.set_backend(b.backend, b.precision);
        s.run(0.01);
        const SolveResult r = s.advance_to(0.02);
        EXPECT_NEAR(r.t, 0.02, 1e-12);
        EXPECT_GT(r.steps, 0);
        EXPECT_THROW(s.run(-1.0), std::invalid_argument);
    }
}

TEST(GpuBackends, MaxStepsIsEnforced)
{
    const auto backends = available_backends();
    if (backends.empty()) {
        GTEST_SKIP() << "no GPU backend available";
    }
    for (const auto& b : backends) {
        MHDSolver s(32, 32);
        s.initialize("orszag_tang");
        s.params().max_steps = 3;
        s.set_backend(b.backend, b.precision);
        EXPECT_THROW(s.run(1.0), std::runtime_error);
    }
}
