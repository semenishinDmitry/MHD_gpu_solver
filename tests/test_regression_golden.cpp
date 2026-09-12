#include <gtest/gtest.h>

#include "boundary_conditions/boundary_conditions.hpp"
#include "golden_io.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/solver_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <numbers>
#include <string>
#include <vector>

namespace {

#ifndef MHD_GOLDEN_DIR
#define MHD_GOLDEN_DIR "."
#endif

std::string golden_path(const std::string& name)
{
    return (std::filesystem::path(MHD_GOLDEN_DIR) / name).string();
}

bool regen_requested()
{
    const char* env = std::getenv("MHD_REGEN_GOLDENS");
    return env != nullptr && env[0] != '\0' && std::string(env) != "0";
}

std::vector<double> interior_field(const MHDSolver& solver, const std::string& name)
{
    const auto full = solver.copy_field(name);
    const int nx_tot = solver.size_x();
    const int ny_tot = solver.size_y();
    const int ng = solver.ng();
    const int nx = solver.nx();
    const int ny = solver.ny();
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(nx * ny));
    for (int j = ng; j < ny_tot - ng; ++j) {
        for (int i = ng; i < nx_tot - ng; ++i) {
            out.push_back(full[static_cast<std::size_t>(j * nx_tot + i)]);
        }
    }
    return out;
}

GoldenSnapshot run_orszag_tang_32()
{
    MHDSolver solver(32, 32);
    solver.set_cfl(0.4);
    solver.set_gamma(5.0 / 3.0);
    solver.set_glm_alpha(0.1);
    solver.set_limiter_mc();
    solver.set_periodic_bc();
    solver.set_ideal();
    solver.initialize("orszag_tang");
    const SolveResult r = solver.run(0.05);

    GoldenSnapshot g;
    g.meta["name"] = "orszag_tang_32";
    g.meta["nx"] = "32";
    g.meta["ny"] = "32";
    g.meta["ng"] = "2";
    g.meta["t_end"] = "0.05";
    g.meta["cfl"] = "0.4";
    g.meta["gamma"] = "1.6666666666666667";
    g.meta["glm_alpha"] = "0.1";
    g.meta["limiter"] = "MC";
    g.meta["steps"] = std::to_string(r.steps);
    g.meta["max_div_b"] = std::to_string(solver.max_div_b());
    g.fields["rho"] = interior_field(solver, "rho");
    g.fields["bx"] = interior_field(solver, "bx");
    g.fields["by"] = interior_field(solver, "by");
    g.fields["energy"] = interior_field(solver, "energy");
    return g;
}

GoldenSnapshot run_ohmic_sine_bz_32()
{
    // Pure Ohmic decay of Bz = sin(2πx); ideal MHD leaves this unchanged.
    MHDSolver solver(32, 32);
    solver.set_cfl(0.3);
    solver.set_gamma(5.0 / 3.0);
    solver.set_glm_alpha(0.1);
    solver.set_limiter_minmod();
    solver.set_periodic_bc();
    solver.enable_ohmic(0.1);

    auto& U = solver.state();
    const auto& grid = solver.grid();
    for (int j = grid.j_begin(); j < grid.j_end(); ++j) {
        for (int i = grid.i_begin(); i < grid.i_end(); ++i) {
            MHDPrimitive W{};
            W.rho = 1.0;
            W.pressure = 1.0;
            W.bz = std::sin(2.0 * std::numbers::pi * grid.get_x(i));
            U.set_state(i, j, to_conservative(W, 5.0 / 3.0));
        }
    }
    apply_boundary_conditions(U, grid, solver.boundary_conditions());
    const SolveResult r = solver.run(0.1);

    GoldenSnapshot g;
    g.meta["name"] = "ohmic_sine_bz_32";
    g.meta["nx"] = "32";
    g.meta["ny"] = "32";
    g.meta["ng"] = "2";
    g.meta["t_end"] = "0.1";
    g.meta["cfl"] = "0.3";
    g.meta["eta_ohm"] = "0.1";
    g.meta["steps"] = std::to_string(r.steps);
    g.meta["max_abs_bz"] = std::to_string([&] {
        double m = 0.0;
        for (double v : interior_field(solver, "bz")) {
            m = std::max(m, std::abs(v));
        }
        return m;
    }());
    g.fields["bz"] = interior_field(solver, "bz");
    g.fields["energy"] = interior_field(solver, "energy");
    return g;
}

void compare_golden(const GoldenSnapshot& got, const GoldenSnapshot& ref, double rtol, double atol)
{
    for (const auto& kv : ref.fields) {
        ASSERT_TRUE(got.fields.count(kv.first)) << "missing field " << kv.first;
        const auto& a = got.fields.at(kv.first);
        const auto& b = kv.second;
        ASSERT_EQ(a.size(), b.size()) << kv.first;
        const double rl2 = rel_l2_error(a, b);
        const double linf = max_abs_error(a, b);
        EXPECT_LE(rl2, rtol) << kv.first << " rel_L2=" << rl2;
        EXPECT_LE(linf, std::max(atol, rtol * 10.0)) << kv.first << " Linf=" << linf;
    }
}

} // namespace

TEST(Regression, OrszagTang32MatchesGolden)
{
    const std::string path = golden_path("orszag_tang_32.golden");
    const GoldenSnapshot got = run_orszag_tang_32();
    if (regen_requested()) {
        write_golden(path, got);
        SUCCEED() << "regenerated " << path;
        return;
    }
    const GoldenSnapshot ref = read_golden(path);
    EXPECT_EQ(got.meta.at("steps"), ref.meta.at("steps"));
    compare_golden(got, ref, /*rtol=*/1e-8, /*atol=*/1e-10);
}

TEST(Regression, OhmicSineBz32MatchesGolden)
{
    const std::string path = golden_path("ohmic_sine_bz_32.golden");
    const GoldenSnapshot got = run_ohmic_sine_bz_32();
    if (regen_requested()) {
        write_golden(path, got);
        SUCCEED() << "regenerated " << path;
        return;
    }
    const GoldenSnapshot ref = read_golden(path);
    EXPECT_EQ(got.meta.at("steps"), ref.meta.at("steps"));
    // Ohmic heating/energy can vary slightly across compilers with fast-math style flags.
    compare_golden(got, ref, /*rtol=*/5e-8, /*atol=*/1e-9);
}
