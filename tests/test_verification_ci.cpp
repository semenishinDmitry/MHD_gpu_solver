#include <gtest/gtest.h>

#include "verification/diagnostics.hpp"
#include "verification/problems.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

#ifndef MHD_VERIFICATION_REF_DIR
#define MHD_VERIFICATION_REF_DIR "."
#endif

std::unordered_map<std::string, double> load_ref_scalars(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open reference file: " + path);
    }
    std::string tag;
    int ver = 0;
    in >> tag >> ver;
    if (tag != "MHD_REF" || ver != 1) {
        throw std::runtime_error("Bad reference header: " + path);
    }
    std::unordered_map<std::string, double> vals;
    std::string key;
    double value = 0.0;
    while (in >> key >> value) {
        vals[key] = value;
    }
    return vals;
}

void expect_near_diag(const std::string& problem, const std::string& key, double actual,
                      double expected, double tol)
{
    EXPECT_NEAR(actual, expected, tol)
        << "problem=" << problem << " diagnostic=" << key << " actual=" << actual
        << " expected=" << expected << " tol=" << tol;
}

void check_physical(const std::string& problem, const VerificationDiagnostics& d)
{
    EXPECT_TRUE(d.finite) << problem << ": non-finite values detected";
    EXPECT_GT(d.min_rho, 0.0) << problem << ": min_rho=" << d.min_rho;
    EXPECT_GT(d.min_pressure, 0.0) << problem << ": min_pressure=" << d.min_pressure;
    EXPECT_TRUE(std::isfinite(d.max_div_b)) << problem;
    EXPECT_GT(d.steps, 0) << problem;
}

} // namespace

TEST(VerificationCI, SodFast)
{
    const auto p = make_fast_sod();
    const MHDSolver solver = run_verification_problem(p);
    const auto d = compute_diagnostics(solver);
    check_physical(p.name, d);

    // Mass on [0,1] x thin strip: integral rho * dx*dy ≈ (0.5*1 + 0.5*0.125)*Ly
    const double Ly = p.y_max - p.y_min;
    const double mass0 = (0.5 * 1.0 + 0.5 * 0.125) * Ly;
    expect_near_diag(p.name, "mass", d.mass, mass0, 5e-3 * Ly);

    const auto ref = load_ref_scalars(std::string(MHD_VERIFICATION_REF_DIR) + "/sod_fast.ref");
    expect_near_diag(p.name, "min_rho", d.min_rho, ref.at("min_rho"), 0.02);
    expect_near_diag(p.name, "max_rho", d.max_rho, ref.at("max_rho"), 0.05);
}

TEST(VerificationCI, BrioWuFast)
{
    const auto p = make_fast_brio_wu();
    const MHDSolver solver = run_verification_problem(p);
    const auto d = compute_diagnostics(solver);
    check_physical(p.name, d);
    EXPECT_LT(d.max_div_b, 5.0) << p.name;

    const auto ref = load_ref_scalars(std::string(MHD_VERIFICATION_REF_DIR) + "/brio_wu_fast.ref");
    expect_near_diag(p.name, "min_rho", d.min_rho, ref.at("min_rho"), 0.03);
    expect_near_diag(p.name, "max_rho", d.max_rho, ref.at("max_rho"), 0.08);
}

TEST(VerificationCI, OrszagTangFast)
{
    const auto p = make_fast_orszag_tang();
    const MHDSolver solver = run_verification_problem(p);
    const auto d = compute_diagnostics(solver);
    check_physical(p.name, d);
    EXPECT_LT(d.max_div_b, 20.0) << p.name;

    const auto ref =
        load_ref_scalars(std::string(MHD_VERIFICATION_REF_DIR) + "/orszag_tang_fast.ref");
    expect_near_diag(p.name, "mass", d.mass, ref.at("mass"), 1e-6);
    expect_near_diag(p.name, "min_rho", d.min_rho, ref.at("min_rho"), 0.05);
}

TEST(VerificationCI, RotorFast)
{
    const auto p = make_fast_rotor();
    const MHDSolver solver = run_verification_problem(p);
    const auto d = compute_diagnostics(solver);
    check_physical(p.name, d);
}

TEST(VerificationCI, KelvinHelmholtzFast)
{
    const auto p = make_fast_kelvin_helmholtz();
    const MHDSolver solver = run_verification_problem(p);
    const auto d = compute_diagnostics(solver);
    check_physical(p.name, d);
}

TEST(VerificationCI, AlfvenFast)
{
    const auto p = make_fast_alfven();
    const MHDSolver solver = run_verification_problem(p);
    const auto d = compute_diagnostics(solver);
    check_physical(p.name, d);

    const double l2 = alfven_wave_l2_error(solver);
    EXPECT_LT(l2, 5e-2) << p.name << " Alfvén L2 error=" << l2
                        << " (smooth wave; expects reasonable phase accuracy on 32 cells)";

    const auto ref = load_ref_scalars(std::string(MHD_VERIFICATION_REF_DIR) + "/alfven_fast.ref");
    expect_near_diag(p.name, "l2_error", l2, ref.at("l2_error"), 0.02);
}

TEST(InitialCondition, SodLeftRightStates)
{
    const auto L = sod_initial_condition(0.25, 0.0);
    const auto R = sod_initial_condition(0.75, 0.0);
    EXPECT_DOUBLE_EQ(L.rho, 1.0);
    EXPECT_DOUBLE_EQ(R.rho, 0.125);
    EXPECT_DOUBLE_EQ(L.pressure, 1.0);
    EXPECT_DOUBLE_EQ(R.pressure, 0.1);
}

TEST(InitialCondition, BrioWuMagneticJump)
{
    const auto L = brio_wu_initial_condition(0.2, 0.0);
    const auto R = brio_wu_initial_condition(0.8, 0.0);
    EXPECT_DOUBLE_EQ(L.bx, 0.75);
    EXPECT_DOUBLE_EQ(R.bx, 0.75);
    EXPECT_DOUBLE_EQ(L.by, 1.0);
    EXPECT_DOUBLE_EQ(R.by, -1.0);
}

TEST(InitialCondition, AlfvenExactAtT0MatchesIC)
{
    const double x = 0.3;
    const auto W0 = alfven_wave_initial_condition(x, 0.0);
    const auto We = alfven_wave_exact(x, 0.0);
    EXPECT_NEAR(W0.by, We.by, 1e-15);
    EXPECT_NEAR(W0.vy, We.vy, 1e-15);
}
