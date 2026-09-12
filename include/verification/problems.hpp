#pragma once

#include "mhd/solver_api.hpp"
#include "verification/diagnostics.hpp"

#include <stdexcept>
#include <string>

// Compact problem descriptors for verification (fast CI vs full runs).
struct VerificationProblem {
    std::string name;
    std::string ic;
    int nx = 32;
    int ny = 32;
    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 1.0;
    double t_end = 0.05;
    double cfl = 0.4;
    double gamma = 5.0 / 3.0;
    double glm_alpha = 0.1;
    bool use_minmod = false;
    BoundaryConditionType bc_x = BoundaryConditionType::Periodic;
    BoundaryConditionType bc_y = BoundaryConditionType::Periodic;
};

inline VerificationProblem make_fast_sod()
{
    VerificationProblem p;
    p.name = "sod_fast";
    p.ic = "sod";
    p.nx = 64;
    p.ny = 1;
    p.y_max = 1.0 / 64.0; // thin strip ≈ one cell in physical y
    p.t_end = 0.1;
    p.cfl = 0.4;
    p.gamma = 1.4;
    p.use_minmod = true;
    p.bc_x = BoundaryConditionType::Outflow;
    p.bc_y = BoundaryConditionType::Periodic;
    return p;
}

inline VerificationProblem make_fast_brio_wu()
{
    VerificationProblem p;
    p.name = "brio_wu_fast";
    p.ic = "brio_wu";
    p.nx = 64;
    p.ny = 1;
    p.y_max = 1.0 / 64.0;
    p.t_end = 0.05;
    p.cfl = 0.4;
    p.gamma = 2.0;
    p.use_minmod = true;
    p.bc_x = BoundaryConditionType::Outflow;
    p.bc_y = BoundaryConditionType::Periodic;
    return p;
}

inline VerificationProblem make_fast_orszag_tang()
{
    VerificationProblem p;
    p.name = "orszag_tang_fast";
    p.ic = "orszag_tang";
    p.nx = 32;
    p.ny = 32;
    p.t_end = 0.05;
    p.cfl = 0.4;
    p.gamma = 5.0 / 3.0;
    return p;
}

inline VerificationProblem make_fast_rotor()
{
    VerificationProblem p;
    p.name = "rotor_fast";
    p.ic = "rotor";
    p.nx = 32;
    p.ny = 32;
    p.t_end = 0.05;
    p.cfl = 0.3;
    p.use_minmod = true;
    return p;
}

inline VerificationProblem make_fast_kelvin_helmholtz()
{
    VerificationProblem p;
    p.name = "kelvin_helmholtz_fast";
    p.ic = "kelvin_helmholtz";
    p.nx = 32;
    p.ny = 32;
    p.t_end = 0.2;
    p.cfl = 0.4;
    return p;
}

inline VerificationProblem make_fast_alfven()
{
    VerificationProblem p;
    p.name = "alfven_fast";
    p.ic = "alfven_wave";
    p.nx = 32;
    p.ny = 1;
    p.y_max = 1.0 / 32.0;
    p.t_end = 1.0; // one Alfvén crossing time (v_A = 1 on unit domain)
    p.cfl = 0.4;
    p.gamma = 5.0 / 3.0;
    p.bc_x = BoundaryConditionType::Periodic;
    p.bc_y = BoundaryConditionType::Periodic;
    return p;
}

inline MHDSolver run_verification_problem(const VerificationProblem& p)
{
    MHDSolver solver(p.nx, p.ny, p.x_min, p.x_max, p.y_min, p.y_max);
    solver.set_cfl(p.cfl);
    solver.set_gamma(p.gamma);
    solver.set_glm_alpha(p.glm_alpha);
    solver.set_bc(p.bc_x, p.bc_y);
    if (p.use_minmod) {
        solver.set_limiter_minmod();
    } else {
        solver.set_limiter_mc();
    }
    solver.set_ideal();
    solver.initialize(p.ic);
    solver.run(p.t_end);
    return solver;
}
