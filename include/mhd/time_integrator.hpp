#pragma once

#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "mhd/cfl.hpp"
#include "mhd/field_ops.hpp"
#include "mhd/mhd_types.hpp"
#include "mhd/rhs.hpp"
#include "state/state_field.hpp"

#include <algorithm>
#include <stdexcept>

// Reusable scratch storage. Allocate once outside the time loop.
struct TimeIntegratorWorkspace {
    StateField U_star;
    StateField rhs;
    RHSWorkspace rhs_work;

    TimeIntegratorWorkspace() = default;

    TimeIntegratorWorkspace(int nx_tot, int ny_tot)
        : U_star(nx_tot, ny_tot), rhs(nx_tot, ny_tot), rhs_work(nx_tot, ny_tot)
    {}

    explicit TimeIntegratorWorkspace(const Grid2D& grid)
        : TimeIntegratorWorkspace(grid.get_size_x(), grid.get_size_y())
    {}
};

inline void ssp_rk2_step(StateField& U,
                         TimeIntegratorWorkspace& work,
                         const Grid2D& grid,
                         const BoundaryConditions& bc,
                         double gamma,
                         double dt,
                         double c_h,
                         double glm_alpha,
                         SlopeLimiter limiter)
{
    if (dt <= 0.0) {
        throw std::invalid_argument("ssp_rk2_step requires positive dt");
    }

    apply_boundary_conditions(U, grid, bc);
    compute_rhs(U, work.rhs, grid, work.rhs_work, gamma, c_h, glm_alpha, limiter);
    field_xpay(work.U_star, U, dt, work.rhs);

    apply_boundary_conditions(work.U_star, grid, bc);
    compute_rhs(work.U_star, work.rhs, grid, work.rhs_work, gamma, c_h, glm_alpha, limiter);
    field_ssp_rk2_combine(U, work.U_star, work.rhs, dt);
}

struct SolveParams {
    double t_end = 0.0;
    double cfl = 0.4;
    double gamma = 5.0 / 3.0;
    double glm_alpha = 0.1;
    SlopeLimiter limiter = SlopeLimiter::MC;
    int max_steps = 1'000'000;
};

struct SolveResult {
    double t = 0.0;
    int steps = 0;
    double c_h = 0.0;
};

inline SolveResult solve(StateField& U,
                         TimeIntegratorWorkspace& work,
                         const Grid2D& grid,
                         const BoundaryConditions& bc,
                         const SolveParams& params)
{
    if (params.t_end < 0.0) {
        throw std::invalid_argument("t_end must be non-negative");
    }
    if (work.U_star.nx != U.nx || work.U_star.ny != U.ny || work.rhs.nx != U.nx ||
        work.rhs.ny != U.ny) {
        throw std::invalid_argument("TimeIntegratorWorkspace size does not match StateField");
    }
    if (grid.ng < 2) {
        throw std::invalid_argument("solve requires grid.ng >= 2 for MUSCL");
    }

    SolveResult result{};
    while (result.t < params.t_end && result.steps < params.max_steps) {
        apply_boundary_conditions(U, grid, bc);
        const CFLResult cfl = compute_cfl_dt(U, grid, params.gamma, params.cfl);
        result.c_h = cfl.c_h;

        double dt = std::min(cfl.dt, params.t_end - result.t);
        ssp_rk2_step(U, work, grid, bc, params.gamma, dt, cfl.c_h, params.glm_alpha, params.limiter);

        result.t += dt;
        ++result.steps;
    }

    if (result.t < params.t_end && result.steps >= params.max_steps) {
        throw std::runtime_error("solve: reached max_steps before t_end");
    }

    apply_boundary_conditions(U, grid, bc);
    return result;
}
