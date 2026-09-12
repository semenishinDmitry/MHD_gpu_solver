#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "initial_condition/inititial_condition.hpp"
#include "mhd/time_integrator.hpp"
#include "physics_config/mhd_config.hpp"
#include "state/state_field.hpp"

#include <iostream>

int main()
{
    constexpr int nx = 64;
    constexpr int ny = 64;
    constexpr double t_end = 0.05;
    constexpr double cfl = 0.4;

    MHDConfig config;
    // Ideal by default. Example non-ideal:
    // config.nonideal.enable_ohmic(1e-3).enable_ambipolar(1e-3);

    const Grid2D grid(nx, ny, 0.0, 1.0, 0.0, 1.0, 2);
    const BoundaryConditions bc(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic);

    StateField U(grid.get_size_x(), grid.get_size_y());
    TimeIntegratorWorkspace work(grid);

    initialize_state_field(U, grid, InitialConditionType::OrszagTang, config.gamma);

    SolveParams params;
    params.t_end = t_end;
    params.cfl = cfl;
    params.gamma = config.gamma;
    params.glm_alpha = 0.1;
    params.limiter = SlopeLimiter::MC;
    params.nonideal = config.nonideal;

    const SolveResult result = solve(U, work, grid, bc, params);

    std::cout << "t = " << result.t << ", steps = " << result.steps << ", c_h = " << result.c_h << '\n';
    std::cout << "nonideal = " << (params.nonideal.any() ? "on" : "off") << '\n';
    std::cout << "max|div B| = " << max_abs_div_b(U, grid) << '\n';
    return 0;
}
