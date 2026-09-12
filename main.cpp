#include "mhd/solver_api.hpp"

#include <iostream>

int main()
{
    MHDSolver solver(64, 64);
    solver.set_cfl(0.4);
    solver.set_glm_alpha(0.1);
    solver.set_limiter_mc();
    // Ideal by default. Non-ideal example: solver.enable_ohmic(1e-3);

    solver.initialize("orszag_tang");
    const SolveResult result = solver.run(0.05);

    std::cout << "t = " << result.t << ", steps = " << result.steps << ", c_h = " << result.c_h << '\n';
    std::cout << "max|div B| = " << solver.max_div_b() << '\n';
    return 0;
}
