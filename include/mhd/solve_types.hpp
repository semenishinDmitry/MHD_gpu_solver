#pragma once

// Parameters and result of a time integration, shared by the CPU solver and the GPU backends.

#include "mhd/mhd_types.hpp"
#include "physics_config/mhd_config.hpp"

struct SolveParams {
    double t_end = 0.0;
    double cfl = 0.4;
    double gamma = 5.0 / 3.0;
    double glm_alpha = 0.1;
    SlopeLimiter limiter = SlopeLimiter::MC;
    NonIdealConfig nonideal = NonIdealConfig::ideal();
    int max_steps = 1'000'000;
};

struct SolveResult {
    double t = 0.0;
    int steps = 0;
    double c_h = 0.0;
};
