#pragma once

// Conserved GLM-MHD state (mu0 = 1), 9 components.
// mx = rho * vx, my = rho * vy, mz = rho * vz.
// psi: Dedner hyperbolic divergence-cleaning potential.
struct MHDState {
    double rho = 0.0;
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    double energy = 0.0;
    double bx = 0.0;
    double by = 0.0;
    double bz = 0.0;
    double psi = 0.0;
};

// Primitive GLM-MHD state (mu0 = 1).
struct MHDPrimitive {
    double rho = 0.0;
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    double pressure = 0.0;
    double bx = 0.0;
    double by = 0.0;
    double bz = 0.0;
    double psi = 0.0;
};

// GLM-MHD flux vector in one coordinate direction.
struct MHDFlux {
    double rho = 0.0;
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    double energy = 0.0;
    double bx = 0.0;
    double by = 0.0;
    double bz = 0.0;
    double psi = 0.0;
};

enum class SlopeLimiter {
    Minmod,
    MC
};
