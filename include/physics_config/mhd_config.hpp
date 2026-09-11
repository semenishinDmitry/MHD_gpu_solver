#pragma once

struct MHDConfig {
    double gamma = 5.0 / 3.0; // Adiabatic index
    double Omega = 0.0; // Rotaion rate
    double q = 0.0; // Shear parameter
    double eta_ohm = 0.0; // Ohmic resistivity
    double eta_hall = 0.0; // Hall resistivity
    double eta_ambip = 0.0; // Ambipolar resistivity

    MHDConfig() = default;
    MHDConfig(const double gamma, const double Omega, const double q, const double eta_ohm, const double eta_hall, const double eta_ambip)
        : gamma(gamma), Omega(Omega), q(q), eta_ohm(eta_ohm), eta_hall(eta_hall), eta_ambip(eta_ambip) {}
};