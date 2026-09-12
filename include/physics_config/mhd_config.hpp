#pragma once

#include <algorithm>
#include <cmath>

// Independent non-ideal MHD diffusivities (μ0 = 1).
// Enable terms separately; disabled terms are not evaluated in the hot loop.
//
// Generalized Ohm electric field:
//   E = η_O J
//     + η_H (J × B) / |B|
//     + η_A (J × B) × B / |B|²
struct NonIdealConfig {
    bool ohmic = false;
    bool hall = false;
    bool ambipolar = false;

    double eta_ohm = 0.0;
    double eta_hall = 0.0;
    double eta_ambipolar = 0.0;

    static NonIdealConfig ideal() { return {}; }

    static NonIdealConfig with_ohmic(double eta)
    {
        NonIdealConfig c;
        c.ohmic = true;
        c.eta_ohm = eta;
        return c;
    }

    static NonIdealConfig with_hall(double eta)
    {
        NonIdealConfig c;
        c.hall = true;
        c.eta_hall = eta;
        return c;
    }

    static NonIdealConfig with_ambipolar(double eta)
    {
        NonIdealConfig c;
        c.ambipolar = true;
        c.eta_ambipolar = eta;
        return c;
    }

    NonIdealConfig& enable_ohmic(double eta)
    {
        ohmic = true;
        eta_ohm = eta;
        return *this;
    }

    NonIdealConfig& enable_hall(double eta)
    {
        hall = true;
        eta_hall = eta;
        return *this;
    }

    NonIdealConfig& enable_ambipolar(double eta)
    {
        ambipolar = true;
        eta_ambipolar = eta;
        return *this;
    }

    NonIdealConfig& disable_ohmic()
    {
        ohmic = false;
        return *this;
    }

    NonIdealConfig& disable_hall()
    {
        hall = false;
        return *this;
    }

    NonIdealConfig& disable_ambipolar()
    {
        ambipolar = false;
        return *this;
    }

    [[nodiscard]] bool any() const
    {
        return (ohmic && eta_ohm != 0.0) || (hall && eta_hall != 0.0) ||
               (ambipolar && eta_ambipolar != 0.0);
    }

    // Effective diffusivity for explicit parabolic CFL (Hall treated as dispersive ~ η_H).
    [[nodiscard]] double max_diffusivity() const
    {
        double eta = 0.0;
        if (ohmic) {
            eta = std::max(eta, std::abs(eta_ohm));
        }
        if (hall) {
            eta = std::max(eta, std::abs(eta_hall));
        }
        if (ambipolar) {
            eta = std::max(eta, std::abs(eta_ambipolar));
        }
        return eta;
    }
};

struct MHDConfig {
    double gamma = 5.0 / 3.0;
    double Omega = 0.0;
    double q = 0.0;
    NonIdealConfig nonideal{};

    MHDConfig() = default;

    MHDConfig(double gamma_,
              double Omega_,
              double q_,
              double eta_ohm_,
              double eta_hall_,
              double eta_ambip_)
        : gamma(gamma_), Omega(Omega_), q(q_)
    {
        if (eta_ohm_ != 0.0) {
            nonideal.enable_ohmic(eta_ohm_);
        }
        if (eta_hall_ != 0.0) {
            nonideal.enable_hall(eta_hall_);
        }
        if (eta_ambip_ != 0.0) {
            nonideal.enable_ambipolar(eta_ambip_);
        }
    }

    // Backward-compatible accessors used by existing tests.
    double eta_ohm() const { return nonideal.eta_ohm; }
    double eta_hall() const { return nonideal.eta_hall; }
    double eta_ambip() const { return nonideal.eta_ambipolar; }
};
