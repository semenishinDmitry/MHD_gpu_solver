#pragma once

#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "initial_condition/inititial_condition.hpp"
#include "mhd/time_integrator.hpp"
#include "physics_config/mhd_config.hpp"
#include "state/state_field.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// High-level owning API for C++ and Python bindings.
class MHDSolver {
  public:
    MHDSolver(int nx, int ny, double x_min = 0.0, double x_max = 1.0, double y_min = 0.0,
              double y_max = 1.0, int ng = 2)
        : grid_(nx, ny, x_min, x_max, y_min, y_max, ng),
          bc_(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic),
          U_(grid_.get_size_x(), grid_.get_size_y()), work_(grid_)
    {
        if (ng < 2) {
            throw std::invalid_argument("MHDSolver requires ng >= 2");
        }
    }

    Grid2D& grid()
    {
        return grid_;
    }
    const Grid2D& grid() const
    {
        return grid_;
    }

    StateField& state()
    {
        return U_;
    }
    const StateField& state() const
    {
        return U_;
    }

    SolveParams& params()
    {
        return params_;
    }
    const SolveParams& params() const
    {
        return params_;
    }

    BoundaryConditions& boundary_conditions()
    {
        return bc_;
    }
    const BoundaryConditions& boundary_conditions() const
    {
        return bc_;
    }

    void set_periodic_bc()
    {
        bc_ = BoundaryConditions(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic);
    }

    void set_outflow_bc()
    {
        bc_ = BoundaryConditions(BoundaryConditionType::Outflow, BoundaryConditionType::Outflow);
    }

    void set_gamma(double gamma)
    {
        params_.gamma = gamma;
    }
    void set_cfl(double cfl)
    {
        params_.cfl = cfl;
    }
    void set_glm_alpha(double alpha)
    {
        params_.glm_alpha = alpha;
    }
    void set_limiter_minmod()
    {
        params_.limiter = SlopeLimiter::Minmod;
    }
    void set_limiter_mc()
    {
        params_.limiter = SlopeLimiter::MC;
    }

    void set_ideal()
    {
        params_.nonideal = NonIdealConfig::ideal();
    }

    void enable_ohmic(double eta)
    {
        params_.nonideal.enable_ohmic(eta);
    }
    void enable_hall(double eta)
    {
        params_.nonideal.enable_hall(eta);
    }
    void enable_ambipolar(double eta)
    {
        params_.nonideal.enable_ambipolar(eta);
    }

    void disable_ohmic()
    {
        params_.nonideal.disable_ohmic();
    }
    void disable_hall()
    {
        params_.nonideal.disable_hall();
    }
    void disable_ambipolar()
    {
        params_.nonideal.disable_ambipolar();
    }

    void initialize(const std::string& name)
    {
        InitialConditionType type = InitialConditionType::Uniform;
        if (name == "uniform" || name == "Uniform") {
            type = InitialConditionType::Uniform;
        } else if (name == "sine" || name == "SineWave") {
            type = InitialConditionType::SineWave;
        } else if (name == "blast" || name == "BlastWave") {
            type = InitialConditionType::BlastWave;
        } else if (name == "orszag_tang" || name == "OrszagTang") {
            type = InitialConditionType::OrszagTang;
        } else {
            throw std::invalid_argument("Unknown initial condition: " + name);
        }
        initialize_state_field(U_, grid_, type, params_.gamma);
        apply_boundary_conditions(U_, grid_, bc_);
        t_ = 0.0;
        steps_ = 0;
    }

    SolveResult run(double t_end)
    {
        if (t_end < 0.0) {
            throw std::invalid_argument("run: t_end must be non-negative");
        }
        SolveParams local = params_;
        local.t_end = t_end;
        const SolveResult r = solve(U_, work_, grid_, bc_, local);
        t_ += r.t;
        steps_ += r.steps;
        last_c_h_ = r.c_h;

        SolveResult out = r;
        out.t = t_;
        out.steps = steps_;
        out.c_h = last_c_h_;
        return out;
    }

    SolveResult advance_to(double t_abs)
    {
        if (t_abs < t_) {
            throw std::invalid_argument("advance_to: target time is earlier than current time");
        }
        return run(t_abs - t_);
    }

    double time() const
    {
        return t_;
    }
    int steps() const
    {
        return steps_;
    }
    double c_h() const
    {
        return last_c_h_;
    }

    double max_div_b() const
    {
        return max_abs_div_b(U_, grid_);
    }

    // Flat row-major copies (ny * nx_tot), including ghosts. Useful for Python/numpy.
    std::vector<double> copy_field(const std::string& name) const
    {
        const std::vector<double>* src = nullptr;
        if (name == "rho") {
            src = &U_.rho;
        } else if (name == "mx") {
            src = &U_.mx;
        } else if (name == "my") {
            src = &U_.my;
        } else if (name == "mz") {
            src = &U_.mz;
        } else if (name == "energy") {
            src = &U_.energy;
        } else if (name == "bx") {
            src = &U_.bx;
        } else if (name == "by") {
            src = &U_.by;
        } else if (name == "bz") {
            src = &U_.bz;
        } else if (name == "psi") {
            src = &U_.psi;
        } else {
            throw std::invalid_argument("Unknown field: " + name);
        }
        return *src;
    }

    int size_x() const
    {
        return grid_.get_size_x();
    }
    int size_y() const
    {
        return grid_.get_size_y();
    }
    int nx() const
    {
        return grid_.nx;
    }
    int ny() const
    {
        return grid_.ny;
    }
    int ng() const
    {
        return grid_.ng;
    }

  private:
    Grid2D grid_;
    BoundaryConditions bc_;
    StateField U_;
    TimeIntegratorWorkspace work_;
    SolveParams params_{};
    double t_ = 0.0;
    int steps_ = 0;
    double last_c_h_ = 0.0;
};
