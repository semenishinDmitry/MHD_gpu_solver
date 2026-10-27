#pragma once

#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "initial_condition/inititial_condition.hpp"
#include "mhd/gpu/device_solver.hpp"
#include "mhd/mask.hpp"
#include "mhd/mhd_physics.hpp"
#include "mhd/time_integrator.hpp"
#include "physics_config/mhd_config.hpp"
#include "state/state_field.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
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
          U_(grid_.get_size_x(), grid_.get_size_y()), work_(grid_), mask_(nx, ny, true)
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

    FluidMask& mask()
    {
        return mask_;
    }
    const FluidMask& mask() const
    {
        return mask_;
    }

    void set_periodic_bc()
    {
        bc_ = BoundaryConditions(BoundaryConditionType::Periodic, BoundaryConditionType::Periodic);
    }

    void set_outflow_bc()
    {
        bc_ = BoundaryConditions(BoundaryConditionType::Outflow, BoundaryConditionType::Outflow);
    }

    void set_bc(BoundaryConditionType type_x, BoundaryConditionType type_y)
    {
        bc_ = BoundaryConditions(type_x, type_y);
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

    // Interior mask: row-major ny*nx, 1=fluid, 0=solid.
    void set_mask(const std::uint8_t* data, std::size_t n)
    {
        mask_.set_from_interior(data, n, grid_.nx, grid_.ny);
        if (walls_reflective_) {
            apply_solid_wall_fill(U_, grid_, mask_, params_.gamma);
        }
    }

    void clear_mask()
    {
        mask_ = FluidMask(grid_.nx, grid_.ny, true);
    }

    std::vector<std::uint8_t> copy_mask() const
    {
        return mask_.fluid;
    }

    // Execution backend: "cpu" (default; SIMD + threads), "metal" (fp32) or "cuda".
    // GPU backends upload the state at the start of run() and download it at the end, so all
    // other methods keep working on the host copy.
    void set_backend(mhd::Backend backend, mhd::Precision precision = mhd::Precision::FP64)
    {
        if (backend != mhd::Backend::CPU && !mhd::backend_available(backend)) {
            throw std::runtime_error(std::string("Backend not available: ") +
                                     mhd::backend_name(backend));
        }
        backend_ = backend;
        precision_ = precision;
        device_.reset();
    }

    void set_backend(const std::string& name, const std::string& precision = "fp64")
    {
        set_backend(mhd::parse_backend(name),
                    precision == "fp32" ? mhd::Precision::FP32 : mhd::Precision::FP64);
    }

    mhd::Backend backend() const
    {
        return backend_;
    }

    // Device description, e.g. "Metal: Apple M5"; "cpu" for the CPU backend.
    std::string backend_device()
    {
        if (backend_ == mhd::Backend::CPU) {
            return "cpu";
        }
        return device().device_name();
    }

    void set_walls_reflective(bool enabled)
    {
        walls_reflective_ = enabled;
    }

    bool walls_reflective() const
    {
        return walls_reflective_;
    }

    // Write a conserved SoA field (full array including ghosts), row-major ny_tot * nx_tot.
    void set_field(const std::string& name, const double* data, std::size_t n)
    {
        const std::size_t expected = static_cast<std::size_t>(grid_.get_size_x()) *
                                     static_cast<std::size_t>(grid_.get_size_y());
        if (n != expected) {
            throw std::invalid_argument("set_field: size does not match full SoA layout");
        }
        std::vector<double>* dst = field_ptr(name);
        dst->assign(data, data + n);
    }

    // Write an interior-only field (ny * nx), leaving ghosts untouched.
    void set_field_interior(const std::string& name, const double* data, std::size_t n)
    {
        const std::size_t expected =
            static_cast<std::size_t>(grid_.nx) * static_cast<std::size_t>(grid_.ny);
        if (n != expected) {
            throw std::invalid_argument("set_field_interior: size does not match nx*ny");
        }
        std::vector<double>* dst = field_ptr(name);
        const int ng = grid_.ng;
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                const int gi = i + ng;
                const int gj = j + ng;
                (*dst)[static_cast<std::size_t>(U_.index(gi, gj))] =
                    data[static_cast<std::size_t>(j * grid_.nx + i)];
            }
        }
    }

    // Fill conserved state from primitive values on interior cells (rho,vx,vy,vz,p,bx,by,bz,psi).
    void set_primitives_interior(const double* rho, const double* vx, const double* vy,
                                 const double* vz, const double* pressure, const double* bx,
                                 const double* by, const double* bz, const double* psi,
                                 std::size_t n)
    {
        const std::size_t expected =
            static_cast<std::size_t>(grid_.nx) * static_cast<std::size_t>(grid_.ny);
        if (n != expected) {
            throw std::invalid_argument("set_primitives_interior: size does not match nx*ny");
        }
        const int ng = grid_.ng;
        for (int j = 0; j < grid_.ny; ++j) {
            for (int i = 0; i < grid_.nx; ++i) {
                const std::size_t k = static_cast<std::size_t>(j * grid_.nx + i);
                MHDPrimitive W{};
                W.rho = rho[k];
                W.vx = vx[k];
                W.vy = vy[k];
                W.vz = vz[k];
                W.pressure = pressure[k];
                W.bx = bx[k];
                W.by = by[k];
                W.bz = bz[k];
                W.psi = psi != nullptr ? psi[k] : 0.0;
                U_.set_state(i + ng, j + ng, to_conservative(W, params_.gamma));
            }
        }
        apply_boundary_conditions(U_, grid_, bc_);
        if (walls_reflective_ && !mask_.all_fluid()) {
            apply_solid_wall_fill(U_, grid_, mask_, params_.gamma);
        }
    }

    void initialize(const std::string& name)
    {
        const InitialConditionType type = parse_initial_condition_name(name);
        // Problem-specific defaults when the caller has not already set them.
        if (type == InitialConditionType::Sod) {
            params_.gamma = 1.4;
        } else if (type == InitialConditionType::BrioWu) {
            params_.gamma = 2.0;
        }
        initialize_state_field(U_, grid_, type, params_.gamma);
        apply_boundary_conditions(U_, grid_, bc_);
        if (walls_reflective_ && !mask_.all_fluid()) {
            apply_solid_wall_fill(U_, grid_, mask_, params_.gamma);
        }
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
        const FluidMask* mask_ptr = mask_.all_fluid() ? nullptr : &mask_;
        SolveResult r{};
        if (backend_ == mhd::Backend::CPU) {
            r = solve(U_, work_, grid_, bc_, local, mask_ptr, walls_reflective_);
        } else {
            mhd::DeviceSolver& dev = device();
            dev.configure(bc_, mask_ptr, walls_reflective_);
            dev.upload(U_);
            r = dev.solve(local);
            dev.download(U_);
        }
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
        return *field_ptr_const(name);
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

    double x_min() const
    {
        return grid_.x_min;
    }
    double x_max() const
    {
        return grid_.x_max;
    }
    double y_min() const
    {
        return grid_.y_min;
    }
    double y_max() const
    {
        return grid_.y_max;
    }

  private:
    mhd::DeviceSolver& device()
    {
        if (!device_) {
            device_ = mhd::make_device_solver(backend_, grid_, precision_);
        }
        return *device_;
    }

    std::vector<double>* field_ptr(const std::string& name)
    {
        if (name == "rho") {
            return &U_.rho;
        }
        if (name == "mx") {
            return &U_.mx;
        }
        if (name == "my") {
            return &U_.my;
        }
        if (name == "mz") {
            return &U_.mz;
        }
        if (name == "energy") {
            return &U_.energy;
        }
        if (name == "bx") {
            return &U_.bx;
        }
        if (name == "by") {
            return &U_.by;
        }
        if (name == "bz") {
            return &U_.bz;
        }
        if (name == "psi") {
            return &U_.psi;
        }
        throw std::invalid_argument("Unknown field: " + name);
    }

    const std::vector<double>* field_ptr_const(const std::string& name) const
    {
        return const_cast<MHDSolver*>(this)->field_ptr(name);
    }

    Grid2D grid_;
    BoundaryConditions bc_;
    StateField U_;
    TimeIntegratorWorkspace work_;
    FluidMask mask_;
    SolveParams params_{};
    double t_ = 0.0;
    int steps_ = 0;
    double last_c_h_ = 0.0;
    bool walls_reflective_ = true;
    mhd::Backend backend_ = mhd::Backend::CPU;
    mhd::Precision precision_ = mhd::Precision::FP64;
    std::shared_ptr<mhd::DeviceSolver> device_;
};
