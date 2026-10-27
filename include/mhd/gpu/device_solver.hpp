#pragma once

// Accelerator backends (CUDA, Metal) behind one interface.
//
// A DeviceSolver owns a device-resident copy of the conserved state and runs the same scheme as
// the CPU solver (MUSCL + HLL + Dedner GLM, SSP-RK2, CFL dt, periodic/outflow BCs, non-ideal
// Ohmic/Hall/ambipolar terms, staircase solid masks). The time step is computed on the device and
// the host synchronizes only once per batch of steps.
//
// Backends are compiled only when enabled in CMake:
//   MHD_ENABLE_METAL (Apple, default ON)  -> MHD_HAS_METAL=1, fp32 (Apple GPUs have no fp64)
//   MHD_ENABLE_CUDA  (default OFF)        -> MHD_HAS_CUDA=1, fp64 or fp32

#include "boundary_conditions/boundary_conditions.hpp"
#include "grid/grid.hpp"
#include "mhd/mask.hpp"
#include "mhd/solve_types.hpp"
#include "state/state_field.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace mhd {

enum class Backend { CPU, CUDA, Metal };

enum class Precision { FP64, FP32 };

inline const char* backend_name(Backend b)
{
    switch (b) {
    case Backend::CPU:
        return "cpu";
    case Backend::CUDA:
        return "cuda";
    case Backend::Metal:
        return "metal";
    }
    return "unknown";
}

inline Backend parse_backend(const std::string& s)
{
    if (s == "cpu" || s == "CPU") {
        return Backend::CPU;
    }
    if (s == "cuda" || s == "CUDA") {
        return Backend::CUDA;
    }
    if (s == "metal" || s == "Metal") {
        return Backend::Metal;
    }
    throw std::invalid_argument("Unknown backend: " + s + " (expected cpu, cuda or metal)");
}

class DeviceSolver {
  public:
    virtual ~DeviceSolver() = default;

    // Human-readable device name, e.g. "Metal: Apple M5".
    virtual std::string device_name() const = 0;
    virtual Precision precision() const = 0;

    // Boundary conditions and (optional) solid mask; mask == nullptr means all fluid.
    virtual void configure(const BoundaryConditions& bc, const FluidMask* mask,
                           bool reflective_walls) = 0;

    // Host <-> device copies of the full conserved state (including ghosts).
    virtual void upload(const StateField& U) = 0;
    virtual void download(StateField& U) = 0;

    // Advances the device state by t_end (relative time). Same contract as ::solve().
    virtual SolveResult solve(const SolveParams& params) = 0;
};

bool metal_available();
bool cuda_available();

// Throw std::runtime_error if the backend was not compiled in or no device is present.
std::unique_ptr<DeviceSolver> make_metal_solver(const Grid2D& grid);
std::unique_ptr<DeviceSolver> make_cuda_solver(const Grid2D& grid, Precision precision);

inline bool backend_available(Backend b)
{
    switch (b) {
    case Backend::CPU:
        return true;
    case Backend::CUDA:
        return cuda_available();
    case Backend::Metal:
        return metal_available();
    }
    return false;
}

inline std::unique_ptr<DeviceSolver> make_device_solver(Backend b, const Grid2D& grid,
                                                        Precision precision = Precision::FP64)
{
    switch (b) {
    case Backend::CUDA:
        return make_cuda_solver(grid, precision);
    case Backend::Metal:
        return make_metal_solver(grid);
    case Backend::CPU:
        break;
    }
    throw std::invalid_argument("make_device_solver: CPU is not a device backend");
}

} // namespace mhd

// Header-only stubs when a backend is not compiled in, so callers never need #ifdefs.
#if !defined(MHD_HAS_METAL)
inline bool mhd::metal_available()
{
    return false;
}
inline std::unique_ptr<mhd::DeviceSolver> mhd::make_metal_solver(const Grid2D&)
{
    throw std::runtime_error("Metal backend not built (configure with -DMHD_ENABLE_METAL=ON on "
                             "macOS)");
}
#endif

#if !defined(MHD_HAS_CUDA)
inline bool mhd::cuda_available()
{
    return false;
}
inline std::unique_ptr<mhd::DeviceSolver> mhd::make_cuda_solver(const Grid2D&, Precision)
{
    throw std::runtime_error("CUDA backend not built (configure with -DMHD_ENABLE_CUDA=ON)");
}
#endif
