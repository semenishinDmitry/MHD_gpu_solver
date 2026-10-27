#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mhd/mhd_types.hpp"
#include "mhd/solver_api.hpp"
#include "physics_config/mhd_config.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

py::array_t<double> field_to_numpy(const MHDSolver& solver, const std::string& name)
{
    auto data = solver.copy_field(name);
    const py::ssize_t ny = solver.size_y();
    const py::ssize_t nx = solver.size_x();
    auto result = py::array_t<double>({ny, nx});
    auto buf = result.mutable_unchecked<2>();
    for (py::ssize_t j = 0; j < ny; ++j) {
        for (py::ssize_t i = 0; i < nx; ++i) {
            buf(j, i) = data[static_cast<std::size_t>(j * nx + i)];
        }
    }
    return result;
}

py::array_t<std::uint8_t> mask_to_numpy(const MHDSolver& solver)
{
    auto data = solver.copy_mask();
    const py::ssize_t ny = solver.ny();
    const py::ssize_t nx = solver.nx();
    auto result = py::array_t<std::uint8_t>({ny, nx});
    auto buf = result.mutable_unchecked<2>();
    for (py::ssize_t j = 0; j < ny; ++j) {
        for (py::ssize_t i = 0; i < nx; ++i) {
            buf(j, i) = data[static_cast<std::size_t>(j * nx + i)];
        }
    }
    return result;
}

void set_mask_from_numpy(MHDSolver& solver,
                         py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> arr)
{
    if (arr.ndim() != 2) {
        throw std::invalid_argument("set_mask expects a 2D array (ny, nx)");
    }
    if (arr.shape(0) != solver.ny() || arr.shape(1) != solver.nx()) {
        throw std::invalid_argument("set_mask shape must be (ny, nx)");
    }
    auto buf = arr.unchecked<2>();
    std::vector<std::uint8_t> flat(static_cast<std::size_t>(solver.ny() * solver.nx()));
    for (py::ssize_t j = 0; j < arr.shape(0); ++j) {
        for (py::ssize_t i = 0; i < arr.shape(1); ++i) {
            flat[static_cast<std::size_t>(j * solver.nx() + i)] = buf(j, i);
        }
    }
    solver.set_mask(flat.data(), flat.size());
}

void set_field_from_numpy(MHDSolver& solver, const std::string& name,
                          py::array_t<double, py::array::c_style | py::array::forcecast> arr)
{
    if (arr.ndim() != 2) {
        throw std::invalid_argument("set_field expects a 2D array");
    }
    auto buf = arr.unchecked<2>();
    const py::ssize_t ny = arr.shape(0);
    const py::ssize_t nx = arr.shape(1);
    std::vector<double> flat(static_cast<std::size_t>(ny * nx));
    for (py::ssize_t j = 0; j < ny; ++j) {
        for (py::ssize_t i = 0; i < nx; ++i) {
            flat[static_cast<std::size_t>(j * nx + i)] = buf(j, i);
        }
    }

    if (ny == solver.size_y() && nx == solver.size_x()) {
        solver.set_field(name, flat.data(), flat.size());
    } else if (ny == solver.ny() && nx == solver.nx()) {
        solver.set_field_interior(name, flat.data(), flat.size());
    } else {
        throw std::invalid_argument(
            "set_field shape must be full (size_y, size_x) or interior (ny, nx)");
    }
}

void set_primitives_from_numpy(
    MHDSolver& solver, py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> vx,
    py::array_t<double, py::array::c_style | py::array::forcecast> vy,
    py::array_t<double, py::array::c_style | py::array::forcecast> vz,
    py::array_t<double, py::array::c_style | py::array::forcecast> pressure,
    py::array_t<double, py::array::c_style | py::array::forcecast> bx,
    py::array_t<double, py::array::c_style | py::array::forcecast> by,
    py::array_t<double, py::array::c_style | py::array::forcecast> bz, py::object psi_obj)
{
    auto require2 = [](py::array_t<double, py::array::c_style | py::array::forcecast>& a,
                       const char* label) {
        if (a.ndim() != 2) {
            throw std::invalid_argument(std::string(label) + " must be 2D");
        }
    };
    require2(rho, "rho");
    require2(vx, "vx");
    require2(vy, "vy");
    require2(vz, "vz");
    require2(pressure, "pressure");
    require2(bx, "bx");
    require2(by, "by");
    require2(bz, "bz");

    const py::ssize_t ny = rho.shape(0);
    const py::ssize_t nx = rho.shape(1);
    if (ny != solver.ny() || nx != solver.nx()) {
        throw std::invalid_argument("primitive arrays must have shape (ny, nx)");
    }

    auto flatten = [&](py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
        if (a.shape(0) != ny || a.shape(1) != nx) {
            throw std::invalid_argument("primitive arrays must share the same shape");
        }
        auto buf = a.unchecked<2>();
        std::vector<double> flat(static_cast<std::size_t>(ny * nx));
        for (py::ssize_t j = 0; j < ny; ++j) {
            for (py::ssize_t i = 0; i < nx; ++i) {
                flat[static_cast<std::size_t>(j * nx + i)] = buf(j, i);
            }
        }
        return flat;
    };

    auto rho_f = flatten(rho);
    auto vx_f = flatten(vx);
    auto vy_f = flatten(vy);
    auto vz_f = flatten(vz);
    auto p_f = flatten(pressure);
    auto bx_f = flatten(bx);
    auto by_f = flatten(by);
    auto bz_f = flatten(bz);

    std::vector<double> psi_f;
    const double* psi_ptr = nullptr;
    if (!psi_obj.is_none()) {
        auto psi =
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(psi_obj);
        require2(psi, "psi");
        psi_f = flatten(psi);
        psi_ptr = psi_f.data();
    }

    solver.set_primitives_interior(rho_f.data(), vx_f.data(), vy_f.data(), vz_f.data(), p_f.data(),
                                   bx_f.data(), by_f.data(), bz_f.data(), psi_ptr, rho_f.size());
}

} // namespace

PYBIND11_MODULE(mhd_solver, m)
{
    m.doc() = "2D ideal / non-ideal GLM-MHD finite-volume solver";


    py::enum_<SlopeLimiter>(m, "SlopeLimiter")
        .value("Minmod", SlopeLimiter::Minmod)
        .value("MC", SlopeLimiter::MC);

    py::enum_<BoundaryConditionType>(m, "BoundaryConditionType")
        .value("Periodic", BoundaryConditionType::Periodic)
        .value("Reflective", BoundaryConditionType::Reflective)
        .value("Outflow", BoundaryConditionType::Outflow);

    py::class_<NonIdealConfig>(m, "NonIdealConfig")
        .def(py::init<>())
        .def_static("ideal", &NonIdealConfig::ideal)
        .def_static("with_ohmic", &NonIdealConfig::with_ohmic, py::arg("eta"))
        .def_static("with_hall", &NonIdealConfig::with_hall, py::arg("eta"))
        .def_static("with_ambipolar", &NonIdealConfig::with_ambipolar, py::arg("eta"))
        .def("enable_ohmic", &NonIdealConfig::enable_ohmic, py::arg("eta"),
             py::return_value_policy::reference_internal)
        .def("enable_hall", &NonIdealConfig::enable_hall, py::arg("eta"),
             py::return_value_policy::reference_internal)
        .def("enable_ambipolar", &NonIdealConfig::enable_ambipolar, py::arg("eta"),
             py::return_value_policy::reference_internal)
        .def("disable_ohmic", &NonIdealConfig::disable_ohmic,
             py::return_value_policy::reference_internal)
        .def("disable_hall", &NonIdealConfig::disable_hall,
             py::return_value_policy::reference_internal)
        .def("disable_ambipolar", &NonIdealConfig::disable_ambipolar,
             py::return_value_policy::reference_internal)
        .def("any", &NonIdealConfig::any)
        .def_readwrite("ohmic", &NonIdealConfig::ohmic)
        .def_readwrite("hall", &NonIdealConfig::hall)
        .def_readwrite("ambipolar", &NonIdealConfig::ambipolar)
        .def_readwrite("eta_ohm", &NonIdealConfig::eta_ohm)
        .def_readwrite("eta_hall", &NonIdealConfig::eta_hall)
        .def_readwrite("eta_ambipolar", &NonIdealConfig::eta_ambipolar);

    py::class_<SolveResult>(m, "SolveResult")
        .def_readonly("t", &SolveResult::t)
        .def_readonly("steps", &SolveResult::steps)
        .def_readonly("c_h", &SolveResult::c_h);

    py::class_<SolveParams>(m, "SolveParams")
        .def(py::init<>())
        .def_readwrite("t_end", &SolveParams::t_end)
        .def_readwrite("cfl", &SolveParams::cfl)
        .def_readwrite("gamma", &SolveParams::gamma)
        .def_readwrite("glm_alpha", &SolveParams::glm_alpha)
        .def_readwrite("limiter", &SolveParams::limiter)
        .def_readwrite("nonideal", &SolveParams::nonideal)
        .def_readwrite("max_steps", &SolveParams::max_steps);

    py::class_<MHDSolver>(m, "MHDSolver")
        .def(py::init<int, int, double, double, double, double, int>(), py::arg("nx"),
             py::arg("ny"), py::arg("x_min") = 0.0, py::arg("x_max") = 1.0, py::arg("y_min") = 0.0,
             py::arg("y_max") = 1.0, py::arg("ng") = 2)
        .def("set_periodic_bc", &MHDSolver::set_periodic_bc)
        .def("set_outflow_bc", &MHDSolver::set_outflow_bc)
        .def("set_bc", &MHDSolver::set_bc, py::arg("type_x"), py::arg("type_y"))
        .def("set_gamma", &MHDSolver::set_gamma)
        .def("set_cfl", &MHDSolver::set_cfl)
        .def("set_glm_alpha", &MHDSolver::set_glm_alpha)
        .def("set_limiter_minmod", &MHDSolver::set_limiter_minmod)
        .def("set_limiter_mc", &MHDSolver::set_limiter_mc)
        .def("set_ideal", &MHDSolver::set_ideal)
        .def("enable_ohmic", &MHDSolver::enable_ohmic)
        .def("enable_hall", &MHDSolver::enable_hall)
        .def("enable_ambipolar", &MHDSolver::enable_ambipolar)
        .def("disable_ohmic", &MHDSolver::disable_ohmic)
        .def("disable_hall", &MHDSolver::disable_hall)
        .def("disable_ambipolar", &MHDSolver::disable_ambipolar)
        .def("set_mask", &set_mask_from_numpy, py::arg("mask"),
             "Set interior fluid/solid mask as uint8 array (ny, nx); 1=fluid, 0=solid")
        .def("get_mask", &mask_to_numpy, "Return interior mask as uint8 array (ny, nx)")
        .def("clear_mask", &MHDSolver::clear_mask)
        .def("set_walls_reflective", &MHDSolver::set_walls_reflective, py::arg("enabled"))
        .def_property_readonly("walls_reflective", &MHDSolver::walls_reflective)
        .def("set_field", &set_field_from_numpy, py::arg("name"), py::arg("array"),
             "Write conserved field: full (size_y,size_x) or interior (ny,nx)")
        .def("set_primitives", &set_primitives_from_numpy, py::arg("rho"), py::arg("vx"),
             py::arg("vy"), py::arg("vz"), py::arg("pressure"), py::arg("bx"), py::arg("by"),
             py::arg("bz"), py::arg("psi") = py::none(),
             "Set interior conserved state from primitive arrays (ny, nx)")
        .def("initialize", &MHDSolver::initialize, py::arg("name"))
        .def("run", &MHDSolver::run, py::arg("t_end"))
        .def("advance_to", &MHDSolver::advance_to, py::arg("t_abs"))
        .def_property_readonly("time", &MHDSolver::time)
        .def_property_readonly("steps", &MHDSolver::steps)
        .def_property_readonly("c_h", &MHDSolver::c_h)
        .def("max_div_b", &MHDSolver::max_div_b)
        .def("field", &field_to_numpy, py::arg("name"),
             "Return field as numpy array (ny, nx) including ghosts. "
             "name in {rho,mx,my,mz,energy,bx,by,bz,psi}")
        .def_property_readonly("nx", &MHDSolver::nx)
        .def_property_readonly("ny", &MHDSolver::ny)
        .def_property_readonly("ng", &MHDSolver::ng)
        .def_property_readonly("size_x", &MHDSolver::size_x)
        .def_property_readonly("size_y", &MHDSolver::size_y)
        .def_property_readonly("x_min", &MHDSolver::x_min)
        .def_property_readonly("x_max", &MHDSolver::x_max)
        .def_property_readonly("y_min", &MHDSolver::y_min)
        .def_property_readonly("y_max", &MHDSolver::y_max)
        .def_property("params", py::overload_cast<>(&MHDSolver::params),
                      py::overload_cast<>(&MHDSolver::params, py::const_));
}
