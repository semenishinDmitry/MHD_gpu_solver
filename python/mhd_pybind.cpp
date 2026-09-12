#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mhd/mhd_types.hpp"
#include "mhd/solver_api.hpp"
#include "physics_config/mhd_config.hpp"

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
        .def_property("params", py::overload_cast<>(&MHDSolver::params),
                      py::overload_cast<>(&MHDSolver::params, py::const_));
}
