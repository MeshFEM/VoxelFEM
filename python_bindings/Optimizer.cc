#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
namespace py = pybind11;
#include "../MethodOfMovingAsymptotes.hh"
#include <pybind11/eigen.h>

PYBIND11_MODULE(pyOptimizer, m)
{
    using VXd = Eigen::VectorXd;
    using AXd = Eigen::ArrayXd;
    using AXXd = Eigen::ArrayXXd;
    py::class_<MMA>(m, "MMA")
        .def(py::init<int, int, const AXd&, const AXd&, const std::function<AXd(const AXd&)>&, const std::function<AXXd(const AXd&)>&>(),
            py::arg("numVars"), py::arg("numConstr"), py::arg("xmin"), py::arg("xmax"), py::arg("f"), py::arg("df_dx"))
        .def("setInitialVar", &MMA::setInitialVar, py::arg("x0"))
        .def("step", &MMA::step)
        .def("enableGCMMA", &MMA::enableGCMMA, py::arg("enable"))
        ;
}
