#include <pybind11/numpy.h>
#include <vector>

#include <driver.h>
#include <ffi.h>

namespace ir {

void init_ffi_driver(py::module_ &m) {
    py::class_<Driver>(m, "_Driver")
        .def(py::init<>())
        .def("buildAndLoad", &Driver::buildAndLoad)
        .def("setParamF32",
             [](Driver &d, int nth, py::array_t<float> &params) {
                 d.setParam(nth, params.mutable_unchecked().mutable_data());
             })
        .def("setParamI32",
             [](Driver &d, int nth, py::array_t<int32_t> &params) {
                 d.setParam(nth, params.mutable_unchecked().mutable_data());
             })
        .def("run", &Driver::run);
}

} // namespace ir

