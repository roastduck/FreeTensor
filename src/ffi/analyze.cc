#include <analyze/structural_feature.h>
#include <ffi.h>

namespace ir {

using namespace pybind11::literals;

void init_ffi_analyze(py::module_ &m) {
    py::class_<NodeFeature>(m, "NodeFeature")
        .def_readonly("loadArea", &NodeFeature::loadArea_)
        .def_readonly("storeArea", &NodeFeature::storeArea_)
        .def_readonly("accessArea", &NodeFeature::accessArea_);
    m.def("structural_feature", structuralFeature);
}

} // namespace ir
