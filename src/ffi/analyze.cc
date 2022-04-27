#include <analyze/find_multi_level_tiling.h>
#include <analyze/fixed_length_feature.h>
#include <analyze/structural_feature.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_analyze(py::module_ &m) {
    py::class_<NodeFeature>(m, "NodeFeature")
        .def_readonly("op_cnt", &NodeFeature::opCnt_)
        .def_readonly("load_cnt", &NodeFeature::loadCnt_)
        .def_readonly("store_cnt", &NodeFeature::storeCnt_)
        .def_readonly("access_cnt", &NodeFeature::accessCnt_)
        .def_readonly("load_area", &NodeFeature::loadArea_)
        .def_readonly("store_area", &NodeFeature::storeArea_)
        .def_readonly("access_area", &NodeFeature::accessArea_);
    m.def("structural_feature", structuralFeature);

    m.def("fixed_length_feature", fixedLengthFeature);

    m.def("feature_length", FixedLengthFeature::featureLen);

    m.def("find_multi_level_tiling", fakeFindMultiLevelTiling);
}

} // namespace freetensor
