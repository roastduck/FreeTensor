#include <analyze/all_uses.h>
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

    m.def(
        "fixed_length_feature",
        static_cast<std::vector<double> (*)(const Stmt &)>(fixedLengthFeature));
    m.def(
        "fixed_length_feature",
        static_cast<std::vector<double> (*)(const Func &)>(fixedLengthFeature));

    m.def("feature_length", FixedLengthFeature::featureLen);

    m.def("find_multi_level_tiling", fakeFindMultiLevelTiling);

    m.def("all_reads",
          static_cast<std::unordered_set<std::string> (*)(const AST &, bool)>(
              &allReads),
          "ast"_a, "no_recurse_idx"_a = false);
    m.def("all_writes",
          static_cast<std::unordered_set<std::string> (*)(const AST &, bool)>(
              &allWrites),
          "ast"_a, "no_recurse_idx"_a = false);
    m.def("all_iters",
          static_cast<std::unordered_set<std::string> (*)(const AST &, bool)>(
              &allIters),
          "ast"_a, "no_recurse_idx"_a = false);
    m.def("all_names",
          static_cast<std::unordered_set<std::string> (*)(const AST &, bool)>(
              &allNames),
          "ast"_a, "no_recurse_idx"_a = false);
}

} // namespace freetensor
