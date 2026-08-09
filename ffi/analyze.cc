#include <optional>
#include <sstream>

#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <analyze/structural_feature.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

namespace {

struct AccessPointSnapshot {
    AST op_;
    Stmt stmt_;
    VarDef def_;
    std::string var_;
    ID defId_;
    int defAxis_;
    std::vector<Expr> access_;
    std::vector<std::pair<Expr, ID>> conds_;
};

struct AxisDirection {
    bool isNode_;
    ID id_;
    std::string parallel_;
    DepDirection direction_;
};

struct DependenceSnapshot {
    std::string var_;
    AccessPointSnapshot later_;
    AccessPointSnapshot earlier_;
    std::vector<AxisDirection> direction_;
    int iterDim_;
    std::optional<std::string> later2EarlierIter_;
    std::optional<std::string> laterIter2Idx_;
    std::optional<std::string> earlierIter2Idx_;
    std::optional<std::string> later2EarlierIterAllPossible_;
    std::optional<std::string> extConstraint_;
    std::string text_;
};

AccessPointSnapshot snapshotAccessPoint(const AccessPoint &acc) {
    return AccessPointSnapshot{acc.op_,         acc.stmt_,      acc.def_,
                               acc.def_->name_, acc.def_->id(), acc.defAxis_,
                               acc.access_,     acc.conds_};
}

std::optional<std::string> serializePBMap(const PBMap &map) {
    if (!map.isValid()) {
        return std::nullopt;
    }
    return map.toSerialized().data();
}

DependenceSnapshot snapshotDependence(const Dependence &dep) {
    std::vector<AxisDirection> direction;
    direction.reserve(dep.dir_.size());
    for (auto &&[nodeOrParallel, dir] : dep.dir_) {
        if (nodeOrParallel.isNode_) {
            direction.emplace_back(
                AxisDirection{true, nodeOrParallel.id_, "", dir});
        } else {
            direction.emplace_back(AxisDirection{
                false, ID(), toString(nodeOrParallel.parallel_), dir});
        }
    }

    std::ostringstream os;
    os << dep;
    return DependenceSnapshot{dep.var_,
                              snapshotAccessPoint(dep.later_),
                              snapshotAccessPoint(dep.earlier_),
                              std::move(direction),
                              dep.iterDim_,
                              serializePBMap(dep.later2EarlierIter_),
                              serializePBMap(dep.laterIter2Idx_),
                              serializePBMap(dep.earlierIter2Idx_),
                              serializePBMap(dep.later2EarlierIterAllPossible_),
                              serializePBMap(dep.extConstraint_),
                              os.str()};
}

FindDeps makeFindDeps(DepType depType, FindDepsMode mode,
                      const std::optional<std::vector<FindDepsDir>> &direction,
                      const std::optional<ID> &filterSubAST,
                      bool ignoreReductionWAW, bool eraseOutsideVarDef,
                      bool noProjectOutPrivateAxis) {
    auto finder = FindDeps()
                      .type(depType)
                      .mode(mode)
                      .ignoreReductionWAW(ignoreReductionWAW)
                      .eraseOutsideVarDef(eraseOutsideVarDef)
                      .noProjectOutPrivateAxis(noProjectOutPrivateAxis);
    if (direction.has_value()) {
        finder = finder.direction(*direction);
    }
    if (filterSubAST.has_value()) {
        finder = finder.filterSubAST(*filterSubAST);
    }
    return finder;
}

} // namespace

void init_ffi_analyze(py::module_ &m) {
    py::class_<NodeFeature>(m, "NodeFeature")
        .def_property_readonly(
            "op_cnt",
            [](const NodeFeature &feat) {
                return ranges::to<std::unordered_map<DataType, int64_t>>(
                    feat.opCnt_ |
                    views::transform(
                        [](auto &&kv) -> std::pair<DataType, int64_t> {
                            return {kv.first, kv.second};
                        }));
            })
        .def_readonly("load_cnt", &NodeFeature::loadCnt_)
        .def_readonly("store_cnt", &NodeFeature::storeCnt_)
        .def_readonly("access_cnt", &NodeFeature::accessCnt_)
        .def_readonly("load_area", &NodeFeature::loadArea_)
        .def_readonly("store_area", &NodeFeature::storeArea_)
        .def_readonly("access_area", &NodeFeature::accessArea_);
    m.def("structural_feature", structuralFeature);

    py::enum_<DepDirection>(m, "DepDirection")
        .value("Normal", DepDirection::Normal)
        .value("Inv", DepDirection::Inv)
        .value("Same", DepDirection::Same)
        .value("Different", DepDirection::Different);
    py::enum_<FindDepsMode>(m, "FindDepsMode")
        .value("Dep", FindDepsMode::Dep)
        .value("KillEarlier", FindDepsMode::KillEarlier)
        .value("KillLater", FindDepsMode::KillLater)
        .value("KillBoth", FindDepsMode::KillBoth);
    m.attr("DEP_WAW") = DEP_WAW;
    m.attr("DEP_WAR") = DEP_WAR;
    m.attr("DEP_RAW") = DEP_RAW;
    m.attr("DEP_ALL") = DEP_ALL;

    py::class_<NodeIDOrParallelScope>(m, "FindDepsAxis")
        .def(py::init<const ID &>(), "id"_a)
        .def(py::init<const ParallelScope &>(), "parallel"_a);

    py::class_<AccessPointSnapshot>(m, "AccessPointSnapshot")
        .def_readonly("op", &AccessPointSnapshot::op_)
        .def_readonly("stmt", &AccessPointSnapshot::stmt_)
        .def_readonly("def_stmt", &AccessPointSnapshot::def_)
        .def_readonly("var", &AccessPointSnapshot::var_)
        .def_readonly("def_id", &AccessPointSnapshot::defId_)
        .def_readonly("def_axis", &AccessPointSnapshot::defAxis_)
        .def_readonly("access", &AccessPointSnapshot::access_)
        .def_readonly("conds", &AccessPointSnapshot::conds_);
    py::class_<AxisDirection>(m, "AxisDirection")
        .def_readonly("is_node", &AxisDirection::isNode_)
        .def_readonly("id", &AxisDirection::id_)
        .def_readonly("parallel", &AxisDirection::parallel_)
        .def_readonly("direction", &AxisDirection::direction_);
    py::class_<DependenceSnapshot>(m, "DependenceSnapshot")
        .def_readonly("var", &DependenceSnapshot::var_)
        .def_readonly("later", &DependenceSnapshot::later_)
        .def_readonly("earlier", &DependenceSnapshot::earlier_)
        .def_readonly("direction", &DependenceSnapshot::direction_)
        .def_readonly("iter_dim", &DependenceSnapshot::iterDim_)
        .def_readonly("later_to_earlier",
                      &DependenceSnapshot::later2EarlierIter_)
        .def_readonly("later_iter_to_idx", &DependenceSnapshot::laterIter2Idx_)
        .def_readonly("earlier_iter_to_idx",
                      &DependenceSnapshot::earlierIter2Idx_)
        .def_readonly("later_to_earlier_all_possible",
                      &DependenceSnapshot::later2EarlierIterAllPossible_)
        .def_readonly("ext_constraint", &DependenceSnapshot::extConstraint_)
        .def_readonly("text", &DependenceSnapshot::text_);

    m.def(
        "find_deps",
        [](const Stmt &ast, DepType depType, FindDepsMode mode,
           const std::optional<std::vector<FindDepsDir>> &direction,
           const std::optional<ID> &filterSubAST, bool ignoreReductionWAW,
           bool eraseOutsideVarDef, bool noProjectOutPrivateAxis) {
            std::vector<DependenceSnapshot> results;
            auto finder = makeFindDeps(depType, mode, direction, filterSubAST,
                                       ignoreReductionWAW, eraseOutsideVarDef,
                                       noProjectOutPrivateAxis);
            finder(ast, syncFunc([&](const Dependence &dep) {
                       results.emplace_back(snapshotDependence(dep));
                   }));
            return results;
        },
        "ast"_a, "dep_type"_a = DEP_ALL, "mode"_a = FindDepsMode::Dep,
        "direction"_a = std::nullopt, "filter_sub_ast"_a = std::nullopt,
        "ignore_reduction_waw"_a = true, "erase_outside_vardef"_a = true,
        "no_project_out_private_axis"_a = false);
    m.def(
        "find_deps_exists",
        [](const Stmt &ast, DepType depType, FindDepsMode mode,
           const std::optional<std::vector<FindDepsDir>> &direction,
           const std::optional<ID> &filterSubAST, bool ignoreReductionWAW,
           bool eraseOutsideVarDef, bool noProjectOutPrivateAxis) {
            auto finder = makeFindDeps(depType, mode, direction, filterSubAST,
                                       ignoreReductionWAW, eraseOutsideVarDef,
                                       noProjectOutPrivateAxis);
            return finder.exists(ast);
        },
        "ast"_a, "dep_type"_a = DEP_ALL, "mode"_a = FindDepsMode::Dep,
        "direction"_a = std::nullopt, "filter_sub_ast"_a = std::nullopt,
        "ignore_reduction_waw"_a = true, "erase_outside_vardef"_a = true,
        "no_project_out_private_axis"_a = false);

    m.def("find_all_stmt",
          static_cast<std::vector<Stmt> (*)(const Stmt &, const ID &)>(
              &findAllStmt),
          "ast"_a, "id"_a);
    m.def("find_stmt",
          static_cast<Stmt (*)(const Stmt &, const ID &)>(&findStmt), "ast"_a,
          "id"_a);
    m.def("find_all_stmt",
          static_cast<std::vector<Stmt> (*)(
              const Stmt &, const std::function<bool(const Stmt &)> &filter)>(
              &findAllStmt),
          "ast"_a, "filter"_a);
    m.def("find_stmt",
          static_cast<Stmt (*)(
              const Stmt &, const std::function<bool(const Stmt &)> &filter)>(
              &findStmt),
          "ast"_a, "filter"_a);
    m.def("find_all_stmt",
          static_cast<std::vector<Stmt> (*)(
              const Stmt &, const Ref<Selector> &selector)>(&findAllStmt),
          "ast"_a, "selector"_a);
    m.def("find_stmt",
          static_cast<Stmt (*)(const Stmt &, const Ref<Selector> &selector)>(
              &findStmt),
          "ast"_a, "selector"_a);
    m.def("find_all_stmt",
          static_cast<std::vector<Stmt> (*)(const Func &, const ID &)>(
              &findAllStmt),
          "func"_a, "id"_a);
    m.def("find_stmt",
          static_cast<Stmt (*)(const Func &, const ID &)>(&findStmt), "func"_a,
          "id"_a);
    m.def("find_all_stmt",
          static_cast<std::vector<Stmt> (*)(
              const Func &, const std::function<bool(const Stmt &)> &filter)>(
              &findAllStmt),
          "func"_a, "filter"_a);
    m.def("find_stmt",
          static_cast<Stmt (*)(
              const Func &, const std::function<bool(const Stmt &)> &filter)>(
              &findStmt),
          "func"_a, "filter"_a);
    m.def("find_all_stmt",
          static_cast<std::vector<Stmt> (*)(
              const Func &, const Ref<Selector> &selector)>(&findAllStmt),
          "func"_a, "selector"_a);
    m.def("find_stmt",
          static_cast<Stmt (*)(const Func &, const Ref<Selector> &selector)>(
              &findStmt),
          "func"_a, "selector"_a);

    m.def("all_reads",
          static_cast<std::unordered_set<std::string> (*)(const AST &, bool,
                                                          bool)>(&allReads),
          "ast"_a, "no_recurse_idx"_a = false, "no_recurse_sub_stmt"_a = false);
    m.def("all_writes",
          static_cast<std::unordered_set<std::string> (*)(const AST &, bool,
                                                          bool)>(&allWrites),
          "ast"_a, "no_recurse_idx"_a = false, "no_recurse_sub_stmt"_a = false);
    m.def("all_iters",
          static_cast<std::unordered_set<std::string> (*)(const AST &, bool,
                                                          bool)>(&allIters),
          "ast"_a, "no_recurse_idx"_a = false, "no_recurse_sub_stmt"_a = false);
    m.def("all_names",
          static_cast<std::unordered_set<std::string> (*)(const AST &, bool,
                                                          bool)>(&allNames),
          "ast"_a, "no_recurse_idx"_a = false, "no_recurse_sub_stmt"_a = false);
}

} // namespace freetensor
