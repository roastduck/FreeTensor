#include <autograd/grad.h>
#include <autograd/output_intermediates.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_autograd(py::module_ &m) {
    py::class_<UserBwd>(m, "UserBwd")
        .def(py::init<const ID &, const ID &, const Stmt &>())
        .def_readonly("ori_begin", &UserBwd::oriBegin_)
        .def_readonly("ori_end", &UserBwd::oriEnd_)
        .def_readonly("bwd_body", &UserBwd::bwdBody_);

    py::enum_<GradTapeMode>(m, "GradTapeMode")
        .value("All", GradTapeMode::All)
        .value("Nothing", GradTapeMode::Nothing)
        .value("NoReuseOnly", GradTapeMode::NoReuseOnly);

    m.def(
        "grad_body",
        static_cast<
            std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<ID, std::string>> (*)(
                const Stmt &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &,
                const std::unordered_set<ID> &, const std::vector<UserBwd> &)>(
            &gradBody),
        "func"_a, "requires"_a, "provides"_a, "tapes"_a,
        "user_bwds"_a = std::vector<UserBwd>{});
    m.def(
        "grad_",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &,
                const std::unordered_set<ID> &, bool,
                const std::vector<UserBwd> &)>(&gradFuncInplace),
        "stmt"_a, "requires"_a, "provides"_a, "tapes"_a,
        "tape_in_closure"_a = true, "user_bwds"_a = std::vector<UserBwd>{});
    m.def(
        "grad",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &,
                const std::unordered_set<ID> &, bool,
                const std::vector<UserBwd> &)>(&gradFuncOutOfPlace),
        "stmt"_a, "requires"_a, "provides"_a, "tapes"_a,
        "tape_in_closure"_a = true, "user_bwds"_a = std::vector<UserBwd>{});

    m.def(
        "grad_body",
        static_cast<
            std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<ID, std::string>> (*)(
                const Stmt &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, GradTapeMode,
                const std::vector<UserBwd> &)>(&gradBody),
        "func"_a, "requires"_a, "provides"_a,
        "tape_mode"_a = GradTapeMode::NoReuseOnly,
        "user_bwds"_a = std::vector<UserBwd>{});
    m.def(
        "grad_",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, GradTapeMode, bool,
                const std::vector<UserBwd> &)>(&gradFuncInplace),
        "stmt"_a, "requires"_a, "provides"_a,
        "tape_mode"_a = GradTapeMode::NoReuseOnly, "tape_in_closure"_a = true,
        "user_bwds"_a = std::vector<UserBwd>{});
    m.def(
        "grad",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, GradTapeMode, bool,
                const std::vector<UserBwd> &)>(&gradFuncOutOfPlace),
        "stmt"_a, "requires"_a, "provides"_a,
        "tape_mode"_a = GradTapeMode::NoReuseOnly, "tape_in_closure"_a = true,
        "user_bwds"_a = std::vector<UserBwd>{});

    py::enum_<OutputIntermediatesStage>(m, "OutputIntermediatesStage")
        .value("Forward", OutputIntermediatesStage::Forward)
        .value("Backward", OutputIntermediatesStage::Backward);

    // std::unordered_map<Load, Expr> cannot be exported to Python
    m.def(
        "output_intermediates",
        [](const Stmt &op, const std::unordered_set<ID> &intermediates,
           OutputIntermediatesStage stage, const std::string &varSuffix) {
            return std::get<0>(
                outputIntermediates(op, intermediates, stage, varSuffix));
        },
        "stmt"_a, "intermediates"_a,
        "stage"_a = OutputIntermediatesStage::Forward,
        "var_suffix"_a = ".tape");
}

} // namespace freetensor
