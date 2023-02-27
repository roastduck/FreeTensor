#include <autograd/grad.h>
#include <autograd/output_intermediates.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_autograd(py::module_ &m) {
    py::class_<RangeToUserGrad>(m, "RangeToUserGrad")
        .def(py::init<const ID &, const ID &, const Stmt &>())
        .def_readonly("ori_begin", &RangeToUserGrad::oriBegin_)
        .def_readonly("ori_end", &RangeToUserGrad::oriEnd_)
        .def_readonly("bwd_body", &RangeToUserGrad::bwdBody_)
        .def("__str__",
             [](const RangeToUserGrad &userGrad) {
                 return "Backward of statements from " +
                        toString(userGrad.oriBegin_) + " to " +
                        toString(userGrad.oriEnd_) + " is " +
                        toString(userGrad.bwdBody_);
             })
        .def("__repr__", [](const RangeToUserGrad &userGrad) {
            return "<RangeToUserGrad " + toString(userGrad.oriBegin_) + " " +
                   toString(userGrad.oriEnd_) + " " +
                   toString(userGrad.bwdBody_) + ">";
        });

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
                const std::unordered_set<ID> &,
                const std::vector<RangeToUserGrad> &)>(&gradBody),
        "func"_a, "requires"_a, "provides"_a, "tapes"_a,
        "user_grads"_a = std::vector<RangeToUserGrad>{});
    m.def(
        "grad_",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &,
                const std::unordered_set<ID> &, bool,
                const std::vector<RangeToUserGrad> &)>(&gradFuncInplace),
        "stmt"_a, "requires"_a, "provides"_a, "tapes"_a,
        "tape_in_closure"_a = true,
        "user_grads"_a = std::vector<RangeToUserGrad>{});
    m.def(
        "grad",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &,
                const std::unordered_set<ID> &, bool,
                const std::vector<RangeToUserGrad> &)>(&gradFuncOutOfPlace),
        "stmt"_a, "requires"_a, "provides"_a, "tapes"_a,
        "tape_in_closure"_a = true,
        "user_grads"_a = std::vector<RangeToUserGrad>{});

    m.def(
        "grad_body",
        static_cast<
            std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<ID, std::string>> (*)(
                const Stmt &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, GradTapeMode,
                const std::vector<RangeToUserGrad> &)>(&gradBody),
        "func"_a, "requires"_a, "provides"_a,
        "tape_mode"_a = GradTapeMode::NoReuseOnly,
        "user_grads"_a = std::vector<RangeToUserGrad>{});
    m.def(
        "grad_",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, GradTapeMode, bool,
                const std::vector<RangeToUserGrad> &)>(&gradFuncInplace),
        "stmt"_a, "requires"_a, "provides"_a,
        "tape_mode"_a = GradTapeMode::NoReuseOnly, "tape_in_closure"_a = true,
        "user_grads"_a = std::vector<RangeToUserGrad>{});
    m.def(
        "grad",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, GradTapeMode, bool,
                const std::vector<RangeToUserGrad> &)>(&gradFuncOutOfPlace),
        "stmt"_a, "requires"_a, "provides"_a,
        "tape_mode"_a = GradTapeMode::NoReuseOnly, "tape_in_closure"_a = true,
        "user_grads"_a = std::vector<RangeToUserGrad>{});

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
