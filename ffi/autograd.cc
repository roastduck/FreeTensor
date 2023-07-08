#include <autograd/grad.h>
#include <autograd/output_intermediates.h>
#include <autograd/tape_strategy.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_autograd(py::module_ &m) {
    py::class_<StmtSetToUserGrad>(m, "StmtSetToUserGrad")
        .def(py::init<const std::unordered_set<ID> &, const Stmt &>())
        .def_readonly("ori_stmts", &StmtSetToUserGrad::oriStmts_)
        .def_readonly("bwd_body", &StmtSetToUserGrad::bwdBody_)
        .def("__str__",
             [](const StmtSetToUserGrad &userGrad) {
                 return "Backward of statements in {" +
                        toString(userGrad.oriStmts_) + "} is " +
                        toString(userGrad.bwdBody_);
             })
        .def("__repr__", [](const StmtSetToUserGrad &userGrad) {
            return "<StmtSeqToUserGrad {" + toString(userGrad.oriStmts_) +
                   "} " + toString(userGrad.bwdBody_) + ">";
        });

    py::enum_<GradTapeMode>(m, "GradTapeMode")
        .value("All", GradTapeMode::All)
        .value("Nothing", GradTapeMode::Nothing)
        .value("NoReuseOnly", GradTapeMode::NoReuseOnly);

    py::class_<TapeStrategy>(m, "TapeStrategy")
        .def(py::init<const std::unordered_set<
                 std::variant<ID, std::string, Ref<Selector>>> &>())
        .def(py::init<const std::vector<
                 std::variant<ID, std::string, Ref<Selector>>> &>())
        .def(py::init<GradTapeMode>())
        .def("always_tape",
             static_cast<TapeStrategy (TapeStrategy::*)(
                 const std::unordered_set<
                     std::variant<ID, std::string, Ref<Selector>>> &)>(
                 &TapeStrategy::alwaysTape))
        .def("always_tape",
             static_cast<TapeStrategy (TapeStrategy::*)(
                 const std::vector<std::variant<ID, std::string, Ref<Selector>>>
                     &)>(&TapeStrategy::alwaysTape))
        .def("never_tape",
             static_cast<TapeStrategy (TapeStrategy::*)(
                 const std::unordered_set<
                     std::variant<ID, std::string, Ref<Selector>>> &)>(
                 &TapeStrategy::neverTape))
        .def("never_tape",
             static_cast<TapeStrategy (TapeStrategy::*)(
                 const std::vector<std::variant<ID, std::string, Ref<Selector>>>
                     &)>(&TapeStrategy::neverTape));
    py::implicitly_convertible<
        std::unordered_set<std::variant<ID, std::string, Ref<Selector>>>,
        TapeStrategy>();
    py::implicitly_convertible<
        std::vector<std::variant<ID, std::string, Ref<Selector>>>,
        TapeStrategy>();
    py::implicitly_convertible<GradTapeMode, TapeStrategy>();

    m.def(
        "grad_body",
        static_cast<
            std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<ID, std::string>> (*)(
                const Stmt &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, const TapeStrategy &,
                bool, bool, const std::vector<StmtSetToUserGrad> &)>(&gradBody),
        "func"_a, "requires"_a, "provides"_a,
        "tapes"_a = GradTapeMode::NoReuseOnly, "reset_provided_grad"_a = true,
        "invert"_a = true, "user_grads"_a = std::vector<StmtSetToUserGrad>{});
    m.def(
        "grad_",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, const TapeStrategy &,
                bool, bool, bool, const std::vector<StmtSetToUserGrad> &)>(
            &gradFuncInplace),
        "stmt"_a, "requires"_a, "provides"_a,
        "tapes"_a = GradTapeMode::NoReuseOnly, "tape_in_closure"_a = true,
        "reset_provided_grad"_a = true, "invert"_a = true,
        "user_grads"_a = std::vector<StmtSetToUserGrad>{});
    m.def(
        "grad",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, const TapeStrategy &,
                bool, bool, bool, const std::vector<StmtSetToUserGrad> &)>(
            &gradFuncOutOfPlace),
        "stmt"_a, "requires"_a, "provides"_a,
        "tapes"_a = GradTapeMode::NoReuseOnly, "tape_in_closure"_a = true,
        "reset_provided_grad"_a = true, "invert"_a = true,
        "user_grads"_a = std::vector<StmtSetToUserGrad>{});

    py::enum_<OutputIntermediatesStage>(m, "OutputIntermediatesStage")
        .value("Forward", OutputIntermediatesStage::Forward)
        .value("Backward", OutputIntermediatesStage::Backward);

    // This FFI binding is only for testing, so it is incomplete
    //
    // - std::unordered_map<Load, Expr> cannot be exported to Python
    // - The `derivatives` parameter is omitted, so the gradient of `y = f(x)`
    // w.r.t. `y` will not be output unless y is used elsewhere
    m.def("output_all_intermediates", outputAllIntermedaites, "stmt"_a,
          "intermediates"_a, "stage"_a = OutputIntermediatesStage::Forward,
          "var_suffix"_a = ".tape");
}

} // namespace freetensor
