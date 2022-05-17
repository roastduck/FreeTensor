#include <autograd/grad.h>
#include <autograd/output_intermediates.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_autograd(py::module_ &m) {
    py::enum_<GradTapeMode>(m, "GradTapeMode")
        .value("All", GradTapeMode::All)
        .value("Nothing", GradTapeMode::Nothing)
        .value("NoReuseOnly", GradTapeMode::NoReuseOnly);

    m.def(
        "grad",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<ID, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &,
                const std::unordered_set<ID> &)>(&grad),
        "stmt"_a, "requires"_a, "provides"_a, "tapes"_a);
    m.def(
        "grad",
        static_cast<
            std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<ID, std::string>> (*)(
                const Stmt &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &,
                const std::unordered_set<ID> &)>(&grad),
        "func"_a, "requires"_a, "provides"_a, "tapes"_a);
    m.def(
        "grad",
        static_cast<
            std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<ID, std::string>> (*)(
                const Func &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, GradTapeMode)>(&grad),
        "stmt"_a, "requires"_a, "provides"_a,
        "tape_mode"_a = GradTapeMode::NoReuseOnly);
    m.def(
        "grad",
        static_cast<
            std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
                       std::unordered_map<std::string, std::string>,
                       std::unordered_map<ID, std::string>> (*)(
                const Stmt &, const std::unordered_set<std::string> &,
                const std::unordered_set<std::string> &, GradTapeMode)>(&grad),
        "func"_a, "requires"_a, "provides"_a,
        "tape_mode"_a = GradTapeMode::NoReuseOnly);

    // std::unordered_map<Load, Expr> cannot be exported to Python
    m.def(
        "output_intermediates",
        [](const Stmt &op, const std::unordered_set<ID> &intermediates) {
            return std::get<0>(outputIntermediates(op, intermediates));
        },
        "stmt"_a, "intermediates"_a);
}

} // namespace freetensor
