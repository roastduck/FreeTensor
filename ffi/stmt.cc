#include <ffi.h>
#include <stmt.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_ast_stmt(py::module_ &m) {
    auto pyStmt = m.attr("Stmt").cast<py::class_<StmtNode, Stmt>>();
    pyStmt.def_property_readonly("id", &StmtNode::id)
        .def("node",
             [](const Stmt &op) {
                 WARNING("`x.node()` is deprecated. Please directly use `x`");
                 return op;
             })
        .def("prev",
             [](const Stmt &op) {
                 WARNING(
                     "`x.prev()` is deprecated. Please use `x.prev_stmt()`");
                 return op->prevStmt();
             })
        .def("next",
             [](const Stmt &op) {
                 WARNING(
                     "`x.next()` is deprecated. Please use `x.next_stmt()`");
                 return op->nextStmt();
             })
        .def("outer",
             [](const Stmt &op) {
                 WARNING(
                     "`x.outer()` is deprecated. Please use `x.parent_stmt()`");
                 return op->parentStmt();
             })
        .def("prev_stmt", &StmtNode::prevStmt)
        .def("next_stmt", &StmtNode::nextStmt)
        .def("parent_stmt", &StmtNode::parentStmt)
        .def("parent_stmt", &StmtNode::parentStmtByFilter, "filter"_a);

    py::class_<StmtSeqNode, StmtSeq>(m, "StmtSeq", pyStmt)
        .def_property_readonly(
            "stmts",
            [](const StmtSeq &op) -> std::vector<Stmt> { return op->stmts_; });
    py::class_<VarDefNode, VarDef>(m, "VarDef", pyStmt)
        .def_readonly("name", &VarDefNode::name_)
        .def_property_readonly(
            "buffer",
            [](const VarDef &op) -> Ref<Buffer> { return op->buffer_; })
        .def_property_readonly(
            "io_tensor",
            [](const VarDef &op) -> Ref<Tensor> { return op->ioTensor_; })
        .def_property_readonly(
            "body", [](const VarDef &op) -> Stmt { return op->body_; });
    py::class_<StoreNode, Store>(m, "Store", pyStmt)
        .def_readonly("var", &StoreNode::var_)
        .def_property_readonly(
            "indices",
            [](const Store &op) -> std::vector<Expr> { return op->indices_; })
        .def_property_readonly(
            "expr", [](const Store &op) -> Expr { return op->expr_; });
    py::class_<AllocNode, Alloc>(m, "Alloc", pyStmt)
        .def_readonly("var", &AllocNode::var_);
    py::class_<FreeNode, Free>(m, "Free", pyStmt)
        .def_readonly("var", &FreeNode::var_);
    py::enum_<ReduceOp>(m, "ReduceOp")
        .value("Add", ReduceOp::Add)
        .value("Sub", ReduceOp::Sub)
        .value("Max", ReduceOp::Max)
        .value("Min", ReduceOp::Min)
        .value("Mul", ReduceOp::Mul)
        .value("And", ReduceOp::LAnd)
        .value("Or", ReduceOp::LOr);
    py::class_<ReduceToNode, ReduceTo>(m, "ReduceTo", pyStmt)
        .def_readonly("var", &ReduceToNode::var_)
        .def_property_readonly("indices",
                               [](const ReduceTo &op) -> std::vector<Expr> {
                                   return op->indices_;
                               })
        .def_readonly("op", &ReduceToNode::op_)
        .def_property_readonly(
            "expr", [](const ReduceTo &op) -> Expr { return op->expr_; });
    py::class_<ForNode, For>(m, "For", pyStmt)
        .def_readonly("iter", &ForNode::iter_)
        .def_property_readonly("begin",
                               [](const For &op) -> Expr { return op->begin_; })
        .def_property_readonly("end",
                               [](const For &op) -> Expr { return op->end_; })
        .def_property_readonly("step",
                               [](const For &op) -> Expr { return op->step_; })
        .def_property_readonly("len",
                               [](const For &op) -> Expr { return op->len_; })
        .def_property_readonly(
            "property",
            [](const For &op) -> Ref<ForProperty> { return op->property_; })
        .def_property_readonly("body",
                               [](const For &op) -> Stmt { return op->body_; });
    py::class_<IfNode, If>(m, "If", pyStmt)
        .def_property_readonly("cond",
                               [](const If &op) -> Expr { return op->cond_; })
        .def_property_readonly(
            "then_case", [](const If &op) -> Stmt { return op->thenCase_; })
        .def_property_readonly(
            "else_case", [](const If &op) -> Stmt { return op->elseCase_; });
    py::class_<AssertNode, Assert>(m, "Assert", pyStmt)
        .def_property_readonly(
            "cond", [](const Assert &op) -> Expr { return op->cond_; })
        .def_property_readonly(
            "body", [](const Assert &op) -> Stmt { return op->body_; });
    py::class_<AssumeNode, Assume>(m, "Assume", pyStmt)
        .def_property_readonly(
            "cond", [](const Assume &op) -> Expr { return op->cond_; })
        .def_property_readonly(
            "body", [](const Assume &op) -> Stmt { return op->body_; });
    py::class_<EvalNode, Eval>(m, "Eval", pyStmt)
        .def_property_readonly(
            "expr", [](const Eval &op) -> Expr { return op->expr_; });
    py::class_<AnyNode, Any> pyAny(m, "Any", pyStmt);

    // makers
    m.def("makeAny", &_makeAny);
    m.def("makeStmtSeq",
          static_cast<Stmt (*)(const std::vector<Stmt> &, const Metadata &,
                               const ID &)>(&_makeStmtSeq),
          "stmts"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def("makeVarDef",
          static_cast<Stmt (*)(const std::string &, const Ref<Buffer> &,
                               const Ref<Tensor> &, const Stmt &, bool,
                               const Metadata &, const ID &)>(&_makeVarDef),
          "name"_a, "buffer"_a, "size_lim"_a, "body"_a, "pinned"_a,
          "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def("makeStore",
          static_cast<Stmt (*)(const std::string &, const std::vector<Expr> &,
                               const Expr &, const Metadata &, const ID &)>(
              &_makeStore<const Expr &>),
          "var"_a, "indices"_a, "expr"_a, "metadata"_a,
          py::arg_v("id", ID(), "ID()"));
    m.def("makeReduceTo",
          static_cast<Stmt (*)(const std::string &, const std::vector<Expr> &,
                               ReduceOp, const Expr &, bool, const Metadata &,
                               const ID &)>(&_makeReduceTo<const Expr &>),
          "var"_a, "indices"_a, "op"_a, "expr"_a, "atomic"_a, "metadata"_a,
          py::arg_v("id", ID(), "ID()"));
    m.def("makeAlloc",
          static_cast<Stmt (*)(const std::string &, const Metadata &,
                               const ID &)>(&_makeAlloc),
          "var"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def("makeFree",
          static_cast<Stmt (*)(const std::string &, const Metadata &,
                               const ID &)>(&_makeFree),
          "var"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def("makeFor",
          static_cast<Stmt (*)(const std::string &, const Expr &, const Expr &,
                               const Expr &, const Expr &,
                               const Ref<ForProperty> &, const Stmt &,
                               const Metadata &, const ID &)>(&_makeFor),
          "iter"_a, "begin"_a, "end"_a, "step"_a, "len"_a, "property"_a,
          "body"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def("makeIf",
          static_cast<Stmt (*)(const Expr &, const Stmt &, const Stmt &,
                               const Metadata &, const ID &)>(&_makeIf),
          "cond"_a, "thenCase"_a, "elseCase"_a, "metadata"_a,
          py::arg_v("id", ID(), "ID()"));
    m.def("makeIf",
          static_cast<Stmt (*)(const Expr &, const Stmt &, const Metadata &,
                               const ID &)>(&_makeIf),
          "cond"_a, "thenCase"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def("makeAssert",
          static_cast<Stmt (*)(const Expr &, const Stmt &, const Metadata &,
                               const ID &)>(&_makeAssert),
          "cond"_a, "body"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def("makeEval",
          static_cast<Stmt (*)(const Expr &, const Metadata &, const ID &)>(
              &_makeEval),
          "expr"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
}

} // namespace freetensor
