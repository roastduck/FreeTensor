#include <ffi.h>
#include <stmt.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_ast_stmt(py::module_ &m) {
    auto pyStmt = m.attr("Stmt").cast<py::class_<StmtNode, Stmt>>();
    pyStmt.def_property_readonly("id", &StmtNode::id)
        .def_property(
            "metadata", [](const Stmt &op) { return op->metadata(); },
            [](Stmt &op, const Metadata &md) { op->metadata() = md; })
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
        .def("parent_stmt", &StmtNode::parentStmtByFilter, "filter"_a)
        .def("prev_in_ctrlflow", &StmtNode::prevInCtrlFlow)
        .def("next_in_ctrlflow", &StmtNode::nextInCtrlFlow)
        .def("parent_ctrlflow", &StmtNode::parentCtrlFlow);

    py::class_<StmtSeqNode, StmtSeq>(m, "StmtSeq", pyStmt)
        .def_property_readonly(
            "stmts",
            [](const StmtSeq &op) -> std::vector<Stmt> { return op->stmts_; });
    py::class_<VarDefNode, VarDef>(m, "VarDef", pyStmt)
        .def_readonly("name", &VarDefNode::name_)
        .def_property_readonly(
            "buffer",
            [](const VarDef &op) -> Ref<Buffer> { return op->buffer_; })
        .def_readonly("view_of", &VarDefNode::viewOf_)
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
    py::class_<MarkVersionNode, MarkVersion>(m, "MarkVersion", pyStmt)
        .def_readonly("tape_name", &MarkVersionNode::tapeName_)
        .def_readonly("var", &MarkVersionNode::var_);

    // makers
    m.def("makeAny", []() { return makeAny(); });
    m.def(
        "makeStmtSeq",
        [](const std::vector<Stmt> &_1, const Metadata &_2, const ID &_3) {
            return makeStmtSeq(_1, _2, _3);
        },
        "stmts"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def(
        "makeVarDef",
        [](const std::string &_1, const Ref<Buffer> &_2,
           const std::optional<std::string> &_3, const Stmt &_4, bool _5,
           const Metadata &_6,
           const ID &_7) { return makeVarDef(_1, _2, _3, _4, _5, _6, _7); },
        "name"_a, "buffer"_a, "view_of"_a, "body"_a, "pinned"_a, "metadata"_a,
        py::arg_v("id", ID(), "ID()"));
    m.def(
        "makeStore",
        [](const std::string &_1, const std::vector<Expr> &_2, const Expr &_3,
           const Metadata &_4,
           const ID &_5) { return makeStore(_1, _2, _3, _4, _5); },
        "var"_a, "indices"_a, "expr"_a, "metadata"_a,
        py::arg_v("id", ID(), "ID()"));
    m.def(
        "makeReduceTo",
        [](const std::string &_1, const std::vector<Expr> &_2, ReduceOp _3,
           const Expr &_4, bool _5, const Metadata &_6,
           const ID &_7) { return makeReduceTo(_1, _2, _3, _4, _5, _6, _7); },
        "var"_a, "indices"_a, "op"_a, "expr"_a, "sync"_a, "metadata"_a,
        py::arg_v("id", ID(), "ID()"));
    m.def(
        "makeAlloc",
        [](const std::string &_1, const Metadata &_2, const ID &_3) {
            return makeAlloc(_1, _2, _3);
        },
        "var"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def(
        "makeFree",
        [](const std::string &_1, const Metadata &_2, const ID &_3) {
            return makeFree(_1, _2, _3);
        },
        "var"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def(
        "makeFor",
        [](const std::string &_1, const Expr &_2, const Expr &_3,
           const Expr &_4, const Expr &_5, const Ref<ForProperty> &_6,
           const Stmt &_7, const Metadata &_8, const ID &_9) {
            return makeFor(_1, _2, _3, _4, _5, _6, _7, _8, _9);
        },
        "iter"_a, "begin"_a, "end"_a, "step"_a, "len"_a, "property"_a, "body"_a,
        "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def(
        "makeIf",
        [](const Expr &_1, const Stmt &_2, const Stmt &_3, const Metadata &_4,
           const ID &_5) { return makeIf(_1, _2, _3, _4, _5); },
        "cond"_a, "thenCase"_a, "elseCase"_a, "metadata"_a,
        py::arg_v("id", ID(), "ID()"));
    m.def(
        "makeIf",
        [](const Expr &_1, const Stmt &_2, const Metadata &_3, const ID &_4) {
            return makeIf(_1, _2, _3, _4);
        },
        "cond"_a, "thenCase"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def(
        "makeAssert",
        [](const Expr &_1, const Stmt &_2, const Metadata &_3, const ID &_4) {
            return makeAssert(_1, _2, _3, _4);
        },
        "cond"_a, "body"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def(
        "makeEval",
        [](const Expr &_1, const Metadata &_2, const ID &_3) {
            return makeEval(_1, _2, _3);
        },
        "expr"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
    m.def(
        "makeMarkVersion",
        [](const std::string &_1, const std::string &_2, const Metadata &_3,
           const ID &_4) { return makeMarkVersion(_1, _2, _3, _4); },
        "tape_name"_a, "var"_a, "metadata"_a, py::arg_v("id", ID(), "ID()"));
}

} // namespace freetensor
