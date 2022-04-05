#include <itertools.hpp>

#include <hash.h>
#include <pass/cpu/lower_parallel_reduction.h>
#include <pass/make_nested_loops.h>
#include <pass/simplify.h>

namespace ir {

namespace cpu {

std::vector<std::pair<For, int>>
LowerParallelReduction::reducedBy(const ReduceTo &op) {
    std::vector<std::pair<For, int>> ret;
    for (auto &&loop : loopStack_) {
        for (auto &&[k, item] : iter::enumerate(loop->property_.reductions_)) {
            auto &&[redOp, var, begins, ends] = item;
            if (var == op->var_) {
                ret.emplace_back(loop, k);
                break;
            }
        }
    }
    return ret;
}

Stmt LowerParallelReduction::visit(const For &_op) {
    if (_op->property_.reductions_.empty()) {
        return BaseClass::visit(_op);
    }

    loopStack_.emplace_back(_op);
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    loopStack_.pop_back();

    std::vector<Stmt> initStmts, flushStmts;

    std::vector<std::string> workspaces;
    std::vector<std::vector<Expr>> workspaceShapes;
    std::vector<DataType> dtypes;
    for (size_t i = 0, n = op->property_.reductions_.size(); i < n; i++) {
        auto &[redOp, var, begins, ends] = op->property_.reductions_[i];
        auto dtype = buffer(var)->tensor().dtype();
        auto workspace =
            "__reduce_" + op->id().strId() + "_" + std::to_string(i);
        std::vector<Expr> workspaceShape;
        for (auto &&[begin, end] : iter::zip(begins, ends)) {
            workspaceShape.emplace_back(makeSub(end, begin));
        }

        std::vector<Expr> indices;
        for (size_t j = 0, m = workspaceShape.size(); j < m; j++) {
            indices.emplace_back(makeVar(workspace + "." + std::to_string(j)));
        }
        auto initStmt =
            makeStore("", workspace, indices, neutralVal(dtype, redOp));
        auto flushStmt = makeReduceTo(
            "", var,
            iter::imap([](auto &&x, auto &&y) { return makeAdd(x, y); }, begins,
                       indices),
            redOp, makeLoad(workspace, indices), false);
        initStmt = makeNestedLoops(
            indices, iter::repeat(makeIntConst(0)), workspaceShape,
            iter::repeat(makeIntConst(1)), workspaceShape,
            iter::repeat(ForProperty().withParallel(OpenMPScope{})), initStmt);
        flushStmt = makeNestedLoops(
            indices, iter::repeat(makeIntConst(0)), workspaceShape,
            iter::repeat(makeIntConst(1)), workspaceShape,
            iter::repeat(ForProperty().withParallel(OpenMPScope{})), flushStmt);
        initStmts.emplace_back(std::move(initStmt));
        flushStmts.emplace_back(std::move(flushStmt));

        // assign back to property_
        var = workspace;
        begins = std::vector<SubTree<ExprNode>>(begins.size(), makeIntConst(0));
        ends = std::vector<SubTree<ExprNode>>(workspaceShape.begin(),
                                              workspaceShape.end());

        workspaces.emplace_back(std::move(workspace));
        workspaceShapes.emplace_back(std::move(workspaceShape));
        dtypes.emplace_back(dtype);
    }

    std::vector<Stmt> stmts;
    stmts.insert(stmts.end(), initStmts.begin(), initStmts.end());
    stmts.emplace_back(op);
    stmts.insert(stmts.end(), flushStmts.begin(), flushStmts.end());
    Stmt ret = makeStmtSeq("", std::move(stmts));
    for (auto &&[workspace, wsShape, dtype] :
         iter::zip(workspaces, workspaceShapes, dtypes)) {
        ret = makeVarDef(
            "", workspace,
            Buffer(Tensor(wsShape, dtype), AccessType::Cache, MemType::CPU),
            nullptr, ret, false);
    }

    return ret;
}

Stmt LowerParallelReduction::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();

    if (op->atomic_) {
        return op;
    }

    auto redLoops = reducedBy(op);
    if (!redLoops.empty()) {
        if (redLoops.size() > 1) {
            ERROR(
                "Parallel reduction over multiple scopes is not supported yet");
        }
        auto &&redLoop = redLoops.front();
        auto workspace = "__reduce_" + redLoop.first->id().strId() + "_" +
                         std::to_string(redLoop.second);
        auto &&begins =
            redLoop.first->property_.reductions_[redLoop.second].begins_;
        ASSERT(op->indices_.size() == begins.size());
        return makeReduceTo(
            op->id(), workspace,
            iter::imap([](auto &&x, auto &&y) { return makeSub(x, y); },
                       op->indices_, begins),
            op->op_, op->expr_, false);
    }

    return op;
}

Stmt lowerParallelReduction(const Stmt &_op) {
    auto op = LowerParallelReduction()(_op);
    op = simplifyPass(op); // flatten singleton loops
    return op;
}

} // namespace cpu

} // namespace ir
