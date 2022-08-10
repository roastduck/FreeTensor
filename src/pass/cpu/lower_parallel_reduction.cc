#include <itertools.hpp>

#include <hash.h>
#include <pass/cpu/lower_parallel_reduction.h>
#include <pass/make_nested_loops.h>
#include <pass/simplify.h>

namespace freetensor {

namespace cpu {

namespace {

template <class T, class U> std::vector<T> asVec(U &&adaptor) {
    return std::vector<T>(adaptor.begin(), adaptor.end());
}

} // namespace

std::vector<std::pair<For, int>>
LowerParallelReduction::reducedBy(const ReduceTo &op) {
    std::vector<std::pair<For, int>> ret;
    for (auto &&loop : loopStack_) {
        for (auto &&[k, item] : iter::enumerate(loop->property_->reductions_)) {
            if (item->var_ == op->var_) {
                ret.emplace_back(loop, k);
                break;
            }
        }
    }
    return ret;
}

Stmt LowerParallelReduction::visit(const For &_op) {
    if (_op->property_->reductions_.empty()) {
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
    for (size_t i = 0, n = op->property_->reductions_.size(); i < n; i++) {
        auto &&r = op->property_->reductions_[i];
        auto dtype = buffer(r->var_)->tensor()->dtype();
        auto workspace =
            "__reduce_" + op->id().strId() + "_" + std::to_string(i);
        std::vector<Expr> workspaceShape;
        for (auto &&[begin, end] : iter::zip(r->begins_, r->ends_)) {
            workspaceShape.emplace_back(makeSub(end, begin));
        }

        std::vector<Expr> indices;
        for (size_t j = 0, m = workspaceShape.size(); j < m; j++) {
            indices.emplace_back(makeVar(workspace + "." + std::to_string(j)));
        }
        auto initStmt =
            makeStore(workspace, indices, neutralVal(dtype, r->op_));
        auto flushStmt =
            makeReduceTo(r->var_,
                         asVec<Expr>(iter::imap(
                             [](auto &&x, auto &&y) { return makeAdd(x, y); },
                             r->begins_, indices)),
                         r->op_, makeLoad(workspace, indices, dtype), false);
        initStmt = makeNestedLoops(
            indices, iter::repeat(makeIntConst(0)), workspaceShape,
            iter::repeat(makeIntConst(1)), workspaceShape,
            iter::repeat(Ref<ForProperty>::make()->withParallel(OpenMPScope{})),
            initStmt);
        flushStmt = makeNestedLoops(
            indices, iter::repeat(makeIntConst(0)), workspaceShape,
            iter::repeat(makeIntConst(1)), workspaceShape,
            iter::repeat(Ref<ForProperty>::make()->withParallel(OpenMPScope{})),
            flushStmt);
        initStmts.emplace_back(std::move(initStmt));
        flushStmts.emplace_back(std::move(flushStmt));

        // assign back to property_
        r->var_ = workspace;
        r->begins_ = std::vector<Expr>(r->begins_.size(), makeIntConst(0));
        r->ends_ = workspaceShape;

        workspaces.emplace_back(std::move(workspace));
        workspaceShapes.emplace_back(std::move(workspaceShape));
        dtypes.emplace_back(dtype);
    }

    std::vector<Stmt> stmts;
    stmts.insert(stmts.end(), initStmts.begin(), initStmts.end());
    stmts.emplace_back(op);
    stmts.insert(stmts.end(), flushStmts.begin(), flushStmts.end());
    Stmt ret = makeStmtSeq(std::move(stmts));
    for (auto &&[workspace, wsShape, dtype] :
         iter::zip(workspaces, workspaceShapes, dtypes)) {
        ret = makeVarDef(workspace,
                         makeBuffer(makeTensor(wsShape, dtype),
                                    AccessType::Cache, MemType::CPU),
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
            redLoop.first->property_->reductions_[redLoop.second]->begins_;
        ASSERT(op->indices_.size() == begins.size());
        return makeReduceTo(
            workspace,
            asVec<Expr>(
                iter::imap([](auto &&x, auto &&y) { return makeSub(x, y); },
                           op->indices_, begins)),
            op->op_, op->expr_, false, op->metadata(), op->id());
    }

    return op;
}

Stmt lowerParallelReduction(const Stmt &_op) {
    auto op = LowerParallelReduction()(_op);
    op = simplify(op); // flatten singleton loops
    return op;
}

} // namespace cpu

} // namespace freetensor
