#include <itertools.hpp>

#include <pass/cpu/lower_parallel_reduction.h>

namespace ir {

namespace cpu {

uint64_t LowerParallelReduction::getHash(const Expr &op) {
    getHash_(op);
    return getHash_.hash().at(op);
}

std::vector<std::pair<For, int>>
LowerParallelReduction::reducedBy(const ReduceTo &op) {
    std::vector<std::pair<For, int>> ret;
    for (auto &&loop : loopStack_) {
        for (auto &&[k, item] : iter::enumerate(loop->property_.reductions_)) {
            if (item.var_ == op->var_) {
                ASSERT(item.indices_.size() == op->indices_.size());
                for (auto &&[lIdx, oIdx] :
                     iter::zip(item.indices_, op->indices_)) {
                    if (lIdx.isValid() && getHash(lIdx) != getHash(oIdx)) {
                        goto mismatch;
                    }
                }
                ret.emplace_back(loop, k);
            }
        mismatch:;
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
    std::vector<std::vector<SubTree<ExprNode>>> workspaceShapes;
    std::vector<DataType> dtypes;
    for (size_t i = 0, n = op->property_.reductions_.size(); i < n; i++) {
        auto &[redOp, var, varIndices] = op->property_.reductions_[i];
        auto dtype = buffer(var)->tensor().dtype();
        auto workspace = "__reduce_" + op->id() + "_" + std::to_string(i);
        std::vector<SubTree<ExprNode>> workspaceShape;
        ASSERT(varIndices.size() == buffer(var)->tensor().shape().size());
        for (auto &&[idx, dim] :
             iter::zip(varIndices, buffer(var)->tensor().shape())) {
            if (!idx.isValid()) {
                workspaceShape.emplace_back(dim);
            }
        }

        std::vector<SubTree<ExprNode>> wIndices, flushVIndices;
        for (size_t j = 0, m = workspaceShape.size(); j < m; j++) {
            wIndices.emplace_back(makeVar(workspace + "." + std::to_string(j)));
        }
        for (size_t j = 0, m = varIndices.size(), k = 0; j < m; j++) {
            if (varIndices[j].isValid()) {
                flushVIndices.emplace_back(varIndices[j]);
            } else {
                auto iter = makeVar(workspace + "." + std::to_string(k++));
                flushVIndices.emplace_back(iter);
            }
        }
        auto initStmt =
            makeStore("", workspace, wIndices, neutralVal(dtype, redOp));
        auto flushStmt =
            makeReduceTo("", var, std::move(flushVIndices), redOp,
                         makeLoad(workspace, std::move(wIndices)), false);
        for (size_t j = workspaceShape.size() - 1; ~j; j--) {
            initStmt =
                makeFor("", workspace + "." + std::to_string(j),
                        makeIntConst(0), workspaceShape[j], makeIntConst(1),
                        workspaceShape[j], ForProperty(), std::move(initStmt));
            flushStmt =
                makeFor("", workspace + "." + std::to_string(j),
                        makeIntConst(0), workspaceShape[j], makeIntConst(1),
                        workspaceShape[j], ForProperty(), std::move(flushStmt));
        }

        initStmts.emplace_back(std::move(initStmt));
        flushStmts.emplace_back(std::move(flushStmt));

        std::vector<SubTree<ExprNode, Nullable>> workspaceIndices;
        for (auto &&idx : varIndices) {
            if (!idx.isValid()) {
                workspaceIndices.emplace_back(idx);
            }
        }
        var = workspace;
        varIndices = std::move(std::move(workspaceIndices));

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

    auto redLoops = reducedBy(op);
    if (!redLoops.empty()) {
        if (redLoops.size() > 1) {
            ERROR(
                "Parallel reduction over multiple scopes is not supported yet");
        }
        auto &&redLoop = redLoops.front();
        auto workspace = "__reduce_" + redLoop.first->id() + "_" +
                         std::to_string(redLoop.second);
        std::vector<SubTree<ExprNode>> indices;
        auto &&redIndices =
            redLoop.first->property_.reductions_[redLoop.second].indices_;
        ASSERT(op->indices_.size() == redIndices.size());
        for (auto &&[lIdx, oIdx] : iter::zip(redIndices, op->indices_)) {
            if (!lIdx.isValid()) {
                indices.emplace_back(oIdx);
            }
        }
        return makeReduceTo(op->id(), workspace, std::move(indices), op->op_,
                            op->expr_, false);
    }

    return op;
}

} // namespace cpu

} // namespace ir
