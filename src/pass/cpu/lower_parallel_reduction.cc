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
        for (size_t k = 0, m = loop->property_.reductions_.size(); k < m; k++) {
            auto &&item = loop->property_.reductions_[k];
            if (item.var_ == op->var_) {
                ASSERT(item.indices_.size() == op->indices_.size());
                for (size_t i = 0, n = item.indices_.size(); i < n; i++) {
                    if (item.indices_[i].isValid() &&
                        getHash(item.indices_[i]) != getHash(op->indices_[i])) {
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

Stmt LowerParallelReduction::visit(const VarDef &op) {
    ASSERT(!buffers_.count(op->name_));
    buffers_[op->name_] = op->buffer_;
    auto ret = Mutator::visit(op);
    buffers_.erase(op->name_);
    return ret;
}

Stmt LowerParallelReduction::visit(const For &_op) {
    if (_op->property_.reductions_.empty()) {
        return Mutator::visit(_op);
    }

    loopStack_.emplace_back(_op);
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    loopStack_.pop_back();

    std::vector<Stmt> initStmts, flushStmts;

    std::vector<std::string> workspaces;
    std::vector<std::vector<SubTree<ExprNode>>> workspaceShapes;
    std::vector<DataType> dtypes;
    for (size_t i = 0, n = op->property_.reductions_.size(); i < n; i++) {
        auto &[redOp, var, varIndices] = op->property_.reductions_[i];
        auto dtype = buffers_.at(var)->tensor().dtype();
        auto workspace = "__reduce_" + op->id() + "_" + std::to_string(i);
        std::vector<SubTree<ExprNode>> workspaceShape;
        ASSERT(varIndices.size() == buffers_.at(var)->tensor().shape().size());
        for (size_t j = 0, m = varIndices.size(); j < m; j++) {
            if (!varIndices[j].isValid()) {
                workspaceShape.emplace_back(
                    buffers_.at(var)->tensor().shape()[j]);
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
        ASSERT(varIndices.size() == buffers_.at(var)->tensor().shape().size());
        for (size_t j = 0, m = varIndices.size(); j < m; j++) {
            if (!varIndices[j].isValid()) {
                workspaceIndices.emplace_back(varIndices[j]);
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
    for (size_t i = 0, n = op->property_.reductions_.size(); i < n; i++) {
        ret = makeVarDef("", workspaces[i],
                         Buffer(Tensor(workspaceShapes[i], dtypes[i]),
                                AccessType::Cache, MemType::CPU),
                         nullptr, ret, false);
    }

    return ret;
}

Stmt LowerParallelReduction::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
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
        for (size_t i = 0, n = op->indices_.size(); i < n; i++) {
            if (!redIndices[i].isValid()) {
                indices.emplace_back(op->indices_[i]);
            }
        }
        return makeReduceTo(op->id(), workspace, std::move(indices), op->op_,
                            op->expr_, false);
    }

    return op;
}

} // namespace cpu

} // namespace ir
