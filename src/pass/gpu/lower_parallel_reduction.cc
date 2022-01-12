#include <itertools.hpp>

#include <pass/gpu/lower_parallel_reduction.h>

namespace ir {

namespace gpu {

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

    if (op->len_->nodeType() != ASTNodeType::IntConst) {
        ERROR("Parallel reduction on a dynamic-lengthed loop is not "
              "supported yet");
    }
    auto len = op->len_.as<IntConstNode>()->val_;
    auto nth = makeSub(makeVar(op->iter_), op->begin_);

    // Note that we are before normalize_threads, so there will be no
    // cross-thread dependencies except the reduction we are working on.
    // Therefore, we don't have to immediately reduce the values at the ReduceTo
    // node, and we can deal with the reduction at the end of the parallel
    // scope.

    std::vector<std::string> workspaces;
    std::vector<std::vector<SubTree<ExprNode>>> workspaceShapes;
    std::vector<DataType> dtypes;
    for (size_t i = 0, n = op->property_.reductions_.size(); i < n; i++) {
        auto &[redOp, var, varIndices] = op->property_.reductions_[i];
        auto dtype = buffer(var)->tensor().dtype();
        auto workspace = "__reduce_" + op->id() + "_" + std::to_string(i);
        std::vector<SubTree<ExprNode>> workspaceShape;
        workspaceShape.emplace_back(op->len_);
        ASSERT(varIndices.size() == buffer(var)->tensor().shape().size());
        for (auto &&[idx, dim] :
             iter::zip(varIndices, buffer(var)->tensor().shape())) {
            if (!idx.isValid()) {
                workspaceShape.emplace_back(dim);
            }
        }

        std::vector<SubTree<ExprNode>> wIndices, wFirstIndices, flushVIndices;
        wIndices.emplace_back(nth);
        wFirstIndices.emplace_back(makeIntConst(0));
        for (size_t j = 0, m = workspaceShape.size(); j < m - 1; j++) {
            auto iter = makeVar(workspace + "." + std::to_string(j));
            wIndices.emplace_back(iter);
            wFirstIndices.emplace_back(iter);
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
                         makeLoad(workspace, wFirstIndices), false);

        // for (int k = 1; k < len; k <<= 1)
        //   if (nth % k == 0 && nth + k < len)
        //     workspace[nth] += workspace[nth + k]
        // where k = 2^p
        //   => 2^p < len
        //   => p < log_2 len
        //   => p < floor(log_2(len - 1)) + 1
        auto count = (63 - __builtin_clzll((unsigned long long)(len - 1))) + 1;
        auto k =
            makeIntrinsic("1 << (%)", {makeVar("__reduce_p")}, DataType::Int32);
        auto wNextIndices = wIndices;
        wNextIndices[0] = makeAdd(nth, k);
        auto reduceStmt =
            makeIf("",
                   makeLAnd(makeEQ(makeMod(nth, makeMul(k, makeIntConst(2))),
                                   makeIntConst(0)),
                            makeLT(makeAdd(nth, k), op->len_)),
                   makeReduceTo("", workspace, wIndices, redOp,
                                makeLoad(workspace, wNextIndices), false));
        reduceStmt =
            makeFor("", "__reduce_p", makeIntConst(0), makeIntConst(count),
                    makeIntConst(1), makeIntConst(count),
                    ForProperty().withUnroll(), std::move(reduceStmt));
        flushStmt = makeStmtSeq("", {reduceStmt, flushStmt});

        for (size_t j = workspaceShape.size() - 2; ~j; j--) {
            initStmt = makeFor("", workspace + "." + std::to_string(j),
                               makeIntConst(0), workspaceShape[j + 1],
                               makeIntConst(1), workspaceShape[j + 1],
                               ForProperty(), std::move(initStmt));
            flushStmt = makeFor("", workspace + "." + std::to_string(j),
                                makeIntConst(0), workspaceShape[j + 1],
                                makeIntConst(1), workspaceShape[j + 1],
                                ForProperty(), std::move(flushStmt));
        }

        op->body_ = makeStmtSeq("", {initStmt, op->body_, flushStmt});

        workspaces.emplace_back(std::move(workspace));
        workspaceShapes.emplace_back(std::move(workspaceShape));
        dtypes.emplace_back(dtype);
    }

    op->property_.reductions_.clear();
    Stmt ret = op;
    for (auto &&[workspace, wsShape, dtype] :
         iter::zip(workspaces, workspaceShapes, dtypes)) {
        ret = makeVarDef("", workspace,
                         Buffer(Tensor(wsShape, dtype), AccessType::Cache,
                                MemType::GPUShared),
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
        auto nth =
            makeSub(makeVar(redLoop.first->iter_), redLoop.first->begin_);
        std::vector<SubTree<ExprNode>> indices;
        indices.emplace_back(nth);
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

} // namespace gpu

} // namespace ir
