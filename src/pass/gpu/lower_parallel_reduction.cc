#ifdef FT_WITH_CUDA

#include <container_utils.h>
#include <hash.h>
#include <pass/const_fold.h>
#include <pass/gpu/lower_parallel_reduction.h>
#include <pass/make_nested_loops.h>
#include <pass/simplify.h>

namespace freetensor {

namespace gpu {

namespace {

template <class T, class U> std::vector<T> asVec(U &&adaptor) {
    return std::vector<T>(adaptor.begin(), adaptor.end());
}

Expr makeCeilLog2(const Expr &_x) {
    // Suppose x is a non-negative integer
    auto x = constFold(_x);
    if (x->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(
            (63 - __builtin_clzll(
                      (unsigned long long)(x.as<IntConstNode>()->val_ - 1))) +
            1);
    }
    switch (x->dtype()) {
    case DataType::Int32:
        // Similar to __builtin_clz, defined in gpu_runtime.h
        return makeIntrinsic("((31 - clz((unsigned int)((%) - 1))) + 1)", {x},
                             DataType::Int32, false);
    case DataType::Int64:
        // Similar to __builtin_clzll, defined in gpu_runtime.h
        return makeIntrinsic(
            "((63 - clzll((unsigned long long)((%) - 1))) + 1)", {x},
            DataType::Int32, false); // clzll returns int
    default:
        ASSERT(false);
    }
}

} // namespace

std::vector<std::pair<For, int>>
LowerParallelReduction::reducedBy(const ReduceTo &op) {
    std::vector<std::pair<For, int>> ret;
    for (auto &&loop : loopStack_) {
        for (auto &&[k, item] :
             views::enumerate(loop->property_->reductions_)) {
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

    auto nth = makeSub(makeVar(op->iter_), op->begin_);

    // Note that we are before normalize_threads, so there will be no
    // cross-thread dependencies except the reduction we are working on.
    // Therefore, we don't have to immediately reduce the values at the ReduceTo
    // node, and we can deal with the reduction at the end of the parallel
    // scope.

    std::vector<std::string> workspaces;
    std::vector<std::vector<Expr>> workspaceShapes;
    std::vector<DataType> dtypes;
    for (size_t i = 0, n = op->property_->reductions_.size(); i < n; i++) {
        auto &&r = op->property_->reductions_[i];
        auto dtype = buffer(r->var_)->tensor()->dtype();
        auto workspace =
            "__reduce_" + toString(op->id()) + "_" + std::to_string(i);
        std::vector<Expr> shape;
        for (auto &&[begin, end] : views::zip(r->begins_, r->ends_)) {
            shape.emplace_back(makeSub(end, begin));
        }

        std::vector<Expr> indices;
        for (size_t j = 0, m = shape.size(); j < m; j++) {
            auto iter = makeVar(workspace + "." + std::to_string(j));
            indices.emplace_back(iter);
        }
        auto initStmt = makeStore(workspace, cat({nth}, indices),
                                  neutralVal(dtype, r->op_));
        auto flushStmt = makeReduceTo(
            r->var_,
            asVec<Expr>(views::zip_with(
                [](auto &&x, auto &&y) { return makeAdd(x, y); }, r->begins_,
                indices)),
            r->op_, makeLoad(workspace, cat({makeIntConst(0)}, indices), dtype),
            false);
        flushStmt = makeIf(makeEQ(nth, makeIntConst(0)), flushStmt);

        // for (int k = 1; k < len; k <<= 1)
        //   if (nth % k == 0 && nth + k < len)
        //     workspace[nth] += workspace[nth + k]
        // where k = 2^p
        //   => 2^p < len
        //   => p < log_2 len
        //   => p < floor(log_2(len - 1)) + 1
        auto count = makeCeilLog2(op->len_);
        auto k = makeIntrinsic("1 << (%)", {makeVar("__reduce_p")},
                               DataType::Int32, false);
        auto reduceStmt = makeIf(
            makeLAnd(makeEQ(makeMod(nth, makeMul(k, makeIntConst(2))),
                            makeIntConst(0)),
                     makeLT(makeAdd(nth, k), op->len_)),
            makeReduceTo(
                workspace, cat({nth}, indices), r->op_,
                makeLoad(workspace, cat({makeAdd(nth, k)}, indices), dtype),
                false));
        auto prop = Ref<ForProperty>::make();
        if (count->nodeType() == ASTNodeType::IntConst) {
            prop = prop->withUnroll();
        }
        reduceStmt =
            makeFor("__reduce_p", makeIntConst(0), count, makeIntConst(1),
                    count, prop, std::move(reduceStmt));
        flushStmt = makeStmtSeq({reduceStmt, flushStmt});

        initStmt =
            makeNestedLoops(indices, views::repeat(makeIntConst(0)), shape,
                            views::repeat(makeIntConst(1)), shape,
                            views::repeat(Ref<ForProperty>::make()), initStmt);
        flushStmt =
            makeNestedLoops(indices, views::repeat(makeIntConst(0)), shape,
                            views::repeat(makeIntConst(1)), shape,
                            views::repeat(Ref<ForProperty>::make()), flushStmt);

        op->body_ = makeStmtSeq({initStmt, op->body_, flushStmt});

        workspaces.emplace_back(std::move(workspace));
        workspaceShapes.emplace_back(cat({op->len_}, shape));
        dtypes.emplace_back(dtype);
    }

    op->property_->reductions_.clear();
    Stmt ret = op;
    for (auto &&[workspace, wsShape, dtype] :
         views::zip(workspaces, workspaceShapes, dtypes)) {
        ret = makeVarDef(workspace,
                         makeBuffer(makeTensor(wsShape, dtype),
                                    AccessType::Cache, MemType::GPUShared),
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
        auto workspace = "__reduce_" + toString(redLoop.first->id()) + "_" +
                         std::to_string(redLoop.second);
        auto nth =
            makeSub(makeVar(redLoop.first->iter_), redLoop.first->begin_);
        std::vector<Expr> indices;
        indices.emplace_back(nth);
        auto &&begins =
            redLoop.first->property_->reductions_[redLoop.second]->begins_;
        ASSERT(op->indices_.size() == begins.size());
        for (auto &&[begin, idx] : views::zip(begins, op->indices_)) {
            indices.emplace_back(makeSub(idx, begin));
        }
        return makeReduceTo(workspace, std::move(indices), op->op_, op->expr_,
                            false, op->metadata(), op->id());
    }

    return op;
}

Stmt lowerParallelReduction(const Stmt &_op) {
    auto op = LowerParallelReduction()(_op);
    op = simplify(op); // flatten singleton loops
    return op;
}

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA
