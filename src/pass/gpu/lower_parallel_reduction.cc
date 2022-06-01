#include <itertools.hpp>

#include <container_utils.h>
#include <hash.h>
#include <pass/gpu/lower_parallel_reduction.h>
#include <pass/make_nested_loops.h>
#include <pass/simplify.h>

namespace freetensor {

namespace gpu {

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
    std::vector<std::vector<Expr>> workspaceShapes;
    std::vector<DataType> dtypes;
    for (size_t i = 0, n = op->property_->reductions_.size(); i < n; i++) {
        auto &&r = op->property_->reductions_[i];
        auto dtype = buffer(r->var_)->tensor()->dtype();
        auto workspace =
            "__reduce_" + op->id().strId() + "_" + std::to_string(i);
        std::vector<Expr> shape;
        for (auto &&[begin, end] : iter::zip(r->begins_, r->ends_)) {
            shape.emplace_back(makeSub(end, begin));
        }

        std::vector<Expr> indices;
        for (size_t j = 0, m = shape.size(); j < m; j++) {
            auto iter = makeVar(workspace + "." + std::to_string(j));
            indices.emplace_back(iter);
        }
        auto initStmt = makeStore("", workspace, cat({nth}, indices),
                                  neutralVal(dtype, r->op_));
        auto flushStmt = makeReduceTo(
            "", r->var_,
            asVec<Expr>(
                iter::imap([](auto &&x, auto &&y) { return makeAdd(x, y); },
                           r->begins_, indices)),
            r->op_, makeLoad(workspace, cat({makeIntConst(0)}, indices), dtype),
            false);
        flushStmt = makeIf("", makeEQ(nth, makeIntConst(0)), flushStmt);

        // for (int k = 1; k < len; k <<= 1)
        //   if (nth % k == 0 && nth + k < len)
        //     workspace[nth] += workspace[nth + k]
        // where k = 2^p
        //   => 2^p < len
        //   => p < log_2 len
        //   => p < floor(log_2(len - 1)) + 1
        auto count = (63 - __builtin_clzll((unsigned long long)(len - 1))) + 1;
        auto k = makeIntrinsic("1 << (%)", {makeVar("__reduce_p")},
                               DataType::Int32, false);
        auto reduceStmt = makeIf(
            "",
            makeLAnd(makeEQ(makeMod(nth, makeMul(k, makeIntConst(2))),
                            makeIntConst(0)),
                     makeLT(makeAdd(nth, k), op->len_)),
            makeReduceTo(
                "", workspace, cat({nth}, indices), r->op_,
                makeLoad(workspace, cat({makeAdd(nth, k)}, indices), dtype),
                false));
        reduceStmt = makeFor(
            "", "__reduce_p", makeIntConst(0), makeIntConst(count),
            makeIntConst(1), makeIntConst(count),
            Ref<ForProperty>::make()->withUnroll(), std::move(reduceStmt));
        flushStmt = makeStmtSeq("", {reduceStmt, flushStmt});

        initStmt =
            makeNestedLoops(indices, iter::repeat(makeIntConst(0)), shape,
                            iter::repeat(makeIntConst(1)), shape,
                            iter::repeat(Ref<ForProperty>::make()), initStmt);
        flushStmt =
            makeNestedLoops(indices, iter::repeat(makeIntConst(0)), shape,
                            iter::repeat(makeIntConst(1)), shape,
                            iter::repeat(Ref<ForProperty>::make()), flushStmt);

        op->body_ = makeStmtSeq("", {initStmt, op->body_, flushStmt});

        workspaces.emplace_back(std::move(workspace));
        workspaceShapes.emplace_back(cat({op->len_}, shape));
        dtypes.emplace_back(dtype);
    }

    op->property_->reductions_.clear();
    Stmt ret = op;
    for (auto &&[workspace, wsShape, dtype] :
         iter::zip(workspaces, workspaceShapes, dtypes)) {
        ret = makeVarDef("", workspace,
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
        auto workspace = "__reduce_" + redLoop.first->id().strId() + "_" +
                         std::to_string(redLoop.second);
        auto nth =
            makeSub(makeVar(redLoop.first->iter_), redLoop.first->begin_);
        std::vector<Expr> indices;
        indices.emplace_back(nth);
        auto &&begins =
            redLoop.first->property_->reductions_[redLoop.second]->begins_;
        ASSERT(op->indices_.size() == begins.size());
        for (auto &&[begin, idx] : iter::zip(begins, op->indices_)) {
            indices.emplace_back(makeSub(idx, begin));
        }
        return makeReduceTo(op->id(), workspace, std::move(indices), op->op_,
                            op->expr_, false);
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
