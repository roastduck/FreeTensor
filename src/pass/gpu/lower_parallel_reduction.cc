#ifdef FT_WITH_CUDA

#include <analyze/all_uses.h>
#include <container_utils.h>
#include <hash.h>
#include <pass/const_fold.h>
#include <pass/gpu/lower_parallel_reduction.h>
#include <pass/gpu/normalize_thread_dims.h>
#include <pass/make_nested_loops.h>
#include <pass/normalize_loops.h>
#include <pass/replace_iter.h>
#include <pass/shrink_var.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>

namespace freetensor {

namespace gpu {

namespace {

Expr makeCeilLog2(const Expr &_x) {
    // Suppose x is a non-negative integer
    auto x = constFold(_x);
    if (x->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(
            (63 - __builtin_clzll(
                      (unsigned long long)(x.as<IntConstNode>()->val_ - 1))) +
            1);
    }
    // TODO: Use std::bit_width after we have C++20 in CUDA
    switch (x->dtype().base()) {
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
InsertWorkspaces::reducedBy(const ReduceTo &op) {
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

Stmt InsertWorkspaces::visit(const VarDef &op) {
    auto ret = BaseClass::visit(op);
    handledVars_.erase(op->name_);
    return ret;
}

Stmt InsertWorkspaces::visit(const For &_op) {
    if (_op->property_->reductions_.empty()) {
        return BaseClass::visit(_op);
    }

    loopStack_.emplace_back(_op);
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    loopStack_.pop_back();

    std::vector<Ref<ReductionItem>> responsibleReductions;
    std::vector<Ref<ReductionItem>> nonResponsibleReductions;
    std::vector<std::string> workspaces;
    std::vector<std::vector<Expr>> workspaceShapes;
    std::vector<DataType> dtypes;
    for (auto &&r : op->property_->reductions_) {
        if (handledVars_.count(r->var_)) {
            nonResponsibleReductions.emplace_back(r);
            continue; // An inner scope is responsible for it
        } else {
            responsibleReductions.emplace_back(r);
        }

        std::vector<Expr> shape;
        shape.reserve(r->begins_.size());
        for (auto &&[begin, end] : views::zip(r->begins_, r->ends_)) {
            shape.emplace_back(makeSub(end, begin));
        }
        workspaces.emplace_back("__reduce_" + toString(op->id()) + "_" +
                                std::to_string(Hasher{}(r)));
        workspaceShapes.emplace_back(std::move(shape));
        dtypes.emplace_back(buffer(r->var_)->tensor()->dtype());
    }

    // VarDef nodes of the workspaces should be inserted INSIDE the parallel
    // loop, so it can be further sinked
    Stmt body = op->body_;
    for (auto &&[workspace, wsShape, dtype, red] : views::zip(
             workspaces, workspaceShapes, dtypes, responsibleReductions)) {
        body = makeVarDef(workspace,
                          makeBuffer(makeTensor(wsShape, dtype),
                                     AccessType::Cache, MemType::GPUShared),
                          std::nullopt, std::move(body), false);
        ws2red_[body->id()] = std::make_pair(op->iter_, red);
        handledVars_.insert(red->var_);
        converged_ = false;
    }

    op->body_ = std::move(body);
    op->property_->reductions_ = std::move(nonResponsibleReductions);
    return op;
}

Stmt InsertWorkspaces::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();

    if (op->sync_) {
        return op;
    }

    auto redLoops = reducedBy(op);
    if (!redLoops.empty()) {
        auto &&redLoop = redLoops.back(); // The most inner scope
        auto &&redItem = redLoop.first->property_->reductions_[redLoop.second];
        auto workspace = "__reduce_" + toString(redLoop.first->id()) + "_" +
                         std::to_string(Hasher{}(redItem));
        auto &&begins = redItem->begins_;
        ASSERT(op->indices_.size() == begins.size());
        std::vector<Expr> indices;
        indices.reserve(begins.size());
        for (auto &&[begin, idx] : views::zip(begins, op->indices_)) {
            indices.emplace_back(makeSub(idx, begin));
        }
        return makeReduceTo(workspace, std::move(indices), op->op_, op->expr_,
                            false, op->metadata(), op->id());
    }

    return op;
}

Expr InsertBinaryReduction::makeCondForNeighborThread(
    const std::string &thisThreadIter, const Expr &neighborThreadIter) {
    Expr ret;
    for (auto &&cond : condStack_) {
        if (allIters(cond).count(thisThreadIter)) {
            ReplaceIter replacer(thisThreadIter, neighborThreadIter);
            ret =
                ret.isValid() ? makeLAnd(ret, replacer(cond)) : replacer(cond);
        }
    }
    return ret;
}

Stmt InsertBinaryReduction::visit(const VarDef &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (auto it = ws2red_.find(op->id()); it != ws2red_.end()) {
        ws2scope_[op->id()] = op->body_->id();

        auto &&l = loop(it->second.first); // loop
        auto &&r = it->second.second;      // reduction
        auto dtype = op->buffer_->tensor()->dtype();
        auto &shape = op->buffer_->tensor()->shape();

        auto nth = makeSub(makeVar(l->iter_), l->begin_);

        std::vector<Expr> indices;
        for (size_t j = 0, m = shape.size(); j < m; j++) {
            auto iter = makeVar(op->name_ + "." + std::to_string(j));
            indices.emplace_back(iter);
        }
        auto initStmt = makeStore(op->name_, cat({nth}, indices),
                                  neutralVal(dtype, r->op_));
        auto flushStmt = makeReduceTo(
            r->var_,
            ranges::to<std::vector>(views::zip_with(
                [](auto &&x, auto &&y) { return makeAdd(x, y); }, r->begins_,
                indices)),
            r->op_, makeLoad(op->name_, cat({makeIntConst(0)}, indices), dtype),
            false);
        flushStmt = makeIf(makeEQ(nth, makeIntConst(0)), flushStmt);

        // for (int k = 1; k < len; k <<= 1)
        //   if (nth % k == 0 && nth + k < len)
        //     workspace[nth] += workspace[nth + k]
        // where k = 2^p
        //   => 2^p < len
        //   => p < log_2 len
        //   => p < floor(log_2(len - 1)) + 1
        auto count = makeCeilLog2(l->len_);
        Expr k;
        if (count->nodeType() == ASTNodeType::IntConst &&
            count.as<IntConstNode>()->val_ == 1) {
            // __reduce_p will always be 0. Manual const fold because we don't
            // have a node for "1 << x"
            k = makeIntConst(1);
        } else {
            k = makeIntrinsic("1 << (%)", {makeVar("__reduce_p")},
                              {DataType::Int32, SignDataType::GT0}, false);
        }
        auto reduceStmt = makeReduceTo(
            op->name_, cat({nth}, indices), r->op_,
            makeLoad(op->name_, cat({makeAdd(nth, k)}, indices), dtype), false);

        // Loops iterating the workspace is contiguous, while loops performing
        // binary reduction is not, so put the former inside and put the latter
        // outside
        auto makeNestLoopsIteratingWorkspace = [&](const Stmt &s) {
            return makeNestedLoops(indices, views::repeat(makeIntConst(0)),
                                   shape, views::repeat(makeIntConst(1)), shape,
                                   views::repeat(Ref<ForProperty>::make()), s);
        };
        initStmt = makeNestLoopsIteratingWorkspace(initStmt);
        reduceStmt = makeNestLoopsIteratingWorkspace(reduceStmt);
        flushStmt = makeNestLoopsIteratingWorkspace(flushStmt);

        auto prop = Ref<ForProperty>::make();
        if (count->nodeType() == ASTNodeType::IntConst) {
            prop = prop->withUnroll();
        }
        auto cond = makeLAnd(
            makeEQ(makeMod(nth, makeMul(k, makeIntConst(2))), makeIntConst(0)),
            makeLT(makeAdd(nth, k), l->len_));
        if (auto &&extraCond = makeCondForNeighborThread(
                l->iter_, makeAdd(makeVar(l->iter_) /* not `nth` */, k));
            extraCond.isValid()) {
            cond = makeLAnd(std::move(cond), std::move(extraCond));
        }
        reduceStmt = makeFor("__reduce_p", makeIntConst(0), count,
                             makeIntConst(1), count, prop,
                             makeIf(std::move(cond), std::move(reduceStmt)));

        op->body_ = makeStmtSeq({initStmt, op->body_, reduceStmt, flushStmt});

        // Insert the first dim after all uses of the old `shape`
        shape.insert(shape.begin(), l->len_);
    }
    return op;
}

Stmt InsertBinaryReduction::visit(const If &op) {
    auto cond = (*this)(op->cond_);
    condStack_.emplace_back(op->cond_);
    auto thenCase = (*this)(op->thenCase_);
    condStack_.pop_back();
    Stmt elseCase;
    if (op->elseCase_.isValid()) {
        condStack_.emplace_back(makeLNot(op->cond_));
        elseCase = (*this)(op->elseCase_);
        condStack_.pop_back();
    }
    return makeIf(std::move(cond), std::move(thenCase), std::move(elseCase),
                  op->metadata(), op->id(), op->debugBlame());
}

Stmt CorrectInterThreadDependence::visit(const VarDef &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (auto it = ws2red_.find(op->id()); it != ws2red_.end()) {
        loop2ws_[loop(it->second.first)->id()].emplace_back(op);
        return op->body_;
    } else {
        return op;
    }
}

Stmt CorrectInterThreadDependence::visit(const For &op) {
    auto ret = BaseClass::visit(op);
    if (auto it = loop2ws_.find(op->id()); it != loop2ws_.end()) {
        for (auto &&ws : it->second) {
            VarDef vardef = ws;

            CompUniqueBounds unique(*this);
            auto &&red = ws2red_.at(ws->id()).second;
            auto &shape = vardef->buffer_->tensor()->shape();
            for (auto &&[dim, oldBegin, oldEnd] :
                 views::zip(shape | views::slice(1ul, shape.size()),
                            red->begins_, red->ends_)) {
                for (auto &&name : allNames(dim)) {
                    if (!names().count(name)) {
                        Expr newDim;
                        for (auto &&b : unique.getDefinedUpper(
                                 makeMin(dim, makeSub(oldEnd, oldBegin)),
                                 names())) {
                            newDim = newDim.isValid()
                                         ? makeMin(std::move(newDim), b.expr())
                                         : b.expr();
                        }
                        ASSERT(newDim.isValid());
                        dim = std::move(newDim);
                        break;
                    }
                }
            }

            vardef->body_ = ret;
            ret = vardef;
        }
    }
    return ret;
}

Stmt lowerParallelReduction(const Stmt &_op) {
    auto op = _op;

    // 0. Pre-run pass/gpu/normalize_thread_dims here before
    // pass/gpu/normalize_threads, so our `O(log n)` reduction loop will be with
    // a legal `n`, where variables in `n` are all defined outside of kernels.
    op = normalizeLoops(op, [](const For &l) {
        return std::holds_alternative<CUDAScope>(l->property_->parallel_);
    });
    op = normalizeThreadDims(op);

    // For each parallel reduction, we don't reduce onto the target variable
    // directly, but first reduce into a thread-local workspace (implemented by
    // a thread-number-fold sized tensor), then do a binary reduction on the
    // workspace, and finally store the reduction result to the target variable.
    //
    // Because this pass is performed before `pass/normalize_threads`, and we
    // always check dependences when doing `schedule/paralleized`, there will be
    // no cross-thread dependences except the reduction we are working on.
    // Therefore, we don't have to immediately reduce the values at the
    // `ReduceTo` nodes. We can freely select where to do the reduction after
    // the `ReduceTo` nodes and before the end of the parallel `For` scope.
    // There is a trade-off: the earlier we do the reduction, there can be more
    // redundant reductions; the later we do the reduction, the workspace can be
    // larger and takes more shared memory. We use a simple criteria: using
    // `pass/sink_var` and `pass/shrink_var` to make the workspace scope as
    // small as possible (which means to put the reduction as early as possible)
    // as long as there is no dependence or loop-carried variance on the
    // workspace.
    //
    // For reducing over multiple scopes, we transform the code iteratively from
    // inner to outer.
    //
    // TODO: When reducing over multiple scopes and the scopes are prefectly
    // nested, we can do a global `O(log n)` binary reduction, where `n` is the
    // product of the lengths of all scopes.
    //
    // NOTE: Potentially we may use NVIDIA CUB for more efficient reduction
    // algorithms, but CUB comes with the following limitations:
    // - It requires the `blockDim` to be constant.
    // - It only works on scalars (but we can call it multiple times or we can
    // make a data type to wrap around an array slice).
    // - It only supports reducing along ALL `threadIdx`es, instead of some of
    // them.

    while (true) {
        // 1. Insert the workspace with the same size of the reduction target.
        InsertWorkspaces insertWorkspaces;
        op = insertWorkspaces(op);
        if (insertWorkspaces.converged()) {
            break;
        }

        // 2. Try to make the workspace more inner by `pass/sink_var`, but make
        // sure all participating threads are working so don't sink into some
        // branches
        op = sinkVar(op,
                     ranges::to<std::unordered_set>(
                         views::keys(insertWorkspaces.ws2red())),
                     [&](const Stmt &scope) {
                         // TODO: A more accurate filter that filters only
                         // variants of the loop representing the collaborative
                         // threads
                         return scope->nodeType() != ASTNodeType::If;
                     });

        // 3. Enlarge the workspace to thread-number-fold, and insert the binary
        // reduction algorithm.
        InsertBinaryReduction insertBinaryReduction(insertWorkspaces.ws2red());
        op = insertBinaryReduction(op);

        // 4. Try to make the workspace smaller by `pass/shrink_var`. Here we
        // use custom bounds only considering the real use of the workspaces
        std::unordered_map<ID, AccessBound> bounds;
        for (auto &&[wsId, scopeId] : insertBinaryReduction.ws2scope()) {
            bounds[wsId] = compAccessBound(op, wsId, COMP_ACCESS_BOUND_READ,
                                           false, scopeId);
        }
        op = ShrinkVar(bounds, true)(op);

        // 5. Simplify, to flatten singleton loops, and to simplify the
        // expressions from `pass/shrink_var`
        op = simplify(op);

        // 6. As per our definition of inter-thread dependence, a VarDef defined
        // inside a parallel For is considered thread local to it, while a
        // VarDef defined outside a parallel For is considered shared by all the
        // threads. The former will be further lower by
        // pass/gpu/multiplex_buffers. We need to put workspace back to the
        // original place to meet this definition, but this time with a shrinked
        // shape
        op = CorrectInterThreadDependence(insertWorkspaces.ws2red())(op);
    }

    return op;
}

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA
