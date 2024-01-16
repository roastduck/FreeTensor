#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <analyze/all_uses.h>
#include <analyze/comp_unique_bounds_pb.h>
#include <analyze/find_stmt.h>
#include <container_utils.h>
#include <get_new_name.h>
#include <math/min_max.h>
#include <math/parse_pb_expr.h>
#include <pass/normalize_loops.h>
#include <pass/replace_iter.h>
#include <pass/shrink_for.h>
#include <pass/simplify.h>
#include <pass/z3_simplify.h>

namespace freetensor {

namespace {

template <PBSetRef T>
PBSet moveDimToNamedParam(const PBCtx &ctx, T &&set, int dim,
                          const std::string &param) {
    // A name is required for the parameter, so we can't simply use
    // isl_set_move_dims. We constuct a map to apply on the set to move the
    // dimension. Example map: [p] -> {[_1, _2, p] -> [_1, _2]}

    int nDims = set.nDims();
    std::ostringstream os;
    os << "[" << param << "] -> {["
       << (views::ints(0, nDims) | views::transform([&](int i) {
               return i == dim ? param : "_" + std::to_string(i);
           }) |
           join(","))
       << "] -> ["
       << (views::ints(0, nDims) |
           views::filter([&](int i) { return i != dim; }) |
           views::transform([](int i) { return "_" + std::to_string(i); }) |
           join(","))
       << "]}";
    PBMap map(ctx, os.str());
    return apply(std::forward<T>(set), std::move(map));
}

class CompUniqueBoundsPBWithStride : public CompUniqueBoundsPB {
  private:
    std::pair<int64_t /* modulo */, Expr /* offset */>
    getStride(const Ref<CompUniqueBoundsPB::Bound> &bound, bool requireConst) {
        isl_stride_info *info = isl_set_get_stride_info(bound->bound_.get(), 0);
        auto stride = PBVal(isl_stride_info_get_stride(info));
        auto offset = PBSingleFunc(isl_stride_info_get_offset(info));
        isl_stride_info_free(info);
        ASSERT(stride.denSi() == 1);
        auto strideInt = stride.numSi();
        ReplaceIter demangler(*bound->demangleMap_);
        auto offsetSimpleFunc = parseSimplePBFunc(toString(offset));
        // offsetSimpleFunc.args_ should be a dummy variable equals to `bound`'s
        // value. Leave it.
        ASSERT(offsetSimpleFunc.values_.size() == 1);
        auto offsetExpr = demangler(offsetSimpleFunc.values_[0]);
        if (requireConst && !HashComparator{}(offsetExpr, makeIntConst(0))) {
            strideInt = 1;
            offsetExpr = makeIntConst(0);
        }
        return {strideInt, offsetExpr};
    }

  public:
    CompUniqueBoundsPBWithStride(const CompTransientBoundsInterface &transients)
        : CompUniqueBoundsPB(transients) {}

    std::tuple<Expr /* lower */, Expr /* upper */, Expr /* upper - lower */,
               int64_t /* modulo */, Expr /* offset */>
    unionBoundsAndGetStride(
        const std::vector<Ref<CompUniqueBounds::Bound>> &bounds,
        bool requireConst) {
        auto bound = unionBoundsAsBound(bounds);

        // if no bound presented, return an empty range
        if (!bound.isValid()) {
            return {makeIntConst(0), makeIntConst(-1), makeIntConst(0), 1,
                    makeIntConst(0)};
        }

        // translate the lower and upper bounds back to expression
        Expr l, u, diff;
        if (requireConst) {
            auto ll = bound->lowerInt();
            auto uu = bound->upperInt();
            l = makeIntConst(ll);
            u = makeIntConst(uu);
            diff = makeIntConst(uu - ll);
        } else {
            std::tie(l, u, diff) = bound->lowerUpperDiffExpr();
        }

        // Addition detction for strides
        auto [strideInt, offsetExpr] = getStride(bound, requireConst);

        return {l, u, diff, strideInt, offsetExpr};
    }

    std::vector<
        std::tuple<Expr /* lower */, Expr /* upper */, Expr /* upper - lower */,
                   int64_t /* modulo */, Expr /* offset */>>
    unionBoundsAndGetHighOrderStride(
        const std::vector<Ref<CompUniqueBounds::Bound>> &bounds,
        bool requireConst) {
        auto bound = unionBoundsAsBound(bounds);

        // if no bound presented, return an empty loop nest
        if (!bound.isValid()) {
            return {};
        }

        PBSet set = bound->bound_;

        // Reveal local dimensions
        set = isl_set_lift(set.move());

        // Put local dimension at front, so we can represent the target
        // dimension by local dimensions, instead of representing local
        // dimensions by the target dimension. The set returned by isl_set_lift
        // is a wrapped set, so we can simply unwrap it and then reverse it.
        set = isl_set_flatten(
            isl_map_wrap(isl_map_reverse(isl_set_unwrap(set.move()))));

        ASSERT(set.nDims() >= 1);
        std::vector<std::tuple<Expr, Expr, Expr, int64_t, Expr>> ret;
        ret.reserve(set.nDims());
        auto demangleMap = *bound->demangleMap_;
        int i = 0;
        while (true) {
            // Project onto the loop we are checking
            PBSet thisLoopSet = projectOutDims(set, 1, set.nDims() - 1);
            if (thisLoopSet.isSingleValued() && set.nDims() > 1) {
                // This dimension has no contribution. But the last dim in `set`
                // must no be skipped, because it is the target loop
                set = projectOutDims(std::move(set), 0, 1);
                continue;
            }

            auto thisLoopBound = Ref<CompUniqueBoundsPB::Bound>::make(
                bound->ctx_,
                Ref<std::unordered_map<std::string, Expr>>::make(demangleMap),
                thisLoopSet);
            Expr l, u, diff;
            if (requireConst) {
                auto ll = thisLoopBound->lowerInt();
                auto uu = thisLoopBound->upperInt();
                l = makeIntConst(ll);
                u = makeIntConst(uu);
                diff = makeIntConst(uu - ll);
            } else {
                std::tie(l, u, diff) = thisLoopBound->lowerUpperDiffExpr();
            }
            auto [strideInt, offsetExpr] =
                getStride(thisLoopBound, requireConst);
            ret.emplace_back(l, u, diff, strideInt, offsetExpr);

            if (set.nDims() == 1) {
                break;
            } else {
                // As we go from outer loops to inner loops, we will move range
                // dimensions to parameter dimensions, so inner loops will be
                // represented by outer loops. The parameter name used here is
                // temporary, and will be replaced later.
                auto paramName = "ft_shrink_for_tmp_" + std::to_string(i++);
                set = moveDimToNamedParam(*bound->ctx_, std::move(set), 0,
                                          paramName);
                demangleMap[paramName] = makeVar(paramName);
            }
        }

        return ret;
    }
};

} // Anonymous namespace

void CheckSideEffect::visit(const Store &op) { hasSideEffect_ = true; }

void CheckSideEffect::visit(const ReduceTo &op) { hasSideEffect_ = true; }

void CheckSideEffect::visit(const Intrinsic &op) {
    if (op->hasSideEffect_) {
        hasSideEffect_ = true;
    } else {
        Visitor::visit(op);
    }
}

Stmt ShrinkFor::visitStmt(const Stmt &stmt) {
    if (stmt == subAST_)
        inSubAST_ = true;
    if (subAST_.isValid() && !(inSubAST_ || subASTAncestors_.contains(stmt)))
        return deepCopy(stmt);

    auto ret = BaseClass::visitStmt(stmt);

    if (stmt == subAST_)
        inSubAST_ = false;

    CheckSideEffect checker;
    switch (stmt->nodeType()) {
    case ASTNodeType::Store:
    case ASTNodeType::ReduceTo:
    case ASTNodeType::Eval:
        checker(stmt);
        break;
    case ASTNodeType::If:
        checker(stmt.as<IfNode>()->cond_);
        break;
    case ASTNodeType::Assert:
        checker(stmt.as<AssertNode>()->cond_);
        break;
    case ASTNodeType::For:
        checker(stmt.as<ForNode>()->begin_);
        checker(stmt.as<ForNode>()->end_);
        checker(stmt.as<ForNode>()->step_);
        break;
    case ASTNodeType::VarDef:
        for (auto &&dim : stmt.as<VarDefNode>()->buffer_->tensor()->shape()) {
            checker(dim);
        }
        break;
    default:;
    }
    if (checker.hasSideEffect()) {
        for (auto &&[_var, _names] : views::zip(iterStack_, namesStack_)) {
            auto &&names = filterNames(_names);

            // We need linear programming from PBCompBounds, because the
            // minimum/maximum value of a linear function does not always appear
            // at the minimum/maximum points of its parameters.
            // See 2.pass/test_shrink_for.py::test_linear_bounds
            //
            // PBCompBounds requires one instance per Stmt
            CompUniqueBoundsPBWithStride bound(*this);

            // Trigger recomputing in analyze/comp_unique_bounds
            auto var = deepCopy(_var).as<VarNode>();
            newRange_[var].emplace_back(
                bound.getBound(var)->restrictScope(names));
        }
    }

    return ret;
}

Stmt ShrinkFor::visit(const For &_op) {
    auto var = makeVar(_op->iter_).as<VarNode>();
    newRange_.erase(var);

    iterStack_.emplace_back(var);
    namesStack_.emplace_back(names());
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    namesStack_.pop_back();
    iterStack_.pop_back();

    if ((subAST_.isValid() && !inSubAST_) || !filterLoop(op)) {
        return op;
    }

    if (!newRange_.count(var)) {
        return makeStmtSeq({});
    }

    // PBCompBounds requires one instance per Stmt
    CompUniqueBoundsPBWithStride bound(*this);

    // Backends do not support these loops to be of variable lengths
    bool requireConst = op->property_->unroll_;

    if (unordered_ && op->step_->nodeType() == ASTNodeType::IntConst &&
        op->property_->parallel_ == serialScope) {
        auto info = bound.unionBoundsAndGetHighOrderStride(newRange_[var],
                                                           requireConst);
        std::unordered_set<std::string> usedNames = uni(names(), allNames(op));
        std::unordered_map<std::string, Expr> replace;
        Stmt ret = op->body_;
        for (auto &&[i, item] : views::reverse(views::enumerate(info))) {
            auto &&[lower, upper, diff, stride, offset] = item;
            ASSERT(stride > 0);

            // The last (first before we reverse it) iter is the original iter.
            // Keep its name. The others are renamed.
            auto thisIterName = op->iter_;
            if (i != info.size() - 1) {
                thisIterName = getNewName(op->iter_, usedNames);
                usedNames.emplace(thisIterName);
            }
            replace["ft_shrink_for_tmp_" + std::to_string(i)] =
                makeVar(thisIterName);

            auto begin = lower;
            auto end = makeAdd(upper, makeIntConst(1));
            auto len = makeAdd(diff, makeIntConst(1));
            if (stride > 1) {
                // Find the lowest integer after `lower` that remains `offset`
                // modulo `stride`: lowerOnOffset = lower + ((offset - lower) %
                // stride + stride) % stride
                begin = makeAdd(lower,
                                makeMod(makeAdd(makeMod(makeSub(offset, lower),
                                                        makeIntConst(stride)),
                                                makeIntConst(stride)),
                                        makeIntConst(stride)));
                len = makeAdd(makeFloorDiv(diff, makeIntConst(stride)),
                              makeIntConst(1));
            }
            auto step = makeIntConst(stride);

            ret = makeFor(thisIterName, std::move(begin), std::move(end),
                          std::move(step), std::move(len), op->property_,
                          std::move(ret));
        }
        ret = ReplaceIter{replace}(ret);

        // Assign the old ID and metadata to the outer-most new loop
        ret->setId(op->id());
        ret->metadata() = op->metadata();

        return ret;

    } else {
        auto [lower, upper, diff, stride, offset] =
            bound.unionBoundsAndGetStride(newRange_[var], requireConst);
        ASSERT(stride > 0);

        // Since we can't normalize the loops (see the comment in shrinkFor), we
        // have to handle step_ here.
        if (op->step_->nodeType() == ASTNodeType::IntConst) {
            auto step = op->step_.as<IntConstNode>()->val_;
            ASSERT(stride % step == 0);
            if (step > 0) {
                if (lower.isValid()) {
                    if (stride > 1) {
                        // Find the lowest integer after `lower` that remains
                        // `offset` modulo `stride`: lowerOnOffset = lower +
                        // ((offset - lower) % stride + stride) % stride
                        op->begin_ = makeAdd(
                            lower,
                            makeMod(makeAdd(makeMod(makeSub(offset, lower),
                                                    makeIntConst(stride)),
                                            makeIntConst(stride)),
                                    makeIntConst(stride)));
                        op->len_ =
                            makeAdd(makeFloorDiv(diff, makeIntConst(stride)),
                                    makeIntConst(1));
                    } else {
                        op->begin_ = lower;
                        op->len_ = makeAdd(diff, makeIntConst(1));
                    }
                }
                if (upper.isValid()) {
                    op->end_ = makeAdd(upper, makeIntConst(1));
                }
                op->step_ = makeIntConst(stride);
            } else if (step < 0) {
                if (upper.isValid()) {
                    if (stride > 1) {
                        // Find the highest integer before `upper` that remains
                        // `offset` modulo `stride`: upperOnOffset = upper -
                        // ((upper - offset) % stride + stride) % stride
                        op->begin_ = makeSub(
                            upper,
                            makeMod(makeAdd(makeMod(makeSub(upper, offset),
                                                    makeIntConst(stride)),
                                            makeIntConst(stride)),
                                    makeIntConst(stride)));
                        op->len_ =
                            makeAdd(makeFloorDiv(diff, makeIntConst(stride)),
                                    makeIntConst(1));
                    } else {
                        op->begin_ = upper;
                        op->len_ = makeAdd(diff, makeIntConst(1));
                    }
                }
                if (lower.isValid()) {
                    op->end_ = makeAdd(lower, makeIntConst(-1));
                }
                op->step_ = makeIntConst(-stride);
            }
        }

        return op;
    }
}

void ShrinkFor::setSubAST(const Stmt &subAST) {
    subAST_ = subAST;
    for (Stmt s = subAST->parentStmt(); s.isValid(); s = s->parentStmt())
        subASTAncestors_.insert(s);
}

Stmt shrinkFor(const Stmt &_op, const ID &_subAST, bool doSimplify,
               bool unordered) {
    auto op = _op;
    auto subAST = _subAST;

    // DO NOT CALL normalizeLoops HERE! Since we often use (-INT_MAX, INT_MAX)
    // for unkown ranges and then do shrinkFor, normalizing loops here will end
    // up with complex expressions around INT_MAX.

    if (doSimplify) { // Const prop + eliminate empty loops
        std::vector<Stmt> allInSubAST;
        if (subAST.isValid()) {
            allInSubAST = findAllStmt(op, "(<<-" + toString(subAST) + ")|" +
                                              toString(subAST));
        }

        op = simplify(op);

        Stmt newSubAST;
        for (auto &&item : allInSubAST) {
            if (auto &&s = findAllStmt(op, item->id()); !s.empty()) {
                newSubAST = newSubAST.isValid() ? lcaStmt(newSubAST, s.front())
                                                : s.front();
            }
        }
        subAST = newSubAST.isValid() ? newSubAST->id() : ID();
    }

    ShrinkFor shrinker{unordered};
    if (subAST.isValid())
        shrinker.setSubAST(findStmt(op, subAST));
    op = shrinker(op);

    // Ranges from lifting are often quite strange. We'd better normalize them
    if (unordered) {
        op = normalizeLoops(op, [&](const For &loop) {
            return subAST.isValid() ? loop->ancestorById(subAST).isValid()
                                    : true;
        });
    }

    if (doSimplify) // Make new ranges simple + remove redundant branches
        op = simplify(pbSimplify(z3Simplify(op)));

    return op;
}

} // namespace freetensor
