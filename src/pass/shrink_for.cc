#include <analyze/comp_unique_bounds_pb.h>
#include <analyze/find_stmt.h>
#include <math/min_max.h>
#include <math/parse_pb_expr.h>
#include <pass/replace_iter.h>
#include <pass/shrink_for.h>
#include <pass/simplify.h>
#include <pass/z3_simplify.h>

namespace freetensor {

namespace {

class CompUniqueBoundsPBWithStride : public CompUniqueBoundsPB {
  public:
    CompUniqueBoundsPBWithStride(const CompTransientBoundsInterface &transients)
        : CompUniqueBoundsPB(transients) {}

    std::tuple<Expr /* lower */, Expr /* upper */, int64_t /* modulo */,
               Expr /* offset */>
    unionBoundsAndGetStride(
        const std::vector<Ref<CompUniqueBounds::Bound>> &bounds) {
        auto bound = unionBoundsAsBound(bounds);

        // if no bound presented, return an empty range
        if (!bound.isValid()) {
            return {makeIntConst(0), makeIntConst(-1), 1, makeIntConst(0)};
        }

        // translate the lower and upper bounds back to expression
        auto l = bound->lowerExpr();
        auto u = bound->upperExpr();

        // Addition detction for strides
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

        return {l, u, strideInt, offsetExpr};
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

    if (!filterLoop(op)) {
        return op;
    }

    if (!newRange_.count(var)) {
        return makeStmtSeq({});
    }

    // PBCompBounds requires one instance per Stmt
    CompUniqueBoundsPBWithStride bound(*this);

    auto [lower, upper, stride, offset] =
        bound.unionBoundsAndGetStride(newRange_[var]);

    if (op->property_->unroll_) {
        // Backends do not support these loops to be of variable lengths
        lower = makeIntConst(bound.getIntLower(lower));
        upper = makeIntConst(bound.getIntUpper(upper));
        if (!HashComparator{}(offset, makeIntConst(0))) {
            stride = 1;
            offset = makeIntConst(0);
        }
    }

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
                        lower, makeMod(makeAdd(makeMod(makeSub(offset, lower),
                                                       makeIntConst(stride)),
                                               makeIntConst(stride)),
                                       makeIntConst(stride)));
                } else {
                    op->begin_ = lower;
                }
            }
            if (upper.isValid()) {
                op->end_ = makeAdd(upper, makeIntConst(1));
            }
            op->step_ = makeIntConst(stride);
            op->len_ = makeCeilDiv(makeSub(op->end_, op->begin_), op->step_);
        } else if (step < 0) {
            if (upper.isValid()) {
                if (stride < -1) {
                    // Find the highest integer before `upper` that remains
                    // `offset` modulo `stride`: upperOnOffset = upper -
                    // ((upper - offset) % stride + stride) % stride
                    op->begin_ = makeSub(
                        upper, makeMod(makeAdd(makeMod(makeSub(upper, offset),
                                                       makeIntConst(stride)),
                                               makeIntConst(stride)),
                                       makeIntConst(stride)));
                } else {
                    op->begin_ = upper;
                }
            }
            if (lower.isValid()) {
                op->end_ = makeAdd(lower, makeIntConst(-1));
            }
            op->step_ = makeIntConst(-stride);
            op->len_ = makeCeilDiv(makeSub(op->end_, op->begin_), op->step_);
        }
    }

    return op;
}

void ShrinkFor::setSubAST(const Stmt &subAST) {
    subAST_ = subAST;
    for (Stmt s = subAST->parentStmt(); s.isValid(); s = s->parentStmt())
        subASTAncestors_.insert(s);
}

Stmt shrinkFor(const Stmt &_op, const ID &subAST, bool doSimplify) {
    auto op = _op;

    // DO NOT CALL normalizeLoops HERE! Since we often use (-INT_MAX, INT_MAX)
    // for unkown ranges and then do shrinkFor, normalizing loops here will end
    // up with complex expressions around INT_MAX.

    if (doSimplify) // Const prop + eliminate empty loops
        op = simplify(op);

    ShrinkFor shrinker;
    if (subAST.isValid())
        shrinker.setSubAST(findStmt(op, subAST));
    op = shrinker(op);

    if (doSimplify) // Make new ranges simple + remove redundant branches
        op = simplify(z3Simplify(op));

    return op;
}

} // namespace freetensor
