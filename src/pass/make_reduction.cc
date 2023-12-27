#include <analyze/all_uses.h>
#include <container_utils.h>
#include <hash.h>
#include <pass/make_reduction.h>

namespace freetensor {

bool MakeReduction::isSameElem(const Store &s, const Load &l) {
    if (s->var_ != l->var_) {
        return false;
    }
    ASSERT(s->indices_.size() == l->indices_.size());
    for (auto &&[sIdx, lIdx] : views::zip(s->indices_, l->indices_)) {
        if (!HashComparator()(sIdx, lIdx)) {
            return false;
        }
    }
    return true;
}

Stmt MakeReduction::doMake(Store op, ASTNodeType binOp, ReduceOp reduceOp,
                           std::optional<ASTNodeType> invBinOp) {
    if (!types_.count(reduceOp)) {
        return op;
    }

    auto expr = op->expr_.as<BinaryExprNode>();

    std::vector<Expr> items, invItems;
    std::function<void(const Expr &, std::vector<Expr> &, std::vector<Expr> &)>
        recur = [&](const Expr &sub, std::vector<Expr> &items,
                    std::vector<Expr> &invItems) {
            if (sub->nodeType() == binOp) {
                recur(sub.as<BinaryExprNode>()->lhs_, items, invItems);
                recur(sub.as<BinaryExprNode>()->rhs_, items, invItems);
            } else if (sub->nodeType() == invBinOp) {
                recur(sub.as<BinaryExprNode>()->lhs_, items, invItems);
                recur(sub.as<BinaryExprNode>()->rhs_, invItems, items);
            } else {
                items.emplace_back(sub);
            }
        };
    recur(expr, items, invItems);

    for (auto &&[i, item] : views::enumerate(items)) {
        if (item->nodeType() == ASTNodeType::Load &&
            isSameElem(op, item.as<LoadNode>())) {
            Expr others, invOthers;
            for (auto &&[j, other] : views::enumerate(items)) {
                if (i != j) {
                    if (canonicalOnly_ && allReads(other).count(op->var_)) {
                        goto fail;
                    }
                    others = others.isValid() ? makeBinary(binOp, others, other)
                                              : other;
                }
            }
            for (auto &&other : invItems) {
                if (canonicalOnly_ && allReads(other).count(op->var_)) {
                    goto fail;
                }
                invOthers = invOthers.isValid()
                                ? makeBinary(binOp, invOthers, other)
                                : other;
            }
            if (others.isValid()) {
                if (invOthers.isValid()) { // a += b - c
                    ASSERT(invBinOp.has_value());
                    return makeReduceTo(
                        op->var_, op->indices_, reduceOp,
                        makeBinary(*invBinOp, others, invOthers), false,
                        op->metadata(), op->id());
                } else { // a += b
                    return makeReduceTo(op->var_, op->indices_, reduceOp,
                                        others, false, op->metadata(),
                                        op->id());
                }
            } else {
                if (invOthers.isValid()) { // a += 0 - b
                    ASSERT(invBinOp.has_value());
                    return makeReduceTo(
                        op->var_, op->indices_, reduceOp,
                        makeBinary(*invBinOp,
                                   neutralVal(invOthers->dtype(), reduceOp),
                                   invOthers),
                        false, op->metadata(), op->id());
                } else { // a = a
                    return makeStmtSeq({});
                }
            }
        fail:;
        }
    }
    return op;
}

Stmt MakeReduction::visit(const Store &_op) {
    // No need to worry about reducing a float expression onto a int. The
    // frontend has already inserted a Cast node in this case.
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    switch (op->expr_->nodeType()) {
    case ASTNodeType::Add:
    case ASTNodeType::Sub:
        return doMake(op, ASTNodeType::Add, ReduceOp::Add, ASTNodeType::Sub);
    case ASTNodeType::Mul:
    case ASTNodeType::RealDiv:
        return doMake(op, ASTNodeType::Mul, ReduceOp::Mul,
                      ASTNodeType::RealDiv);
    case ASTNodeType::Min:
        return doMake(op, ASTNodeType::Min, ReduceOp::Min);
    case ASTNodeType::Max:
        return doMake(op, ASTNodeType::Max, ReduceOp::Max);
    case ASTNodeType::LAnd:
        return doMake(op, ASTNodeType::LAnd, ReduceOp::LAnd);
    case ASTNodeType::LOr:
        return doMake(op, ASTNodeType::LOr, ReduceOp::LOr);
    case ASTNodeType::Load:
        return isSameElem(op, op->expr_.as<LoadNode>()) ? makeStmtSeq({}) : op;
    default:
        return op;
    }
}

} // namespace freetensor
