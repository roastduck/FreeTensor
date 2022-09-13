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

Stmt MakeReduction::doMake(Store op, ReduceOp reduceOp) {
    if (types_.count(reduceOp)) {
        auto expr = op->expr_.as<BinaryExprNode>();

        std::vector<Expr> items;
        std::function<void(const Expr &)> recur = [&](const Expr &sub) {
            if (sub->nodeType() == expr->nodeType()) {
                recur(sub.as<BinaryExprNode>()->lhs_);
                recur(sub.as<BinaryExprNode>()->rhs_);
            } else {
                items.emplace_back(sub);
            }
        };
        recur(expr);

        for (auto &&[i, item] : views::enumerate(items)) {
            if (item->nodeType() == ASTNodeType::Load &&
                isSameElem(op, item.as<LoadNode>())) {
                Expr others;
                for (auto &&[j, other] : views::enumerate(items)) {
                    if (i != j) {
                        if (canonicalOnly_ && allReads(other).count(op->var_)) {
                            goto fail;
                        }
                        others = others.isValid() ? makeBinary(expr->nodeType(),
                                                               others, other)
                                                  : other;
                    }
                }
                return makeReduceTo(op->var_, op->indices_, reduceOp, others,
                                    false, op->metadata(), op->id());
            fail:;
            }
        }
    }
    return op;
}

Stmt MakeReduction::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    switch (op->expr_->nodeType()) {
    case ASTNodeType::Add:
        return doMake(op, ReduceOp::Add);
    case ASTNodeType::Sub:
        return doMake(op, ReduceOp::Sub);
    case ASTNodeType::Mul:
        return doMake(op, ReduceOp::Mul);
    case ASTNodeType::Min:
        return doMake(op, ReduceOp::Min);
    case ASTNodeType::Max:
        return doMake(op, ReduceOp::Max);
    case ASTNodeType::LAnd:
        return doMake(op, ReduceOp::LAnd);
    case ASTNodeType::LOr:
        return doMake(op, ReduceOp::LOr);
    default:
        return op;
    }
}

} // namespace freetensor
