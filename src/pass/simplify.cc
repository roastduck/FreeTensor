#include <arith/analyzer.h>
#include <arith/hash.h>
#include <pass/simplify.h>

namespace ir {

uint64_t SimplifyPass::getHash(const Expr &op) {
    if (hash_.count(op.get())) { // maybe not, beacuse Mutator::visit
        return hash_.at(op.get());
    } else {
        return ::ir::getHash(op);
    }
}

bool SimplifyPass::alwaysLT(const Expr &lhs, const Expr &rhs) {
    auto hl = getHash(lhs);
    auto hr = getHash(rhs);
    if (upper_.count(hl) && lower_.count(hr)) {
        auto &&l = upper_.at(hl);
        auto &&r = lower_.at(hr);
        if (l->nodeType() == ASTNodeType::IntConst &&
            r->nodeType() == ASTNodeType::IntConst &&
            l.as<IntConstNode>()->val_ < r.as<IntConstNode>()->val_) {
            return true;
        }
    }
    return false;
}

bool SimplifyPass::alwaysLE(const Expr &lhs, const Expr &rhs) {
    auto hl = getHash(lhs);
    auto hr = getHash(rhs);
    if (upper_.count(hl) && lower_.count(hr)) {
        auto &&l = upper_.at(hl);
        auto &&r = lower_.at(hr);
        if (l->nodeType() == ASTNodeType::IntConst &&
            r->nodeType() == ASTNodeType::IntConst &&
            l.as<IntConstNode>()->val_ <= r.as<IntConstNode>()->val_) {
            return true;
        }
    }
    return false;
}

Expr SimplifyPass::visit(const LT &_op) {
    auto op = Mutator::visit(_op);
    ASSERT(op->nodeType() == ASTNodeType::LT);
    if (alwaysLT(op.as<LTNode>()->lhs_, op.as<LTNode>()->rhs_)) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    if (alwaysLE(op.as<LTNode>()->rhs_, op.as<LTNode>()->lhs_)) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    return op;
}

Expr SimplifyPass::visit(const LE &_op) {
    auto op = Mutator::visit(_op);
    ASSERT(op->nodeType() == ASTNodeType::LE);
    if (alwaysLE(op.as<LENode>()->lhs_, op.as<LENode>()->rhs_)) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    if (alwaysLT(op.as<LENode>()->rhs_, op.as<LENode>()->lhs_)) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    return op;
}

Expr SimplifyPass::visit(const GT &_op) {
    auto op = Mutator::visit(_op);
    ASSERT(op->nodeType() == ASTNodeType::GT);
    if (alwaysLT(op.as<GTNode>()->rhs_, op.as<GTNode>()->lhs_)) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    if (alwaysLE(op.as<GTNode>()->lhs_, op.as<GTNode>()->rhs_)) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    return op;
}

Expr SimplifyPass::visit(const GE &_op) {
    auto op = Mutator::visit(_op);
    ASSERT(op->nodeType() == ASTNodeType::GE);
    if (alwaysLE(op.as<GENode>()->rhs_, op.as<GENode>()->lhs_)) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    if (alwaysLT(op.as<GENode>()->lhs_, op.as<GENode>()->rhs_)) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    return op;
}

Expr SimplifyPass::visit(const EQ &_op) {
    auto op = Mutator::visit(_op);
    ASSERT(op->nodeType() == ASTNodeType::EQ);
    auto l = op.as<EQNode>()->lhs_;
    auto r = op.as<EQNode>()->rhs_;
    if (l->nodeType() == ASTNodeType::IntConst &&
        r->nodeType() == ASTNodeType::IntConst) {
        bool eq = l.as<IntConstNode>()->val_ == r.as<IntConstNode>()->val_;
        isFixPoint_ = false;
        return makeIntConst(eq);
    }
    return op;
}

Expr SimplifyPass::visit(const NE &_op) {
    auto op = Mutator::visit(_op);
    ASSERT(op->nodeType() == ASTNodeType::NE);
    auto l = op.as<NENode>()->lhs_;
    auto r = op.as<NENode>()->rhs_;
    if (l->nodeType() == ASTNodeType::IntConst &&
        r->nodeType() == ASTNodeType::IntConst) {
        bool ne = l.as<IntConstNode>()->val_ != r.as<IntConstNode>()->val_;
        isFixPoint_ = false;
        return makeIntConst(ne);
    }
    return op;
}

Stmt SimplifyPass::visit(const If &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();
    if (op->cond_->nodeType() == ASTNodeType::IntConst) {
        isFixPoint_ = false;
        if (op->cond_.as<IntConstNode>()->val_) {
            return op->thenCase_;
        } else {
            return op->elseCase_;
        }
    }
    return op;
}

Stmt simplifyPass(const Stmt &_op) {
    Stmt op = _op;
    for (int i = 0;; i++) {
        if (i > 100) {
            WARNING(
                "SimplifyPass iterates over 100 rounds. Maybe there is a bug");
            return op;
        }

        std::unordered_map<const ExprNode *, uint64_t> hash; // expr -> hash
        std::unordered_map<uint64_t, Expr> subexpr;          // hash -> expr
        std::tie(hash, subexpr) = getHashMap(op);

        AnalyzeLinear analyzeLinear(hash);
        analyzeLinear(op);
        auto &&linear = analyzeLinear.result();

        AnalyzeBounds analyzeBounds(hash, subexpr, linear);
        analyzeBounds(op);
        auto &&lower = analyzeBounds.lower();
        auto &&upper = analyzeBounds.upper();

        SimplifyPass simplifyVisitor(hash, lower, upper);
        op = simplifyVisitor(op);
        if (simplifyVisitor.isFixPoint()) {
            break;
        }
    }
    return op;
}

} // namespace ir

