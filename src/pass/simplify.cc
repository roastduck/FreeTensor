#include <analyze/bounds.h>
#include <analyze/hash.h>
#include <analyze/linear.h>
#include <analyze/normalize_if.h>
#include <pass/simplify.h>

namespace ir {

void FindInnerMostScope::visit(const Var &op) {
    Visitor::visit(op);
    innerMost_ = std::max(innerMost_, varScope_.at(op->name_));
}

void FindInnerMostScope::visit(const Load &op) {
    Visitor::visit(op);
    innerMost_ = std::max(innerMost_, varScope_.at(op->var_));
}

int findInnerMostScope(const std::unordered_map<std::string, int> &varScope,
                       const Expr &op) {
    FindInnerMostScope visitor(varScope);
    visitor(op);
    return visitor.innnerMost();
}

uint64_t SimplifyPass::getHash(const Expr &op) {
    if (hash_.count(op.get())) { // maybe not, beacuse Mutator::visit
        return hash_.at(op.get());
    } else {
        return ::ir::getHash(op);
    }
}

bool SimplifyPass::alwaysLT(const Expr &lhs, const Expr &rhs) {
    if (upper_.count(lhs.get()) && lower_.count(rhs.get())) {
        for (auto &&l : upper_.at(lhs.get())) {
            for (auto &&r : lower_.at(rhs.get())) {
                if (l->nodeType() == ASTNodeType::IntConst &&
                    r->nodeType() == ASTNodeType::IntConst &&
                    l.as<IntConstNode>()->val_ < r.as<IntConstNode>()->val_) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool SimplifyPass::alwaysLE(const Expr &lhs, const Expr &rhs) {
    if (upper_.count(lhs.get()) && lower_.count(rhs.get())) {
        for (auto &&l : upper_.at(lhs.get())) {
            for (auto &&r : lower_.at(rhs.get())) {
                if (l->nodeType() == ASTNodeType::IntConst &&
                    r->nodeType() == ASTNodeType::IntConst &&
                    l.as<IntConstNode>()->val_ <= r.as<IntConstNode>()->val_) {
                    return true;
                }
            }
        }
    }
    return false;
}

Expr SimplifyPass::visit(const LT &op) {
    if (alwaysLT(op.as<LTNode>()->lhs_, op.as<LTNode>()->rhs_)) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    if (alwaysLE(op.as<LTNode>()->rhs_, op.as<LTNode>()->lhs_)) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const LE &op) {
    if (alwaysLE(op.as<LENode>()->lhs_, op.as<LENode>()->rhs_)) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    if (alwaysLT(op.as<LENode>()->rhs_, op.as<LENode>()->lhs_)) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const GT &op) {
    if (alwaysLT(op.as<GTNode>()->rhs_, op.as<GTNode>()->lhs_)) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    if (alwaysLE(op.as<GTNode>()->lhs_, op.as<GTNode>()->rhs_)) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const GE &op) {
    if (alwaysLE(op.as<GENode>()->rhs_, op.as<GENode>()->lhs_)) {
        isFixPoint_ = false;
        return makeIntConst(1);
    }
    if (alwaysLT(op.as<GENode>()->lhs_, op.as<GENode>()->rhs_)) {
        isFixPoint_ = false;
        return makeIntConst(0);
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const EQ &op) {
    auto l = op.as<EQNode>()->lhs_;
    auto r = op.as<EQNode>()->rhs_;
    if (l->nodeType() == ASTNodeType::IntConst &&
        r->nodeType() == ASTNodeType::IntConst) {
        bool eq = l.as<IntConstNode>()->val_ == r.as<IntConstNode>()->val_;
        isFixPoint_ = false;
        return makeIntConst(eq);
    }
    return Mutator::visit(op);
}

Expr SimplifyPass::visit(const NE &op) {
    auto l = op.as<NENode>()->lhs_;
    auto r = op.as<NENode>()->rhs_;
    if (l->nodeType() == ASTNodeType::IntConst &&
        r->nodeType() == ASTNodeType::IntConst) {
        bool ne = l.as<IntConstNode>()->val_ != r.as<IntConstNode>()->val_;
        isFixPoint_ = false;
        return makeIntConst(ne);
    }
    return Mutator::visit(op);
}

Stmt SimplifyPass::visit(const VarDef &_op) {
    if (varScope_.count(_op->name_)) {
        ERROR("Conflict var name: " + _op->name_ +
              ". Nested vars with the same name are not allowed");
    }
    varScope_[_op->name_] = curScope_++;
    auto op = Mutator::visit(_op);
    varScope_.erase(_op->name_), curScope_--;
    return op;
}

Stmt SimplifyPass::visit(const For &_op) {
    if (varScope_.count(_op->iter_)) {
        ERROR("iterators with the same name in nested loops are not allowed");
    }
    varScope_[_op->iter_] = curScope_++;
    auto op = Mutator::visit(_op);
    varScope_.erase(_op->iter_), curScope_--;
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
    // Use _op, as op is not analyzed yet
    if (_op->info_equival_cond_.isValid()) {
        if (upper_.count(_op->info_equival_cond_.get())) {
            for (auto &&upper : upper_.at(_op->info_equival_cond_.get())) {
                if (upper->nodeType() == ASTNodeType::IntConst &&
                    upper.as<IntConstNode>()->val_ < 0) {
                    isFixPoint_ = false;
                    return _op->thenCase_;
                }
            }
        }
        if (lower_.count(_op->info_equival_cond_.get())) {
            for (auto &&lower : lower_.at(_op->info_equival_cond_.get())) {
                if (lower->nodeType() == ASTNodeType::IntConst &&
                    lower.as<IntConstNode>()->val_ >= 0) {
                    isFixPoint_ = false;
                    return _op->elseCase_;
                }
            }
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

        op = normalizeIf(op);
        op = Disambiguous()(op);

        auto hash = getHashMap(op);

        AnalyzeLinear analyzeLinear(hash);
        analyzeLinear(op);
        auto &&linear = analyzeLinear.result();

        AnalyzeBounds analyzeBounds(hash, linear);
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

