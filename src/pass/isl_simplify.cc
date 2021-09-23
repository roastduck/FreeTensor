#include <pass/flatten_stmt_seq.h>
#include <pass/isl_simplify.h>

#include "detail/simplify.h"

namespace ir {

static std::unordered_set<int> uni(const std::unordered_set<int> &lhs,
                                   const std::unordered_set<int> &rhs) {
    auto ret = lhs;
    ret.insert(rhs.begin(), rhs.end());
    return ret;
}

static std::vector<std::string> cat(const std::vector<std::string> &lhs,
                                    const std::vector<std::string> &rhs) {
    auto ret = lhs;
    ret.insert(ret.end(), rhs.begin(), rhs.end());
    return ret;
}

int ISLCompBounds::getVarId(const Expr &op) {
    getHash_(op);
    auto h = getHash_.hash().at(op);
    if (!varId_.count(h)) {
        varId_[h] = varCnt_++;
    }
    return varId_.at(h);
}

Expr ISLCompBounds::visitExpr(
    const Expr &_op, const std::function<Expr(const Expr &)> &visitNode) {
    auto op = CompUniqueBounds::visitExpr(_op, visitNode);
    if (islExprs_.count(op)) {
        ISLExpr &e = islExprs_.at(op);
        auto tr = transient(op);
        for (auto &&first : tr.lower_) {
            if (islExprs_.count(first)) {
                auto &&bound = islExprs_.at(first);
                for (auto &&var : bound.var_) {
                    e.var_.insert(var);
                }
                e.cond_.emplace_back(e.expr_ + " >= " + bound.expr_);
            }
        }
        for (auto &&second : tr.upper_) {
            if (islExprs_.count(second)) {
                auto &&bound = islExprs_.at(second);
                for (auto &&var : bound.var_) {
                    e.var_.insert(var);
                }
                e.cond_.emplace_back(e.expr_ + " <= " + bound.expr_);
            }
        }

        std::string str = "{[";
        bool first = true;
        for (auto &&var : e.var_) {
            str += (first ? "" : ", ") + ("x" + std::to_string(var));
            first = false;
        }
        str += "] -> [" + e.expr_ + "]";
        first = true;
        for (auto &&cond : e.cond_) {
            str += (first ? ": " : " and ") + cond;
            first = false;
        }
        str += "}";
        ISLMap map(isl_, str);
        ISLSet image = range(std::move(map));
        ISLVal maxVal = dimMaxVal(image, 0);
        if (maxVal.isRat()) {
            auto &&list = getUpper(op);
            auto maxP = maxVal.numSi();
            auto maxQ = maxVal.denSi();
            updUpper(list, UpperBound{LinearExpr<Rational<int64_t>>{
                               {}, Rational<int64_t>{maxP, maxQ}}});
            setUpper(op, std::move(list));
        }
        ISLVal minVal = dimMinVal(image, 0);
        if (minVal.isRat()) {
            auto &&list = getLower(op);
            auto minP = minVal.numSi();
            auto minQ = minVal.denSi();
            updLower(list, LowerBound{LinearExpr<Rational<int64_t>>{
                               {}, Rational<int64_t>{minP, minQ}}});
            setLower(op, std::move(list));
        }
    }
    return op;
}

Expr ISLCompBounds::visit(const Var &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    auto var = getVarId(op);
    islExprs_[op] = ISLExpr{{var}, {}, "x" + std::to_string(var)};
    return op;
}

Expr ISLCompBounds::visit(const Load &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    auto var = getVarId(op);
    islExprs_[op] = ISLExpr{{var}, {}, "x" + std::to_string(var)};
    return op;
}

Expr ISLCompBounds::visit(const IntConst &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IntConst);
    auto op = __op.as<IntConstNode>();
    islExprs_[op] = ISLExpr{{}, {}, std::to_string(op->val_)};
    return op;
}

Expr ISLCompBounds::visit(const Add &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.as<AddNode>();
    if (islExprs_.count(op->lhs_) && islExprs_.count(op->rhs_)) {
        auto &&l = islExprs_.at(op->lhs_);
        auto &&r = islExprs_.at(op->rhs_);
        islExprs_[op] = ISLExpr{uni(l.var_, r.var_), cat(l.cond_, r.cond_),
                                "(" + l.expr_ + " + " + r.expr_ + ")"};
    }
    return op;
}

Expr ISLCompBounds::visit(const Sub &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.as<SubNode>();
    if (islExprs_.count(op->lhs_) && islExprs_.count(op->rhs_)) {
        auto &&l = islExprs_.at(op->lhs_);
        auto &&r = islExprs_.at(op->rhs_);
        islExprs_[op] = ISLExpr{uni(l.var_, r.var_), cat(l.cond_, r.cond_),
                                "(" + l.expr_ + " - " + r.expr_ + ")"};
    }
    return op;
}

Expr ISLCompBounds::visit(const Mul &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mul);
    auto op = __op.as<MulNode>();
    if (op->lhs_->nodeType() == ASTNodeType::IntConst ||
        op->rhs_->nodeType() == ASTNodeType::IntConst) {
        if (islExprs_.count(op->lhs_) && islExprs_.count(op->rhs_)) {
            auto &&l = islExprs_.at(op->lhs_);
            auto &&r = islExprs_.at(op->rhs_);
            islExprs_[op] = ISLExpr{uni(l.var_, r.var_), cat(l.cond_, r.cond_),
                                    "(" + l.expr_ + " * " + r.expr_ + ")"};
        }
    }
    return op;
}

Expr ISLCompBounds::visit(const FloorDiv &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.as<FloorDivNode>();
    if (op->rhs_->nodeType() == ASTNodeType::IntConst) {
        if (islExprs_.count(op->lhs_) && islExprs_.count(op->rhs_)) {
            auto &&l = islExprs_.at(op->lhs_);
            auto &&r = islExprs_.at(op->rhs_);
            islExprs_[op] = ISLExpr{uni(l.var_, r.var_), cat(l.cond_, r.cond_),
                                    "floor(" + l.expr_ + " / " + r.expr_ + ")"};
        }
    }
    return op;
}

Expr ISLCompBounds::visit(const CeilDiv &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::CeilDiv);
    auto op = __op.as<CeilDivNode>();
    if (op->rhs_->nodeType() == ASTNodeType::IntConst) {
        if (islExprs_.count(op->lhs_) && islExprs_.count(op->rhs_)) {
            auto &&l = islExprs_.at(op->lhs_);
            auto &&r = islExprs_.at(op->rhs_);
            islExprs_[op] = ISLExpr{uni(l.var_, r.var_), cat(l.cond_, r.cond_),
                                    "ceil(" + l.expr_ + " / " + r.expr_ + ")"};
        }
    }
    return op;
}

Expr ISLCompBounds::visit(const Mod &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.as<ModNode>();
    if (op->rhs_->nodeType() == ASTNodeType::IntConst) {
        if (islExprs_.count(op->lhs_) && islExprs_.count(op->rhs_)) {
            auto &&l = islExprs_.at(op->lhs_);
            auto &&r = islExprs_.at(op->rhs_);
            islExprs_[op] = ISLExpr{uni(l.var_, r.var_), cat(l.cond_, r.cond_),
                                    "(" + l.expr_ + " % " + r.expr_ + ")"};
        }
    }
    return op;
}

Expr ISLCompBounds::visit(const Min &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Min);
    auto op = __op.as<MinNode>();
    if (islExprs_.count(op->lhs_) && islExprs_.count(op->rhs_)) {
        auto &&l = islExprs_.at(op->lhs_);
        auto &&r = islExprs_.at(op->rhs_);
        islExprs_[op] = ISLExpr{uni(l.var_, r.var_), cat(l.cond_, r.cond_),
                                "min(" + l.expr_ + ", " + r.expr_ + ")"};
    }
    return op;
}

Expr ISLCompBounds::visit(const Max &_op) {
    auto __op = CompUniqueBounds::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Max);
    auto op = __op.as<MaxNode>();
    if (islExprs_.count(op->lhs_) && islExprs_.count(op->rhs_)) {
        auto &&l = islExprs_.at(op->lhs_);
        auto &&r = islExprs_.at(op->rhs_);
        islExprs_[op] = ISLExpr{uni(l.var_, r.var_), cat(l.cond_, r.cond_),
                                "max(" + l.expr_ + ", " + r.expr_ + ")"};
    }
    return op;
}

Stmt islSimplify(const Stmt &op) {
    return flattenStmtSeq(std::get<0>(simplifyAndGetBounds<ISLSimplify>(op)));
}

} // namespace ir

