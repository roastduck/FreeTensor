#include <functional>

#include <analyze/bounds.h>

namespace ir {

Expr AnalyzeBounds::compLinear(int k, const Expr &a, const Expr &b) const {
    Expr x;
    if (a->nodeType() == ASTNodeType::IntConst) {
        x = makeIntConst(k * a.as<IntConstNode>()->val_);
    } else if (k == 1) {
        x = a;
    } else {
        x = makeMul(makeIntConst(k), a);
    }
    if (x->nodeType() == ASTNodeType::IntConst &&
        b->nodeType() == ASTNodeType::IntConst) {
        x = makeIntConst(x.as<IntConstNode>()->val_ +
                         b.as<IntConstNode>()->val_);
    } else if (b->nodeType() == ASTNodeType::IntConst &&
               b.as<IntConstNode>()->val_ == 0) {
        // do nothing
    } else {
        x = makeAdd(x, b);
    }
    return x;
}

std::vector<Expr> AnalyzeBounds::getLower(const LinearExpr &linear) const {
    std::vector<Expr> ret;
    typedef std::unordered_map<uint64_t, Scale>::const_iterator Iter;
    std::function<void(Iter, Expr)> dfs = [&](Iter i, Expr expr) {
        if (i == linear.coeff_.end()) {
            ret.emplace_back(expr);
            return;
        }
        auto ii = i;
        ii++;
        if (i->second.k > 0 && lower_.count(i->second.a.get())) {
            for (auto &&candidate : lower_.at(i->second.a.get())) {
                dfs(ii, compLinear(i->second.k, candidate, expr));
            }
        }
        if (i->second.k < 0 && upper_.count(i->second.a.get())) {
            for (auto &&candidate : upper_.at(i->second.a.get())) {
                dfs(ii, compLinear(i->second.k, candidate, expr));
            }
        }
        if (i->second.k == 0) {
            dfs(ii, expr);
        }
    };
    dfs(linear.coeff_.begin(), makeIntConst(linear.bias_));
    return ret;
}

std::vector<Expr> AnalyzeBounds::getUpper(const LinearExpr &linear) const {
    std::vector<Expr> ret;
    typedef std::unordered_map<uint64_t, Scale>::const_iterator Iter;
    std::function<void(Iter, Expr)> dfs = [&](Iter i, Expr expr) {
        if (i == linear.coeff_.end()) {
            ret.emplace_back(expr);
            return;
        }
        auto ii = i;
        ii++;
        if (i->second.k > 0 && upper_.count(i->second.a.get())) {
            for (auto &&candidate : upper_.at(i->second.a.get())) {
                dfs(ii, compLinear(i->second.k, candidate, expr));
            }
        }
        if (i->second.k < 0 && lower_.count(i->second.a.get())) {
            for (auto &&candidate : lower_.at(i->second.a.get())) {
                dfs(ii, compLinear(i->second.k, candidate, expr));
            }
        }
        if (i->second.k == 0) {
            dfs(ii, expr);
        }
    };
    dfs(linear.coeff_.begin(), makeIntConst(linear.bias_));
    return ret;
}

void AnalyzeBounds::updLower(const Expr &op, const std::vector<Expr> &exprs) {
    if (!lower_.count(op.get())) {
        lower_[op.get()] = exprs;
        return;
    }
    for (auto &&expr : exprs) {
        auto h = getHash(expr);
        for (Expr &old : lower_.at(op.get())) {
            if (getHash(old) == h) {
                goto done;
            }
            if (expr->nodeType() == ASTNodeType::IntConst &&
                old->nodeType() == ASTNodeType::IntConst) {
                auto oldVal = old.as<IntConstNode>()->val_;
                auto newVal = expr.as<IntConstNode>()->val_;
                if (newVal > oldVal) {
                    old = makeIntConst(newVal);
                }
                goto done;
            }
        }
        lower_.at(op.get()).emplace_back(expr);
    done:;
    }
}

void AnalyzeBounds::updUpper(const Expr &op, const std::vector<Expr> &exprs) {
    if (!upper_.count(op.get())) {
        upper_[op.get()] = exprs;
        return;
    }
    for (auto &&expr : exprs) {
        auto h = getHash(expr);
        for (Expr &old : upper_.at(op.get())) {
            if (getHash(old) == h) {
                goto done;
            }
            if (expr->nodeType() == ASTNodeType::IntConst &&
                old->nodeType() == ASTNodeType::IntConst) {
                auto oldVal = old.as<IntConstNode>()->val_;
                auto newVal = expr.as<IntConstNode>()->val_;
                if (newVal < oldVal) {
                    old = makeIntConst(newVal);
                }
                goto done;
            }
        }
        upper_.at(op.get()).emplace_back(expr);
    done:;
    }
}

uint64_t AnalyzeBounds::getHash(const Expr &op) {
    if (hash_.count(op.get())) {
        return hash_.at(op.get());
    } else { // lowers / uppers are new exprs
        return ::ir::getHash(op);
    }
}

Expr AnalyzeBounds::sub1(const Expr &op) {
    if (op->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(op.as<IntConstNode>()->val_ - 1);
    } else {
        return makeSub(op, makeIntConst(1));
    }
}

Expr AnalyzeBounds::add1(const Expr &op) {
    if (op->nodeType() == ASTNodeType::IntConst) {
        return makeIntConst(op.as<IntConstNode>()->val_ + 1);
    } else {
        return makeAdd(op, makeIntConst(1));
    }
}

void AnalyzeBounds::visit(const VarDef &op) {

    for (auto &&dim : op->buffer_->tensor().shape()) {
        (*this)(dim);
    }
    if (buffers_.count(op->name_)) {
        ERROR("Conflict var name: " + op->name_ +
              ". Nested vars with the same name are not allowed");
    }
    buffers_[op->name_] = op->buffer_;
    (*this)(op->body_);
    buffers_.erase(op->name_);
}

void AnalyzeBounds::visit(const Var &op) {
    Visitor::visit(op);
    updLower(op, {op}); // Don't forget itself
    updUpper(op, {op});
    if (iters_.count(op->name_)) {
        updLower(op, {iters_[op->name_].first});
        updUpper(op, {iters_[op->name_].second});
    }
}

void AnalyzeBounds::visit(const Load &op) {
    Visitor::visit(op);
    updLower(op, {op}); // Don't forget itself
    updUpper(op, {op});
}

void AnalyzeBounds::visit(const IntConst &op) { doAnalyze(op); }
void AnalyzeBounds::visit(const Add &op) { doAnalyze(op); }
void AnalyzeBounds::visit(const Sub &op) { doAnalyze(op); }
void AnalyzeBounds::visit(const Mul &op) { doAnalyze(op); }
void AnalyzeBounds::visit(const Div &op) { doAnalyze(op); }

void AnalyzeBounds::visit(const For &op) {
    if (iters_.count(op->iter_)) {
        ERROR("iterators with the same name in nested loops are not allowed");
    }
    iters_[op->iter_] = {op->begin_, sub1(op->end_)};
    Visitor::visit(op);
    iters_.erase(op->iter_);
}

void AnalyzeBounds::visit(const If &op) {
    (*this)(op->cond_);

    auto oldMap = iters_;
    switch (op->cond_->nodeType()) {
    case ASTNodeType::LT: {
        auto lt = op->cond_.as<LTNode>();
        if (lt->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[lt->lhs_.as<VarNode>()->name_].second = sub1(lt->rhs_);
        }
        if (lt->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[lt->rhs_.as<VarNode>()->name_].first = add1(lt->lhs_);
        }
        break;
    }
    case ASTNodeType::GT: {
        auto gt = op->cond_.as<GTNode>();
        if (gt->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[gt->lhs_.as<VarNode>()->name_].first = add1(gt->rhs_);
        }
        if (gt->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[gt->rhs_.as<VarNode>()->name_].second = sub1(gt->lhs_);
        }
        break;
    }
    case ASTNodeType::LE: {
        auto le = op->cond_.as<LTNode>();
        if (le->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[le->lhs_.as<VarNode>()->name_].second = le->rhs_;
        }
        if (le->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[le->rhs_.as<VarNode>()->name_].first = le->lhs_;
        }
        break;
    }
    case ASTNodeType::GE: {
        auto ge = op->cond_.as<GTNode>();
        if (ge->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[ge->lhs_.as<VarNode>()->name_].first = ge->rhs_;
        }
        if (ge->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[ge->rhs_.as<VarNode>()->name_].second = ge->lhs_;
        }
        break;
    }
    case ASTNodeType::EQ: {
        auto eq = op->cond_.as<GTNode>();
        if (eq->lhs_->nodeType() == ASTNodeType::Var) {
            iters_[eq->lhs_.as<VarNode>()->name_] = {eq->rhs_, eq->rhs_};
        }
        if (eq->rhs_->nodeType() == ASTNodeType::Var) {
            iters_[eq->rhs_.as<VarNode>()->name_] = {eq->lhs_, eq->lhs_};
        }
        break;
    }
    default:;
        // Do nothing
    }
    (*this)(op->thenCase_);
    iters_ = oldMap;

    if (op->elseCase_.isValid()) {
        auto oldMap = iters_;
        switch (op->cond_->nodeType()) {
        case ASTNodeType::GE: { // not LT
            auto lt = op->cond_.as<LTNode>();
            if (lt->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[lt->lhs_.as<VarNode>()->name_].second = sub1(lt->rhs_);
            }
            if (lt->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[lt->rhs_.as<VarNode>()->name_].first = add1(lt->lhs_);
            }
            break;
        }
        case ASTNodeType::LE: { // not GT
            auto gt = op->cond_.as<GTNode>();
            if (gt->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[gt->lhs_.as<VarNode>()->name_].first = add1(gt->rhs_);
            }
            if (gt->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[gt->rhs_.as<VarNode>()->name_].second = sub1(gt->lhs_);
            }
            break;
        }
        case ASTNodeType::GT: { // not LE
            auto le = op->cond_.as<LTNode>();
            if (le->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[le->lhs_.as<VarNode>()->name_].second = le->rhs_;
            }
            if (le->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[le->rhs_.as<VarNode>()->name_].first = le->lhs_;
            }
            break;
        }
        case ASTNodeType::LT: { // not GE
            auto ge = op->cond_.as<GTNode>();
            if (ge->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[ge->lhs_.as<VarNode>()->name_].first = ge->rhs_;
            }
            if (ge->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[ge->rhs_.as<VarNode>()->name_].second = ge->lhs_;
            }
            break;
        }
        case ASTNodeType::NE: { // not EQ
            auto eq = op->cond_.as<GTNode>();
            if (eq->lhs_->nodeType() == ASTNodeType::Var) {
                iters_[eq->lhs_.as<VarNode>()->name_] = {eq->rhs_, eq->rhs_};
            }
            if (eq->rhs_->nodeType() == ASTNodeType::Var) {
                iters_[eq->rhs_.as<VarNode>()->name_] = {eq->lhs_, eq->lhs_};
            }
            break;
        }
        default:;
            // Do nothing
        }
        (*this)(op->elseCase_);
        iters_ = oldMap;
    }
}

} // namespace ir
