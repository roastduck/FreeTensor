#include <analyze/all_reads.h>
#include <pass/simplify.h>
#include <pass/z3_simplify.h>

namespace ir {

void OutDatedCondsRemover::remove(const std::string &name) {
    for (auto &i : condList_) {
        if (i.second) {
            if (allReads(i.first).count(name))
                i.second = false;
        }
    }
}

void OutDatedCondsRemover::visit(const Store &op) {
    Visitor::visit(op);
    remove(op->var_);
}

void OutDatedCondsRemover::visit(const ReduceTo &op) {
    Visitor::visit(op);
    remove(op->var_);
}

int Z3Simplify::getVarId(const Expr &op) {
    if (!varId_.count(op)) {
        varId_[op] = varCnt_++;
    }
    return varId_.at(op);
}

void Z3Simplify::put(const Expr &key, const z3::expr &expr) {
    z3Exprs_[key] = Opt<z3::expr>::make(expr);
}

bool Z3Simplify::exists(const Expr &key) { return z3Exprs_.count(key); }

const z3::expr &Z3Simplify::get(const Expr &key) { return *z3Exprs_.at(key); }

bool Z3Simplify::prove(const Expr &op) {
    // expr can be proved <==> !expr can not be satisfied
    solver_.reset();
    for (auto i : condList_)
        if (i.second)
            solver_.add(get(i.first));
    if (exists(op)) {
        auto toCheck = !get(op);
        return solver_.check(1, &toCheck) == z3::unsat;
    }
    return false;
}

void Z3Simplify::push(const Expr &op) {
    std::pair<Expr, bool> cond = std::make_pair(op, false);
    if (exists(op))
        cond.second = true;
    condList_.emplace_back(cond);
}

void Z3Simplify::pop() { condList_.pop_back(); }

Expr Z3Simplify::visit(const Var &_op) {
    if (replace_.count(_op->name_)) {
        return (*this)(replace_.at(_op->name_));
    }
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    put(op, ctx_.int_const(("x" + std::to_string(getVarId(op))).c_str()));
    return op;
}

Expr Z3Simplify::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    // FIXME: Determine the data type
    put(op, ctx_.int_const(("x" + std::to_string(getVarId(op))).c_str()));
    return op;
}

Expr Z3Simplify::visit(const IntConst &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IntConst);
    auto op = __op.as<IntConstNode>();
    put(op, ctx_.int_val(op->val_));
    return op;
}

Expr Z3Simplify::visit(const BoolConst &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::BoolConst);
    auto op = __op.as<BoolConstNode>();
    put(op, ctx_.bool_val(op->val_));
    return op;
}

Expr Z3Simplify::visit(const Add &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.as<AddNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) + get(op->rhs_));
    }
    return op;
}

Expr Z3Simplify::visit(const Sub &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.as<SubNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) - get(op->rhs_));
    }
    return op;
}

Expr Z3Simplify::visit(const Mul &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mul);
    auto op = __op.as<MulNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) * get(op->rhs_));
    }
    return op;
}

Expr Z3Simplify::visit(const FloorDiv &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.as<FloorDivNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        // "/" in z3 means floor div. To verify, run the following in PyZ3
        // z3.prove(z3.Implies(z3.And(a == -3, b == 2), a / b == -2))
        put(op, get(op->lhs_) / get(op->rhs_));
    }
    return op;
}

Expr Z3Simplify::visit(const CeilDiv &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::CeilDiv);
    auto op = __op.as<CeilDivNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, (get(op->lhs_) - 1) / get(op->rhs_) + 1);
    }
    return op;
}

Expr Z3Simplify::visit(const Mod &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.as<ModNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, z3::mod(get(op->lhs_), get(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const Min &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Min);
    auto op = __op.as<MinNode>();

    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, z3::min(get(op->lhs_), get(op->rhs_)));
    }

    std::function<void(const Expr &, std::unordered_set<Expr> &)> recur =
        [&recur](const Expr &expr, std::unordered_set<Expr> &list) {
            if (expr->nodeType() == ASTNodeType::Min) {
                recur(expr.as<MinNode>()->lhs_, list);
                recur(expr.as<MinNode>()->rhs_, list);
            } else {
                list.insert(expr);
            }
        };
    std::unordered_set<Expr> lhs, rhs, all;
    recur(op->lhs_, lhs);
    recur(op->rhs_, rhs);
    all.insert(lhs.begin(), lhs.end());
    all.insert(rhs.begin(), rhs.end());

    for (auto &&l : lhs) {
        for (auto &&r : rhs) {
            if (prove((*this)(makeLE(l, r)))) {
                all.erase(r);
            } else if (prove((*this)(makeGE(l, r)))) {
                all.erase(l);
            }
        }
    }

    if (all.size() < lhs.size() + rhs.size()) {
        ASSERT(!all.empty());
        Expr ret;
        for (auto &&item : all) {
            ret = ret.isValid() ? makeMin(ret, item) : item;
        }
        if (exists(op)) {
            put(ret, get(op));
        }
        return ret;
    }

    return op;
}

Expr Z3Simplify::visit(const Max &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Max);
    auto op = __op.as<MaxNode>();

    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, z3::max(get(op->lhs_), get(op->rhs_)));
    }

    std::function<void(const Expr &, std::unordered_set<Expr> &)> recur =
        [&recur](const Expr &expr, std::unordered_set<Expr> &list) {
            if (expr->nodeType() == ASTNodeType::Max) {
                recur(expr.as<MaxNode>()->lhs_, list);
                recur(expr.as<MaxNode>()->rhs_, list);
            } else {
                list.insert(expr);
            }
        };
    std::unordered_set<Expr> lhs, rhs, all;
    recur(op->lhs_, lhs);
    recur(op->rhs_, rhs);
    all.insert(lhs.begin(), lhs.end());
    all.insert(rhs.begin(), rhs.end());

    for (auto &&l : lhs) {
        for (auto &&r : rhs) {
            if (prove((*this)(makeGE(l, r)))) {
                all.erase(r);
            } else if (prove((*this)(makeLE(l, r)))) {
                all.erase(l);
            }
        }
    }

    if (all.size() < lhs.size() + rhs.size()) {
        ASSERT(!all.empty());
        Expr ret;
        for (auto &&item : all) {
            ret = ret.isValid() ? makeMax(ret, item) : item;
        }
        if (exists(op)) {
            put(ret, get(op));
        }
        return ret;
    }

    return op;
}

Expr Z3Simplify::visit(const LT &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LT);
    auto op = __op.as<LTNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) < get(op->rhs_));
    }
    return op;
}

Expr Z3Simplify::visit(const LE &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LE);
    auto op = __op.as<LENode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) <= get(op->rhs_));
    }
    return op;
}

Expr Z3Simplify::visit(const GT &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GT);
    auto op = __op.as<GTNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) > get(op->rhs_));
    }
    return op;
}

Expr Z3Simplify::visit(const GE &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GE);
    auto op = __op.as<GENode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) >= get(op->rhs_));
    }
    return op;
}

Expr Z3Simplify::visit(const EQ &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::EQ);
    auto op = __op.as<EQNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) == get(op->rhs_));
    }
    return op;
}

Expr Z3Simplify::visit(const NE &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::NE);
    auto op = __op.as<NENode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) != get(op->rhs_));
    }
    return op;
}

Expr Z3Simplify::visit(const LAnd &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LAnd);
    auto op = __op.as<LAndNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        if (prove(op->lhs_)) {
            return op->rhs_;
        }
        if (prove(op->rhs_)) {
            return op->lhs_;
        }
        // If one of the operands is always false, visit(If) will deal with
        // it
        put(op, get(op->lhs_) && get(op->rhs_));
    }
    return op;
}

Expr Z3Simplify::visit(const LOr &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LOr);
    auto op = __op.as<LOrNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        if (prove((*this)(makeLNot(op->lhs_)))) {
            return op->rhs_;
        }
        if (prove((*this)(makeLNot(op->rhs_)))) {
            return op->lhs_;
        }
        // If one of the operands is always true, visit(If) will deal with
        // it
        put(op, get(op->lhs_) || get(op->rhs_));
    }
    return op;
}

Expr Z3Simplify::visit(const LNot &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LNot);
    auto op = __op.as<LNotNode>();
    if (exists(op->expr_)) {
        put(op, !get(op->expr_));
    }
    return op;
}

Expr Z3Simplify::visit(const IfExpr &op) {
    auto cond = (*this)(op->cond_);
    auto notCond = (*this)(makeLNot(op->cond_));
    if (prove(cond)) {
        return (*this)(op->thenCase_);
    }
    if (prove(notCond)) {
        return (*this)(op->elseCase_);
    }
    auto thenCase = (*this)(op->thenCase_);
    auto elseCase = (*this)(op->elseCase_);
    auto ret =
        makeIfExpr(std::move(cond), std::move(thenCase), std::move(elseCase));
    return COPY_DEBUG_INFO(ret, op);
}

Stmt Z3Simplify::visit(const If &op) {
    auto cond = (*this)(op->cond_);
    auto notCond = (*this)(makeLNot(op->cond_));
    if (prove(cond)) {
        return (*this)(op->thenCase_);
    }
    if (prove(notCond)) {
        return op->elseCase_.isValid() ? (*this)(op->elseCase_)
                                       : makeStmtSeq("", {});
    }

    push(cond);
    auto thenCase = (*this)(op->thenCase_);
    pop();

    Stmt elseCase = nullptr;
    if (op->elseCase_.isValid()) {
        push(notCond);
        elseCase = (*this)(op->elseCase_);
        pop();
    }

    auto ret = makeIf(op->id(), std::move(cond), std::move(thenCase),
                      std::move(elseCase));
    return COPY_DEBUG_INFO(ret, op);
}

Stmt Z3Simplify::visit(const Assert &op) {
    auto cond = (*this)(op->cond_);
    auto notCond = (*this)(makeLNot(op->cond_));
    if (prove(cond)) {
        return op->body_;
    }
    if (prove(notCond)) {
        throw AssertAlwaysFalse("Assertion always false: " + toString(op));
    }

    push(cond);
    auto body = (*this)(op->body_);
    pop();

    return makeAssert(op->id(), std::move(cond), std::move(body));
}

Stmt Z3Simplify::visit(const For &op) {
    OutDatedCondsRemover localRemover(condList_);
    localRemover(op);

    auto var = makeVar(op->iter_);
    auto begin = (*this)(op->begin_);
    auto end = (*this)(op->end_);
    auto step = (*this)(op->step_);
    auto len = (*this)(op->len_);

    if (prove((*this)(makeEQ(len, makeIntConst(0))))) {
        return makeStmtSeq("", {});
    }
    if (prove((*this)(makeEQ(len, makeIntConst(1))))) {
        ASSERT(!replace_.count(op->iter_));
        replace_[op->iter_] = begin;
        auto body = (*this)(op->body_);
        replace_.erase(op->iter_);
        return body;
    }

    Stmt body;
    if (op->step_->nodeType() == ASTNodeType::IntConst) {
        auto step = op->step_.as<IntConstNode>()->val_;
        if (step > 0) {
            push((*this)(makeGE(var, begin)));
            push((*this)(makeLT(var, end)));
            body = (*this)(op->body_);
            pop();
            pop();
        } else if (step < 0) {
            push((*this)(makeLE(var, begin)));
            push((*this)(makeGT(var, end)));
            body = (*this)(op->body_);
            pop();
            pop();
        } else {
            push((*this)(makeEQ(var, begin)));
            body = (*this)(op->body_);
            pop();
        }
    } else {
        body = (*this)(op->body_);
    }

    auto ret = makeFor(op->id(), op->iter_, std::move(begin), std::move(end),
                       std::move(step), std::move(len), op->property_,
                       std::move(body));
    return COPY_DEBUG_INFO(ret, op);
}

Stmt Z3Simplify::visit(const Store &op) {
    auto ret = Mutator::visit(op);
    remove_(op);
    return ret;
}

Stmt Z3Simplify::visit(const ReduceTo &op) {
    auto ret = Mutator::visit(op);
    remove_(op);
    return ret;
}

Stmt z3Simplify(const Stmt &_op) {
    auto op = Z3Simplify()(_op);
    op = simplifyPass(op); // to remove some empty blocks
    return op;
}

} // namespace ir
