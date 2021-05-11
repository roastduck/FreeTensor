#include <pass/simplify.h>
#include <pass/z3_simplify.h>

namespace ir {

int Z3Simplify::getVarId(const Expr &op) {
    getHash_(op);
    auto h = getHash_.hash().at(op);
    if (!varId_.count(h)) {
        varId_[h] = varCnt_++;
    }
    return varId_.at(h);
}

void Z3Simplify::put(const Expr &key, const z3::expr &expr) {
    z3Exprs_[key] = Ref<z3::expr>::make(expr);
}

bool Z3Simplify::exists(const Expr &key) { return z3Exprs_.count(key); }

const z3::expr &Z3Simplify::get(const Expr &key) { return *z3Exprs_.at(key); }

bool Z3Simplify::prove(const Expr &op) {
    // expr can be proved <==> !expr can not be satisfied
    if (exists(op)) {
        auto toCheck = !get(op);
        return solver_.check(1, &toCheck) == z3::unsat;
    }
    return false;
}

void Z3Simplify::push(const Expr &op) {
    solver_.push();
    if (exists(op)) {
        solver_.add(get(op));
    }
}

void Z3Simplify::pop() { solver_.pop(); }

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

Expr Z3Simplify::visit(const LAnd &op) {
    auto lhs = (*this)(op->lhs_);
    auto rhs = (*this)(op->rhs_);
    if (exists(lhs) && exists(rhs)) {
        if (prove(lhs)) {
            return rhs;
        }
        if (prove(rhs)) {
            return lhs;
        }
        // If one of the operands is always false, visit(If) will deal with it
        put(op, get(lhs) && get(rhs));
    }
    return makeLAnd(std::move(lhs), std::move(rhs));
}

Expr Z3Simplify::visit(const LOr &op) {
    auto lhs = (*this)(op->lhs_);
    auto rhs = (*this)(op->rhs_);
    if (exists(lhs) && exists(rhs)) {
        if (prove((*this)(makeLNot(lhs)))) {
            return rhs;
        }
        if (prove((*this)(makeLNot(rhs)))) {
            return lhs;
        }
        // If one of the operands is always true, visit(If) will deal with it
        put(op, get(lhs) || get(rhs));
    }
    return makeLOr(std::move(lhs), std::move(rhs));
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
        std::ostringstream os;
        os << "Assertion always false: " << op;
        throw InvalidProgram(os.str());
    }

    push(cond);
    auto body = (*this)(op->body_);
    pop();

    return makeAssert(op->id(), std::move(cond), std::move(body));
}

Stmt Z3Simplify::visit(const For &op) {
    auto var = makeVar(op->iter_);
    auto begin = (*this)(op->begin_);
    auto end = (*this)(op->end_);
    auto len = (*this)(op->len_);

    if (prove((*this)(makeGE(begin, end)))) {
        return makeStmtSeq("", {});
    }
    if (prove((*this)(makeEQ(makeAdd(begin, makeIntConst(1)), end)))) {
        ASSERT(!replace_.count(op->iter_));
        replace_[op->iter_] = begin;
        auto body = (*this)(op->body_);
        replace_.erase(op->iter_);
        return body;
    }

    push((*this)(makeGE(var, begin)));
    push((*this)(makeLT(var, end)));
    auto body = (*this)(op->body_);
    pop();
    pop();

    auto ret = makeFor(op->id(), op->iter_, std::move(begin), std::move(end),
                       std::move(len), op->parallel_, op->unroll_,
                       op->vectorize_, std::move(body));
    return COPY_DEBUG_INFO(ret, op);
}

Stmt z3Simplify(const Stmt &_op) {
    auto op = Z3Simplify()(_op);
    op = simplifyPass(op); // to remove some empty blocks
    return op;
}

} // namespace ir
