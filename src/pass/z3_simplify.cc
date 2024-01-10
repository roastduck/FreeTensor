#include <analyze/all_uses.h>
#include <container_utils.h>
#include <pass/annotate_conds.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/replace_iter.h>
#include <pass/simplify.h>
#include <pass/z3_simplify.h>

namespace freetensor {

static bool noIntersect(const std::unordered_set<std::string> &set1,
                        const std::unordered_set<std::string> &set2) {
    for (auto &&x : set1) {
        if (set2.count(x)) {
            return false;
        }
    }
    return true;
}

int Z3Simplify::getVarId(const Expr &op) {
    if (!varId_.count(op)) {
        varId_[op] = varCnt_++;
    }
    return varId_.at(op);
}

void Z3Simplify::put(const Expr &key, const z3::expr &expr,
                     const std::vector<std::optional<z3::expr>> &conds) {
    z3Exprs_[key] = ExprInfo{expr, conds};
}

bool Z3Simplify::exists(const Expr &key) { return z3Exprs_.count(key); }

const z3::expr &Z3Simplify::get(const Expr &key) {
    return *z3Exprs_.at(key).self_;
}

const std::vector<std::optional<z3::expr>> &Z3Simplify::conds(const Expr &key) {
    return z3Exprs_.at(key).conds_;
}

bool Z3Simplify::prove(const Expr &op) {
    // expr can be proved <==> !expr can not be satisfied
    if (exists(op)) {
        solver_.push();
        for (auto &&cond : conds(op)) {
            solver_.add(*cond);
        }
        auto toCheck = !get(op);
        auto ret = solver_.check(1, &toCheck) == z3::unsat;
        solver_.pop();
        return ret;
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
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    put(op, ctx_.int_const(("x" + std::to_string(getVarId(op))).c_str()));
    return op;
}

Expr Z3Simplify::visit(const Load &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    auto dtype = op->dtype();
    std::optional<z3::expr> expr;
    if (isInt(dtype)) {
        expr = ctx_.int_const(("x" + std::to_string(getVarId(op))).c_str());
    } else if (isBool(dtype)) {
        expr = ctx_.bool_const(("x" + std::to_string(getVarId(op))).c_str());
    } else {
        return op;
        // We don't simplify float in Z3Simplify
    }
    std::vector<std::optional<z3::expr>> conds;
    switch (dtype.sign()) {
    case SignDataType::GT0:
        conds.emplace_back(*expr > 0);
        break;
    case SignDataType::GE0:
        conds.emplace_back(*expr >= 0);
        break;
    case SignDataType::LT0:
        conds.emplace_back(*expr < 0);
        break;
    case SignDataType::LE0:
        conds.emplace_back(*expr <= 0);
        break;
    case SignDataType::NE0:
        conds.emplace_back(*expr != 0);
        break;
    case SignDataType::EQ0:
        conds.emplace_back(*expr == 0);
        break;
    default:; // do nothing
    }
    put(op, *expr, conds);
    return op;
}

Expr Z3Simplify::visit(const IntConst &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IntConst);
    auto op = __op.as<IntConstNode>();
    put(op, ctx_.int_val(op->val_));
    return op;
}

Expr Z3Simplify::visit(const BoolConst &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::BoolConst);
    auto op = __op.as<BoolConstNode>();
    put(op, ctx_.bool_val(op->val_));
    return op;
}

Expr Z3Simplify::visit(const Add &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Add);
    auto op = __op.as<AddNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) + get(op->rhs_),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const Sub &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Sub);
    auto op = __op.as<SubNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) - get(op->rhs_),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const Mul &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mul);
    auto op = __op.as<MulNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) * get(op->rhs_),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const FloorDiv &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::FloorDiv);
    auto op = __op.as<FloorDivNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        // "/" in z3 means floor div. To verify, run the following in PyZ3
        // z3.prove(z3.Implies(z3.And(a == -3, b == 2), a / b == -2))
        put(op, get(op->lhs_) / get(op->rhs_),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const CeilDiv &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::CeilDiv);
    auto op = __op.as<CeilDivNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, (get(op->lhs_) - 1) / get(op->rhs_) + 1,
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const Mod &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Mod);
    auto op = __op.as<ModNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, z3::mod(get(op->lhs_), get(op->rhs_)),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const Min &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Min);
    auto op = __op.as<MinNode>();

    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, z3::min(get(op->lhs_), get(op->rhs_)),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }

    std::function<void(const Expr &, std::unordered_set<Expr> &)> recur =
        [this, &recur](const Expr &expr, std::unordered_set<Expr> &list) {
            if (expr->nodeType() == ASTNodeType::Min) {
                recur(expr.as<MinNode>()->lhs_, list);
                recur(expr.as<MinNode>()->rhs_, list);
            } else {
                // We will make a shorter Min while the original Min still
                // alive, which will trigger a copy by SubTree and z3Expr will
                // be lost, so deepCopy here
                auto newExpr = deepCopy(expr);
                if (exists(expr)) {
                    put(newExpr, get(expr), conds(expr));
                }
                list.insert(std::move(newExpr));
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
            put(ret, get(op), conds(op));
        }
        return ret;
    }

    return op;
}

Expr Z3Simplify::visit(const Max &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Max);
    auto op = __op.as<MaxNode>();

    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, z3::max(get(op->lhs_), get(op->rhs_)),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }

    std::function<void(const Expr &, std::unordered_set<Expr> &)> recur =
        [this, &recur](const Expr &expr, std::unordered_set<Expr> &list) {
            if (expr->nodeType() == ASTNodeType::Max) {
                recur(expr.as<MaxNode>()->lhs_, list);
                recur(expr.as<MaxNode>()->rhs_, list);
            } else {
                // We will make a shorter Min while the original Min still
                // alive, which will trigger a copy by SubTree and z3Expr will
                // be lost, so deepCopy here
                auto newExpr = deepCopy(expr);
                if (exists(expr)) {
                    put(newExpr, get(expr), conds(expr));
                }
                list.insert(std::move(newExpr));
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
            put(ret, get(op), conds(op));
        }
        return ret;
    }

    return op;
}

Expr Z3Simplify::visit(const LT &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LT);
    auto op = __op.as<LTNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) < get(op->rhs_),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const LE &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LE);
    auto op = __op.as<LENode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) <= get(op->rhs_),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const GT &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GT);
    auto op = __op.as<GTNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) > get(op->rhs_),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const GE &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::GE);
    auto op = __op.as<GENode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) >= get(op->rhs_),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const EQ &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::EQ);
    auto op = __op.as<EQNode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) == get(op->rhs_),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const NE &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::NE);
    auto op = __op.as<NENode>();
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) != get(op->rhs_),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const LAnd &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LAnd);
    auto op = __op.as<LAndNode>();
    if (prove(op->lhs_)) {
        return op->rhs_;
    }
    if (prove(op->rhs_)) {
        return op->lhs_;
    }
    // If one of the operands is always false, visit(If) will deal with it
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) && get(op->rhs_),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const LOr &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LOr);
    auto op = __op.as<LOrNode>();
    if (prove((*this)(makeLNot(op->lhs_)))) {
        return op->rhs_;
    }
    if (prove((*this)(makeLNot(op->rhs_)))) {
        return op->lhs_;
    }
    // If one of the operands is always true, visit(If) will deal with it
    if (exists(op->lhs_) && exists(op->rhs_)) {
        put(op, get(op->lhs_) || get(op->rhs_),
            cat(conds(op->lhs_), conds(op->rhs_)));
    }
    return op;
}

Expr Z3Simplify::visit(const LNot &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LNot);
    auto op = __op.as<LNotNode>();
    if (exists(op->expr_)) {
        put(op, !get(op->expr_), conds(op->expr_));
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
    auto ret = makeIfExpr(cond, thenCase, elseCase, op->debugBlame());
    if (exists(cond) && exists(thenCase) && exists(elseCase)) {
        put(ret, z3::ite(get(cond), get(thenCase), get(elseCase)),
            cat(conds(cond), cat(conds(thenCase), conds(elseCase))));
    }
    return ret;
}

Stmt Z3Simplify::visit(const If &op) {
    auto cond = (*this)(op->cond_);
    auto notCond = (*this)(makeLNot(op->cond_));
    if (prove(cond)) {
        return (*this)(op->thenCase_);
    }
    if (prove(notCond)) {
        return op->elseCase_.isValid() ? (*this)(op->elseCase_)
                                       : makeStmtSeq({});
    }

    Stmt thenCase;
    if (noIntersect(allReads(cond), allWrites(op->thenCase_))) {
        push(cond);
        thenCase = (*this)(op->thenCase_);
        pop();
    } else {
        thenCase = (*this)(op->thenCase_);
    }

    Stmt elseCase = nullptr;
    if (op->elseCase_.isValid()) {
        if (noIntersect(allReads(cond), allWrites(op->elseCase_))) {
            push(notCond);
            elseCase = (*this)(op->elseCase_);
            pop();
        } else {
            elseCase = (*this)(op->elseCase_);
        }
    }

    return makeIf(std::move(cond), std::move(thenCase), std::move(elseCase),
                  op->metadata(), op->id(), op->debugBlame());
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

    Stmt body;
    if (noIntersect(allReads(cond), allWrites(op->body_))) {
        push(cond);
        body = (*this)(op->body_);
        pop();
    } else {
        body = (*this)(op->body_);
    }

    return makeAssert(std::move(cond), std::move(body), op->metadata(),
                      op->id());
}

Stmt Z3Simplify::visit(const Assume &op) {
    auto cond = (*this)(op->cond_);

    Stmt body;
    if (noIntersect(allReads(cond), allWrites(op->body_))) {
        push(cond);
        body = (*this)(op->body_);
        pop();
    } else {
        body = (*this)(op->body_);
    }

    return makeAssume(std::move(cond), std::move(body), op->metadata(),
                      op->id());
}

Stmt Z3Simplify::visit(const For &op) {
    auto var = makeVar(op->iter_);
    auto begin = (*this)(op->begin_);
    auto end = (*this)(op->end_);
    auto step = (*this)(op->step_);
    auto len = (*this)(op->len_);

    if (prove((*this)(makeEQ(len, makeIntConst(0))))) {
        return makeStmtSeq({});
    }
    if (prove((*this)(makeEQ(len, makeIntConst(1))))) {
        auto body = ReplaceIter(op->iter_, begin)(op->body_);
        return (*this)(body);
    }

    Stmt body;
    auto bodyAllWrites = allWrites(op->body_);
    if (op->step_->nodeType() == ASTNodeType::IntConst &&
        noIntersect(allReads(begin), bodyAllWrites) &&
        noIntersect(allReads(end), bodyAllWrites)) {
        auto step = op->step_.as<IntConstNode>()->val_;
        if (step > 0) {
            push((*this)(makeGE(var, begin)));
            push((*this)(makeLT(var, end)));
            push(
                (*this)(makeEQ(makeMod(makeSub(var, begin), makeIntConst(step)),
                               makeIntConst(0))));
            body = (*this)(op->body_);
            pop();
            pop();
            pop();
        } else if (step < 0) {
            push((*this)(makeLE(var, begin)));
            push((*this)(makeGT(var, end)));
            push((*this)(
                makeEQ(makeMod(makeSub(begin, var), makeIntConst(-step)),
                       makeIntConst(0))));
            body = (*this)(op->body_);
            pop();
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

    return makeFor(op->iter_, std::move(begin), std::move(end), std::move(step),
                   std::move(len), op->property_, std::move(body),
                   op->metadata(), op->id(), op->debugBlame());
}

Stmt Z3SimplifyWithSymbolTable::visit(const VarDef &op) {
    pushDef(op);
    auto ret = Z3Simplify::visit(op);
    popDef(op);
    return ret;
}

Stmt Z3SimplifyWithSymbolTable::visit(const For &op) {
    pushFor(op);
    auto ret = Z3Simplify::visit(op);
    popFor(op);
    return ret;
}

Stmt z3Simplify(const Stmt &_op) {
    auto op = annotateConds(_op);
    op = Z3SimplifyWithSymbolTable()(op);
    op = flattenStmtSeq(op);
    return op;
}

} // namespace freetensor
