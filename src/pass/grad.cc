#include <analyze/all_defs.h>
#include <analyze/all_no_reuse_defs.h>
#include <analyze/all_reads.h>
#include <analyze/deps.h>
#include <cursor.h>
#include <pass/float_simplify.h>
#include <pass/grad.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/output_intermediates.h>
#include <pass/prop_const.h>
#include <pass/prop_one_time_use.h>
#include <pass/remove_cyclic_assign.h>
#include <pass/remove_dead_var.h>
#include <pass/remove_writes.h>
#include <pass/simplify.h>

namespace ir {

static bool isSameElem(const Store &s, const Load &l) {
    if (s->var_ != l->var_) {
        return false;
    }
    ASSERT(s->indices_.size() == l->indices_.size());
    for (size_t i = 0, iEnd = s->indices_.size(); i < iEnd; i++) {
        if (getHash(s->indices_[i]) != getHash(l->indices_[i])) {
            return false;
        }
    }
    return true;
}

static Expr canonicalReduceSumExpr(const Store &store) {
    // Check if var[indices] = expr is a canonical reduce sum, which can be
    // written as var[indices] = var[indices] + X, where there is no access to
    // var in X

    // NOTE: Do not copy the AST because `loadMap` requires the address
    if (store->expr_->nodeType() == ASTNodeType::Add) {
        auto expr = store->expr_.as<AddNode>();
        if (expr->lhs_->nodeType() == ASTNodeType::Load &&
            isSameElem(store, expr->lhs_.template as<LoadNode>())) {
            return expr->rhs_;
        }
        if (expr->rhs_->nodeType() == ASTNodeType::Load &&
            isSameElem(store, expr->rhs_.template as<LoadNode>())) {
            return expr->lhs_;
        }
    }
    return nullptr;
}

DataType PropagateRequire::dtype(const Expr &op) {
    typeInfer_(op);
    return typeInfer_.types().at(op);
}

void PropagateRequire::visit(const Load &op) {
    if (isFloat(dtype(op)) && !curTarget_.empty() &&
        defs_.at(op->var_)->buffer_->atype() == AccessType::Cache) {
        affectedDefs_.insert(defs_.at(op->var_)->id());
        // No need to recurse deeper
    }
}

void PropagateRequire::visit(const Store &op) {
    if (affectedDefs_.count(defs_.at(op->var_)->id())) {
        curTarget_ = defs_.at(op->var_)->id();
        (*this)(op->expr_);
        // No need to recurse into indices
        curTarget_ = "";
    }
}

void PropagateRequire::visit(const ReduceTo &op) {
    if (affectedDefs_.count(defs_.at(op->var_)->id())) {
        curTarget_ = defs_.at(op->var_)->id();
        (*this)(op->expr_);
        // No need to recurse into indices
        curTarget_ = "";
    }
}

void PropagateRequire::visit(const VarDef &op) {
    if (requires_.count(op->name_) || provides_.count(op->name_)) {
        affectedDefs_.insert(op->id());
    }
    ASSERT(!defs_.count(op->name_));
    defs_[op->name_] = op;
    buffers_[op->name_] = op->buffer_;
    Visitor::visit(op);
    defs_.erase(op->name_);
    buffers_.erase(op->name_);
}

Expr ReplaceVar::visit(const Var &op) {
    if (op->name_ == from_) {
        return to_;
    } else {
        return Mutator::visit(op);
    }
}

Expr ReplaceByTape::visit(const Load &op) {
    if (loadMap_.count(op)) {
        return (*this)(loadMap_.at(op));
    } else {
        return Mutator::visit(op);
    }
}

Stmt Grad::visit(const StmtSeq &op) {
    if (isRecompute_) {
        auto ret = Mutator::visit(op);
        ret->setId("");
        return ret;
    } else {
        std::vector<Stmt> stmts;
        stmts.reserve(op->stmts_.size() * 2);
        isRecompute_ = true;
        for (auto &&stmt : op->stmts_) {
            stmts.emplace_back((*this)(stmt));
        }
        isRecompute_ = false;
        for (auto it = op->stmts_.rbegin(); it != op->stmts_.rend(); it++) {
            stmts.emplace_back((*this)(*it));
        }
        return makeStmtSeq(op->id(), std::move(stmts));
    }
}

Stmt Grad::visit(const For &op) {
    if (isRecompute_) {
        auto ret = Mutator::visit(op);
        ret->setId("");
        return ret;
    } else {
        auto noDeps = op->property_.noDeps_;
        for (auto &&fwdVar : op->property_.noDeps_) {
            if (defs_.count(fwdVar) &&
                affectedDefs_.count(defs_.at(fwdVar)->id())) {
                noDeps.emplace_back(fwdVar + ".grad");
            }
        }
        return makeFor(
            op->id(), op->iter_, op->begin_, op->end_, op->len_,
            op->property_.withNoDeps(noDeps),
            ReplaceVar(
                op->iter_,
                makeSub(makeSub(makeAdd(op->begin_, op->end_), makeIntConst(1)),
                        makeVar(op->iter_)))((*this)(op->body_)));
    }
}

Stmt Grad::visit(const VarDef &_op) {
    ASSERT(!gradNames_.count(_op->name_));
    ASSERT(!defs_.count(_op->name_));
    ASSERT(!recomputed_.count(_op->name_));
    std::string gradName;
    if (affectedDefs_.count(_op->id())) {
        gradName = gradNames_[_op->name_] = _op->name_ + ".grad";
    }
    defs_[_op->name_] = _op;
    if (tapes_.count(_op->id())) {
        taped_.insert(_op->name_);
    }
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    taped_.erase(op->name_);
    defs_.erase(op->name_);
    gradNames_.erase(op->name_);
    recomputed_.erase(op->name_);

    if (isRecompute_) {
        op->setId("");
        return op;
    } else {
        if (affectedDefs_.count(_op->id())) {
            if (requires_.count(op->name_)) {
                requireGrads_[op->name_] = gradName;
            }
            if (provides_.count(op->name_)) {
                provideGrads_[op->name_] = gradName;
            }

            auto grad = op->body_;
            if ((op->buffer_->atype() != AccessType::Output &&
                 op->buffer_->atype() != AccessType::InOut) ||
                isTape_.count(op->name_)) {
                // We use reverse order in the init, so it can be better fused
                // with the backward pass
                std::vector<std::string> iters;
                std::vector<Expr> indices;
                int nDim = op->buffer_->tensor().shape().size();
                iters.reserve(nDim);
                indices.reserve(nDim);
                for (int i = 0; i < nDim; i++) {
                    std::string iter =
                        "." + gradName + ".i" + std::to_string(i);
                    indices.emplace_back(
                        makeSub(makeSub(op->buffer_->tensor().shape()[i],
                                        makeIntConst(1)),
                                makeVar(iter)));
                    iters.emplace_back(std::move(iter));
                }
                auto init = makeStore("", gradName, std::move(indices),
                                      makeIntConst(0));
                for (int i = nDim - 1; i >= 0; i--) {
                    init = makeFor("", iters[i], makeIntConst(0),
                                   op->buffer_->tensor().shape()[i],
                                   op->buffer_->tensor().shape()[i],
                                   ForProperty(), init);
                }
                grad = makeStmtSeq("", {init, grad});
            }

            grad = makeVarDef(op->id() + ".grad", gradName, *op->buffer_,
                              op->sizeLim_, grad, op->pinned_);
            grad = makeVarDef(op->id(), op->name_, *op->buffer_, op->sizeLim_,
                              grad, op->pinned_);

            switch (op->buffer_->atype()) {
            case AccessType::Input:
                grad.as<VarDefNode>()
                    ->body_.as<VarDefNode>()
                    ->buffer_->setAtype(AccessType::Output);
                break;
            case AccessType::Output:
            case AccessType::InOut:
                grad.as<VarDefNode>()->buffer_->setAtype(AccessType::Input);
                grad.as<VarDefNode>()
                    ->body_.as<VarDefNode>()
                    ->buffer_->setAtype(!isTape_.count(op->name_)
                                            ? AccessType::InOut
                                            : AccessType::Cache);
                break;
            case AccessType::Cache:
                break; // do nothing
            }

            return grad;
        } else {
            switch (op->buffer_->atype()) {
            case AccessType::Output:
            case AccessType::InOut:
                op.as<VarDefNode>()->buffer_->setAtype(AccessType::Input);
                break;
            default:
                break; // do nothing
            }

            return op;
        }
    }
}

Stmt Grad::visit(const Store &op) {
    auto &&buffer = defs_.at(op->var_)->buffer_;
    if (isRecompute_) {
        // FIXME: What if an intermediate variable is assigned and used multiple
        // times? E.g. a = x; use a; a = y; use a;
        bool recomputed =
            recomputed_.count(op->var_) && recomputed_.at(op->var_).count(op);
        if (!recomputed && buffer->atype() == AccessType::Cache &&
            !taped_.count(op->var_)) {
            recomputed_[op->var_].insert(op);
            auto ret = replaceByTape_(op);
            ret->setId("");
            return ret;
        } else {
            return makeStmtSeq("", {});
        }
    } else {
        std::vector<Stmt> stmts;
        if (gradNames_.count(op->var_)) {
            auto &&grad = gradNames_.at(op->var_);
            auto &&indices = op->indices_;
            if (!allReads(op->expr_).count(op->var_)) {
                // Quick path for acyclic assignment
                GradExpr exprVisitor(loadMap_, gradNames_, op->expr_,
                                     makeLoad(grad, indices),
                                     makeLoad(op->var_, indices));
                exprVisitor(op->expr_);

                for (auto &&stmt : exprVisitor.appends()) {
                    stmts.emplace_back(stmt);
                }
                if (notSingleWrite_.count(op)) {
                    stmts.emplace_back(
                        makeStore("", grad, indices, makeIntConst(0)));
                }
                return makeStmtSeq("", std::move(stmts));
            } else if (auto &&expr = canonicalReduceSumExpr(op);
                       expr.isValid()) {
                // Quick path for canonical reduce sum
                GradExpr exprVisitor(loadMap_, gradNames_, expr,
                                     makeLoad(grad, indices),
                                     makeLoad(op->var_, indices));
                exprVisitor(expr);

                for (auto &&stmt : exprVisitor.appends()) {
                    stmts.emplace_back(stmt);
                }
                return makeStmtSeq("", std::move(stmts));
            } else {
                // General case
                // Gradient of y[i] = f(x[i], y[i]) is:
                // d_y.old = d_y[i]
                // d_y[i] = 0
                // deduce d_x[i] and d_y[i] using d_y.old
                auto oldGrad = grad + ".old";
                stmts.emplace_back(
                    makeStore("", oldGrad, {}, makeLoad(grad, indices)));
                stmts.emplace_back(
                    makeStore("", grad, indices, makeIntConst(0)));

                GradExpr exprVisitor(loadMap_, gradNames_, op->expr_,
                                     makeLoad(oldGrad, {}),
                                     makeLoad(op->var_, indices));
                exprVisitor(op->expr_);

                for (auto &&stmt : exprVisitor.appends()) {
                    stmts.emplace_back(stmt);
                }
                return makeVarDef("", oldGrad,
                                  Buffer(Tensor({}, buffer->tensor().dtype()),
                                         AccessType::Cache, buffer->mtype()),
                                  nullptr, makeStmtSeq("", std::move(stmts)),
                                  false);
            }
        } else {
            return makeStmtSeq("", {});
        }
    }
}

void GradExpr::visit(const Load &op) {
    Visitor::visit(op);
    if (gradExprs_.count(op) && gradNames_.count(op->var_)) {
        appends_.push_back(makeReduceTo("", gradNames_.at(op->var_),
                                        op->indices_, ReduceOp::Add,
                                        gradExprs_.at(op), false));
    }
}

void GradExpr::visit(const Add &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->lhs_] = gradExprs_[op->rhs_] = gradExprs_.at(op);
    }
    Visitor::visit(op);
}

void GradExpr::visit(const Sub &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->lhs_] = gradExprs_.at(op);
        gradExprs_[op->rhs_] = makeSub(makeIntConst(0), gradExprs_.at(op));
    }
    Visitor::visit(op);
}

void GradExpr::visit(const Mul &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->lhs_] =
            makeMul(gradExprs_.at(op), useForwardVal(op->rhs_));
        gradExprs_[op->rhs_] =
            makeMul(gradExprs_.at(op), useForwardVal(op->lhs_));
    }
    Visitor::visit(op);
}

void GradExpr::visit(const RealDiv &op) {
    if (gradExprs_.count(op)) {
        auto lhs = useForwardVal(op->lhs_);
        auto rhs = useForwardVal(op->rhs_);
        gradExprs_[op->lhs_] = makeRealDiv(gradExprs_.at(op), rhs);
        gradExprs_[op->rhs_] = makeSub(
            makeIntConst(0),
            makeRealDiv(makeMul(gradExprs_.at(op), lhs), makeMul(rhs, rhs)));
    }
    Visitor::visit(op);
}

void GradExpr::visit(const Min &op) {
    if (gradExprs_.count(op)) {
        auto lhs = useForwardVal(op->lhs_);
        auto rhs = useForwardVal(op->rhs_);
        gradExprs_[op->lhs_] =
            makeIfExpr(makeLE(lhs, rhs), gradExprs_.at(op), makeIntConst(0));
        gradExprs_[op->rhs_] =
            makeIfExpr(makeLT(rhs, lhs), gradExprs_.at(op), makeIntConst(0));
    }
    Visitor::visit(op);
}

void GradExpr::visit(const Max &op) {
    if (gradExprs_.count(op)) {
        auto lhs = useForwardVal(op->lhs_);
        auto rhs = useForwardVal(op->rhs_);
        gradExprs_[op->lhs_] =
            makeIfExpr(makeGE(lhs, rhs), gradExprs_.at(op), makeIntConst(0));
        gradExprs_[op->rhs_] =
            makeIfExpr(makeGT(rhs, lhs), gradExprs_.at(op), makeIntConst(0));
    }
    Visitor::visit(op);
}

void GradExpr::visit(const IfExpr &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->thenCase_] =
            makeIfExpr(op->cond_, gradExprs_.at(op), makeIntConst(0));
        gradExprs_[op->elseCase_] =
            makeIfExpr(makeLNot(op->cond_), gradExprs_.at(op), makeIntConst(0));
    }
    Visitor::visit(op);
}

void GradExpr::visit(const Sqrt &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->expr_] = makeRealDiv(
            gradExprs_.at(op), makeMul(makeIntConst(2), useForwardVal(op)));
    }
    Visitor::visit(op);
}

void GradExpr::visit(const Exp &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->expr_] = makeMul(gradExprs_.at(op), useForwardVal(op));
    }
    Visitor::visit(op);
}

void GradExpr::visit(const Square &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->expr_] =
            makeMul(makeIntConst(2),
                    makeMul(gradExprs_.at(op), useForwardVal(op->expr_)));
    }
    Visitor::visit(op);
}

void GradExpr::visit(const Sigmoid &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->expr_] =
            makeMul(gradExprs_.at(op),
                    makeMul(makeSub(makeIntConst(1), useForwardVal(op)),
                            useForwardVal(op)));
    }
    Visitor::visit(op);
}

void GradExpr::visit(const Tanh &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->expr_] =
            makeMul(gradExprs_.at(op),
                    makeSub(makeIntConst(1), makeSquare(useForwardVal(op))));
    }
    Visitor::visit(op);
}

void GradExpr::visit(const Abs &op) {
    if (gradExprs_.count(op)) {
        gradExprs_[op->expr_] = makeIfExpr(
            makeGE(useForwardVal(op->expr_), makeIntConst(0)),
            gradExprs_.at(op), makeSub(makeIntConst(0), gradExprs_.at(op)));
    }
    Visitor::visit(op);
}

std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
grad(const Stmt &_op, const std::unordered_set<std::string> &requires,
     const std::unordered_set<std::string> &provides,
     const std::unordered_set<std::string> &tapes) {

    // expand the scope of each local variable, to avoid unnecessary recomputing
    auto op = hoistVarOverStmtSeq(_op);

    // Simplify before grad. E.g. grad of x^2 is much simpler than x * x
    op = floatSimplify(op);

    auto [forward, tapeMap, loadMap] = outputIntermediates(op, tapes);
    // loadMap contains pointers to forward. Do not modify forward

    // TODO: Are every tape vars requiring grad?
    auto requiresAndTapes = requires;
    for (auto &&[defId, name] : tapeMap) {
        requiresAndTapes.insert(name);
    }

    PropagateRequire propagator(requiresAndTapes, provides);
    size_t affectCnt;
    do {
        affectCnt = propagator.affectedDefs().size();
        propagator(forward);
    } while (propagator.affectedDefs().size() > affectCnt);

    std::unordered_set<Stmt> notSingleWrite;
    auto foundWAW = [&](const Dependency &d) {
        notSingleWrite.insert(d.earlier().as<StmtNode>());
    };
    findDeps(forward, {{}}, foundWAW, FindDepsMode::Dep, DEP_WAW, nullptr,
             false);

    Grad mutator(requires, provides, tapes, propagator.affectedDefs(), tapeMap,
                 loadMap, notSingleWrite);
    auto backward = mutator(forward);

    // We do some basic simplifications here, to reduce burden on auto-schedule
    backward = propOneTimeUse(backward);
    backward = simplifyPass(backward);
    backward = propConst(backward);
    backward = removeWrites(backward);
    backward = removeCyclicAssign(backward);
    backward = removeDeadVar(backward);

    return std::make_tuple(forward, backward, mutator.requireGrads(),
                           mutator.provideGrads(), tapeMap);
}

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
grad(const Func &func, const std::unordered_set<std::string> &requires,
     const std::unordered_set<std::string> &provides,
     const std::unordered_set<std::string> &tapes) {
    auto [forward, backward, requireGrads, provideGrads, tapeMap] =
        grad(func->body_, requires, provides, tapes);

    auto backwardParams = func->params_;
    auto forwardReturns = func->returns_;
    auto closure = func->closure_;
    for (auto &&[_oriDef, tapeName] : tapeMap) {
        auto &&oriDef = _oriDef;
        auto def = getCursorByFilter(
            func->body_, [&](const Cursor &c) { return c.id() == oriDef; });
        ASSERT(def.size() == 1 &&
               def.front().nodeType() == ASTNodeType::VarDef);
        auto tapeDType =
            def.front().node().as<VarDefNode>()->buffer_->tensor().dtype();
        forwardReturns.emplace_back(tapeName, tapeDType);
        backwardParams.emplace_back(tapeName);
        closure[tapeName] = Ref<Ref<Array>>::make(nullptr);
    }
    auto forwardFunc = makeFunc(func->name_, func->params_,
                                std::move(forwardReturns), forward, closure);

    for (auto &&[x, dzdx] : requireGrads) {
        backwardParams.emplace_back(dzdx);
    }
    for (auto &&[y, dzdy] : provideGrads) {
        backwardParams.emplace_back(dzdy);
    }
    auto backwardFunc =
        makeFunc(func->name_ + ".grad", std::move(backwardParams),
                 func->returns_, backward, closure);

    return std::make_tuple(forwardFunc, backwardFunc, requireGrads,
                           provideGrads, tapeMap);
}

static std::vector<std::string> _findTapeDefs(const Stmt &op,
                                              GradTapeMode mode) {
    switch (mode) {
    case GradTapeMode::All: {
        std::vector<std::string> ret;
        for (auto &&[id, name] : allDefs(op, {AccessType::Cache})) {
            ret.emplace_back(id);
        }
        return ret;
    }
    case GradTapeMode::Nothing:
        return {};
    case GradTapeMode::NoReuseOnly:
        return allNoReuseDefs(op, {AccessType::Cache});
    default:
        ASSERT(false);
    }
}

static std::unordered_set<std::string> findTapeDefs(const Stmt &op,
                                                    GradTapeMode mode) {
    auto ret = _findTapeDefs(op, mode);
    return std::unordered_set<std::string>(ret.begin(), ret.end());
}

std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
grad(const Stmt &op, const std::unordered_set<std::string> &requires,
     const std::unordered_set<std::string> &provides, GradTapeMode tapeMode) {
    return grad(op, requires, provides, findTapeDefs(op, tapeMode));
}

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
grad(const Func &func, const std::unordered_set<std::string> &requires,
     const std::unordered_set<std::string> &provides, GradTapeMode tapeMode) {
    return grad(func, requires, provides, findTapeDefs(func->body_, tapeMode));
}

} // namespace ir
