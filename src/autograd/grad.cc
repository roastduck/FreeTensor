#include <analyze/all_defs.h>
#include <analyze/all_no_reuse_defs.h>
#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <autograd/grad.h>
#include <autograd/output_intermediates.h>
#include <pass/float_simplify.h>
#include <pass/hoist_return_vars.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/make_reduction.h>
#include <pass/prop_one_time_use.h>
#include <pass/remove_cyclic_assign.h>
#include <pass/remove_dead_var.h>
#include <pass/remove_writes.h>
#include <pass/simplify.h>
#include <pass/tensor_prop_const.h>
#include <pass/undo_make_reduction.h>

namespace freetensor {

void PropagateRequires::visit(const Load &op) {
    if (isFloat(op->dtype()) && curTarget_.isValid() &&
        affectedDefs_.count(def(op->var_)->id())) {
        affectedDefs_.insert(curTarget_);
        // No need to recurse deeper
    }
}

void PropagateRequires::visit(const Store &op) {
    if (buffer(op->var_)->atype() == AccessType::Cache) {
        curTarget_ = def(op->var_)->id();
        (*this)(op->expr_);
        // No need to recurse into indices
        curTarget_ = {};
    }
}

void PropagateRequires::visit(const ReduceTo &op) {
    if (buffer(op->var_)->atype() == AccessType::Cache) {
        curTarget_ = def(op->var_)->id();
        (*this)(op->expr_);
        // No need to recurse into indices
        curTarget_ = {};
    }
}

void PropagateRequires::visit(const VarDef &op) {
    if (requires_.count(op->name_) || provides_.count(op->name_)) {
        affectedDefs_.insert(op->id());
    }
    BaseClass::visit(op);
}

std::unordered_set<ID> PropagateRequires::propagateUntilConverge(
    const Stmt &op, const std::unordered_set<std::string> &_requires,
    const std::unordered_set<std::string> &provides) {
    PropagateRequires propagator(_requires, provides);
    size_t affectCnt;
    do {
        affectCnt = propagator.affectedDefs().size();
        propagator(op);
    } while (propagator.affectedDefs().size() > affectCnt);
    return propagator.affectedDefs();
}

void PropagateProvides::visit(const Load &op) {
    if (isFloat(op->dtype()) && curTarget_.isValid() &&
        buffer(op->var_)->atype() == AccessType::Cache) {
        affectedDefs_.insert(def(op->var_)->id());
        // No need to recurse deeper
    }
}

void PropagateProvides::visit(const Store &op) {
    if (affectedDefs_.count(def(op->var_)->id())) {
        curTarget_ = def(op->var_)->id();
        (*this)(op->expr_);
        // No need to recurse into indices
        curTarget_ = {};
    }
}

void PropagateProvides::visit(const ReduceTo &op) {
    if (affectedDefs_.count(def(op->var_)->id())) {
        curTarget_ = def(op->var_)->id();
        (*this)(op->expr_);
        // No need to recurse into indices
        curTarget_ = {};
    }
}

void PropagateProvides::visit(const VarDef &op) {
    if (requires_.count(op->name_) || provides_.count(op->name_)) {
        affectedDefs_.insert(op->id());
    }
    BaseClass::visit(op);
}

std::unordered_set<ID> PropagateProvides::propagateUntilConverge(
    const Stmt &op, const std::unordered_set<std::string> &_requires,
    const std::unordered_set<std::string> &provides) {
    PropagateProvides propagator(_requires, provides);
    size_t affectCnt;
    do {
        affectCnt = propagator.affectedDefs().size();
        propagator(op);
    } while (propagator.affectedDefs().size() > affectCnt);
    return propagator.affectedDefs();
}

Expr ReplaceByTape::replaceForwardValue(const Expr &_equLoad) {
    auto __equLoad = deepCopy(_equLoad);
    ASSERT(__equLoad->nodeType() == ASTNodeType::Load);
    auto equLoad = __equLoad.as<LoadNode>();
    if (tapeMap_.count(symbolTable_.def(equLoad->var_)->id())) {
        auto tapeVar = tapeMap_.at(symbolTable_.def(equLoad->var_)->id());
        if (tapeVar != equLoad->var_) {
            equLoad->var_ = tapeVar;
            equLoad->indices_.insert(equLoad->indices_.begin(),
                                     versions_.at(parent_->id()));
        }
    }
    return equLoad;
}

Expr ReplaceByTape::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (tapeMap_.count(symbolTable_.def(_op->var_)->id())) {
        auto tapeVar = tapeMap_.at(symbolTable_.def(_op->var_)->id());
        if (tapeVar != op->var_) {
            op->var_ = tapeVar;
            op->indices_.insert(op->indices_.begin(),
                                versions_.at(StmtOrExprID(_op, parent_)));
        }
    }
    return op;
}

Stmt Grad::visit(const StmtSeq &op) {
    if (isRecompute_) {
        return BaseClass::visit(op);
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
        return makeStmtSeq(std::move(stmts), makeMetadata("grad", op));
    }
}

Stmt Grad::visit(const For &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    ReplaceByTape replaceByTape(*this, tapeMap_, versions_, op);
    if (isRecompute_) {
        op->begin_ = replaceByTape(op->begin_);
        op->end_ = replaceByTape(op->end_);
        op->step_ = replaceByTape(op->step_);
        op->len_ = replaceByTape(op->len_);
    } else {
        auto noDeps = op->property_->noDeps_;
        for (auto &&fwdVar : op->property_->noDeps_) {
            if (hasDef(fwdVar) && affectedDefs_.count(def(fwdVar)->id())) {
                noDeps.emplace_back(fwdVar + ".grad");
            }
        }
        auto begin = replaceByTape(
            makeAdd(op->begin_,
                    makeMul(op->step_, makeSub(op->len_, makeIntConst(1)))));
        auto end = replaceByTape(makeSub(op->begin_, op->step_));
        auto step = replaceByTape(makeSub(makeIntConst(0), op->step_));
        auto len = replaceByTape(op->len_);

        op->property_->noDeps_ = std::move(noDeps);
        op->begin_ = std::move(begin);
        op->end_ = std::move(end);
        op->step_ = std::move(step);
        op->len_ = std::move(len);
        op->metadata() = makeMetadata("grad", op);
        op->setId();
    }
    return op;
}

Stmt Grad::visit(const If &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();
    ReplaceByTape replaceByTape(*this, tapeMap_, versions_, op);
    op->cond_ = replaceByTape(op->cond_);
    if (!isRecompute_) {
        op->metadata() = makeMetadata("grad", op);
        op->setId();
    }
    return op;
}

Stmt Grad::visit(const Assert &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Assert);
    auto op = __op.as<AssertNode>();
    ReplaceByTape replaceByTape(*this, tapeMap_, versions_, op);
    op->cond_ = replaceByTape(op->cond_);
    if (!isRecompute_) {
        op->metadata() = makeMetadata("grad", op);
        op->setId();
    }
    return op;
}

Stmt Grad::visit(const VarDef &_op) {
    ASSERT(!gradNames_.count(_op->name_));
    ASSERT(!recomputed_.count(_op->name_));
    std::string gradName;
    if (affectedDefs_.count(_op->id())) {
        gradName = gradNames_[_op->name_] = _op->name_ + ".grad";
    }
    if (tapes_.count(_op->id())) {
        taped_.insert(_op->name_);
    }
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    taped_.erase(op->name_);
    gradNames_.erase(op->name_);
    recomputed_.erase(op->name_);

    if (isRecompute_) {
        return op;
    } else {
        VarDef ret = op;

        if (affectedDefs_.count(_op->id())) {
            if (requires_.count(op->name_)) {
                requireGrads_[op->name_] = gradName;
            }
            if (provides_.count(op->name_)) {
                provideGrads_[op->name_] = gradName;
            }

            Stmt grad = op->body_;
            if ((op->buffer_->atype() != AccessType::Output &&
                 op->buffer_->atype() != AccessType::InOut)) {
                // We use reverse order in the init, so it can be better fused
                // with the backward pass
                std::vector<std::string> iters;
                std::vector<Expr> indices;
                int nDim = op->buffer_->tensor()->shape().size();
                iters.reserve(nDim);
                indices.reserve(nDim);
                for (int i = 0; i < nDim; i++) {
                    std::string iter =
                        "." + gradName + ".i" + std::to_string(i);
                    indices.emplace_back(makeVar(iter));
                    iters.emplace_back(std::move(iter));
                }
                auto init =
                    makeStore(gradName, std::move(indices), makeIntConst(0));
                for (int i = nDim - 1; i >= 0; i--) {
                    init = makeFor(iters[i],
                                   makeSub(op->buffer_->tensor()->shape()[i],
                                           makeIntConst(1)),
                                   makeIntConst(-1), makeIntConst(-1),
                                   op->buffer_->tensor()->shape()[i],
                                   Ref<ForProperty>::make(), init);
                }
                grad = makeStmtSeq({init, grad});
            }

            grad = makeVarDef(gradName, op->buffer_, op->ioTensor_, grad,
                              op->pinned_, makeMetadata("grad", op));
            switch (op->buffer_->atype()) {
            case AccessType::Input:
                grad.as<VarDefNode>()->buffer_->setAtype(AccessType::Output);
                break;
            case AccessType::Output:
            case AccessType::InOut:
                grad.as<VarDefNode>()->buffer_->setAtype(AccessType::InOut);
                break;
            case AccessType::Cache:
                break; // do nothing
            default:
                ASSERT(false);
            }

            ret = makeVarDef(op->name_, op->buffer_, op->ioTensor_, grad,
                             op->pinned_, op->metadata(), op->id())
                      .as<VarDefNode>();
        }

        if (ret->buffer_->atype() == AccessType::Output ||
            ret->buffer_->atype() == AccessType::InOut) {
            ret->buffer_->setAtype(AccessType::Cache);
        }
        if (tapeMap_.count(op->id())) {
            auto tapeVar = tapeMap_.at(op->id());
            if (tapeVar != ret->name_) {
                ret = makeVarDef(tapeVar, ret->buffer_, ret->ioTensor_, ret,
                                 ret->pinned_, makeMetadata("tape", ret))
                          .as<VarDefNode>();
                auto &shape = ret->buffer_->tensor()->shape();
                shape.insert(shape.begin(), totLens_.at(op->id()));
            }
            ret.as<VarDefNode>()->buffer_->setAtype(AccessType::Input);
        }

        return ret;
    }
}

Stmt Grad::visit(const Store &op) {
    auto &&b = buffer(op->var_);
    if (isRecompute_) {
        // FIXME: What if an intermediate variable is assigned and used multiple
        // times? E.g. a = x; use a; a = y; use a;
        bool recomputed =
            recomputed_.count(op->var_) && recomputed_.at(op->var_).count(op);
        if (!recomputed && !taped_.count(op->var_)) {
            recomputed_[op->var_].insert(op);
            return ReplaceByTape(*this, tapeMap_, versions_, op)(op);
        } else {
            return makeStmtSeq({});
        }
    } else {
        auto newMetadata = makeMetadata("grad", op);
        std::vector<Stmt> stmts;
        if (gradNames_.count(op->var_)) {
            auto &&grad = gradNames_.at(op->var_);
            auto &&indices = op->indices_;
            if (!allReads(op->expr_).count(op->var_)) {
                ReplaceByTape replaceByTape(*this, tapeMap_, versions_, op);
                // Quick path for acyclic assignment
                GradExpr exprVisitor(
                    replaceByTape, gradNames_, op->expr_,
                    makeLoad(grad, indices, b->tensor()->dtype()),
                    makeLoad(op->var_, indices, b->tensor()->dtype()));
                exprVisitor(op->expr_);

                for (auto &&stmt : exprVisitor.appends()) {
                    stmt->metadata() = newMetadata;
                    stmts.emplace_back(stmt);
                }
                if (notSingleWrite_.count(op)) {
                    stmts.emplace_back(
                        makeStore(grad, indices, makeIntConst(0), newMetadata));
                }
                return makeStmtSeq(std::move(stmts));
            } else {
                // General case
                // Gradient of y[i] = f(x[i], y[i]) is:
                // d_y.old = d_y[i]
                // d_y[i] = 0
                // deduce d_x[i] and d_y[i] using d_y.old
                auto oldGrad = grad + ".old";
                stmts.emplace_back(makeStore(
                    oldGrad, {}, makeLoad(grad, indices, b->tensor()->dtype()),
                    newMetadata));
                stmts.emplace_back(
                    makeStore(grad, indices, makeIntConst(0), newMetadata));

                ReplaceByTape replaceByTape(*this, tapeMap_, versions_, op);
                GradExpr exprVisitor(
                    replaceByTape, gradNames_, op->expr_,
                    makeLoad(oldGrad, {}, b->tensor()->dtype()),
                    makeLoad(op->var_, indices, b->tensor()->dtype()));
                exprVisitor(op->expr_);

                for (auto &&stmt : exprVisitor.appends()) {
                    stmt->metadata() = newMetadata;
                    stmts.emplace_back(stmt);
                }
                return makeVarDef(
                    oldGrad,
                    makeBuffer(makeTensor({}, b->tensor()->dtype()),
                               AccessType::Cache, b->mtype()),
                    nullptr, makeStmtSeq(std::move(stmts)), false);
            }
        } else {
            return makeStmtSeq({});
        }
    }
}

Stmt Grad::visit(const ReduceTo &op) {
    auto &&b = buffer(op->var_);
    if (isRecompute_) {
        // FIXME: What if an intermediate variable is assigned and used multiple
        // times? E.g. a = x; use a; a = y; use a;
        bool recomputed =
            recomputed_.count(op->var_) && recomputed_.at(op->var_).count(op);
        if (!recomputed && b->atype() == AccessType::Cache &&
            !taped_.count(op->var_)) {
            recomputed_[op->var_].insert(op);
            return ReplaceByTape(*this, tapeMap_, versions_, op)(op);
        } else {
            return makeStmtSeq({});
        }
    } else {
        auto newMetadata = makeMetadata("grad", op);
        std::vector<Stmt> stmts;
        if (gradNames_.count(op->var_)) {
            auto &&grad = gradNames_.at(op->var_);
            auto &&indices = op->indices_;
            if (op->op_ == ReduceOp::Add &&
                !allReads(op->expr_).count(op->var_)) {
                ReplaceByTape replaceByTape(*this, tapeMap_, versions_, op);
                // Quick path for canonical reduce sum
                GradExpr exprVisitor(
                    replaceByTape, gradNames_, op->expr_,
                    makeLoad(grad, indices, b->tensor()->dtype()),
                    makeLoad(op->var_, indices, b->tensor()->dtype()));
                exprVisitor(op->expr_);

                for (auto &&stmt : exprVisitor.appends()) {
                    stmt->metadata() = newMetadata;
                    stmts.emplace_back(stmt);
                }
                return makeStmtSeq(std::move(stmts));
            } else {
                ASSERT(false);
            }
        } else {
            return makeStmtSeq({});
        }
    }
}

void GradExpr::visit(const Load &op) {
    Visitor::visit(op);
    if (gradExprs_.count(op) && gradNames_.count(op->var_)) {
        appends_.push_back(makeReduceTo(gradNames_.at(op->var_), op->indices_,
                                        ReduceOp::Add, gradExprs_.at(op),
                                        false));
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
           std::unordered_map<ID, std::string>>
gradBody(const Stmt &_op, const std::unordered_set<std::string> &_requires,
         const std::unordered_set<std::string> &provides,
         const std::unordered_set<ID> &tapes) {

    // expand the scope of each local variable, to avoid unnecessary recomputing
    auto op = hoistVarOverStmtSeq(_op);

    // Simplify before grad. E.g. grad of x^2 is much simpler than x * x
    op = floatSimplify(op);

    // Reduce min and reduce max may need the intermediate value for
    // gradients, but reduce add does not
    op = undoMakeReduction(op); // Because we need to record loadMap
    op = makeReduction(op, {ReduceOp::Add}, true);

    auto [forward, tapeMap, versions, totLens] = outputIntermediates(op, tapes);

    auto affectedDefs = intersect(
        PropagateProvides::propagateUntilConverge(op, _requires, provides),
        PropagateRequires::propagateUntilConverge(op, _requires, provides));

    std::unordered_set<Stmt> notSingleWrite;
    auto foundWAW = [&](const Dependency &d) {
        notSingleWrite.insert(d.earlier().as<StmtNode>());
    };
    FindDeps().type(DEP_WAW).ignoreReductionWAW(false)(op, foundWAW);

    Grad mutator(_requires, provides, tapes, affectedDefs, tapeMap, versions,
                 totLens, notSingleWrite);
    auto backward = mutator(op);

    // We do some basic simplifications here, to reduce burden on auto-schedule
    backward = removeDeadVar(backward);
    backward = propOneTimeUse(backward);
    backward = simplify(backward);
    backward = tensorPropConst(backward);
    backward = removeWrites(backward);
    backward = removeCyclicAssign(backward);

    return std::make_tuple(forward, backward, mutator.requireGrads(),
                           mutator.provideGrads(), tapeMap);
}

template <bool inplace>
static std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                  std::unordered_map<std::string, std::string>>
gradFuncImpl(const Func &func, const std::unordered_set<std::string> &_requires,
             const std::unordered_set<std::string> &provides,
             const std::unordered_set<ID> &tapes) {
    auto [forward, backward, requireGrads, provideGrads, tapeMap] =
        gradBody(func->body_, _requires, provides, tapes);

    std::vector<FuncParam> forwardParams, backwardParams;
    std::vector<FuncRet> backwardRets;
    for (auto &&param : func->params_) {
        auto node = findStmt(func->body_, [&](const Stmt &stmt) {
            return stmt->nodeType() == ASTNodeType::VarDef &&
                   stmt.as<VarDefNode>()->name_ == param.name_;
        });
        if (node.template as<VarDefNode>()->buffer_->atype() ==
            AccessType::Input) {
            auto closureArr = Ref<Ref<Array>>::make();
            // Redirect input arguments from forward to backward
            forwardParams.emplace_back(param.name_, closureArr, true);
            backwardParams.emplace_back(param.name_, closureArr, false);
        } else {
            // Backward does not need a froward's output argument. If needed, it
            // will be found in the tape
            forwardParams.emplace_back(param);
        }
    }

    auto forwardReturns = func->returns_;
    for (auto &&[_oriDef, _tapeName] : tapeMap) {
        auto &&oriDef = _oriDef;
        auto &&tapeName = _tapeName;
        auto def = findStmt(func->body_, oriDef);
        auto tapeDType =
            def.template as<VarDefNode>()->buffer_->tensor()->dtype();
        auto tapeArr = Ref<Ref<Array>>::make(nullptr);
        if (auto iter = std::find_if(
                forwardReturns.begin(), forwardReturns.end(),
                [&](const FuncRet &r) { return r.name_ == tapeName; });
            iter == forwardReturns.end()) {
            forwardReturns.emplace_back(tapeName, tapeDType, tapeArr, false);
        } else {
            // The tape is already a return value
            ASSERT(!iter->isInClosure());
            iter->closure_ = tapeArr;
            iter->returnClosure_ = true;
        }
        backwardParams.emplace_back(tapeName, tapeArr, false);
    }
    auto forwardFunc = makeFunc(func->name_, std::move(forwardParams),
                                std::move(forwardReturns), forward);

    forwardFunc = hoistReturnVars(forwardFunc);

    for (auto &&[x, _dzdx] : requireGrads) {
        auto &&dzdx = _dzdx;
        if (inplace) {
            backwardParams.emplace_back(dzdx, nullptr, false);
        } else {
            auto node = findStmt(backward, [&](const Stmt &stmt) {
                return stmt->nodeType() == ASTNodeType::VarDef &&
                       stmt.as<VarDefNode>()->name_ == dzdx;
            });
            auto dtype =
                node.template as<VarDefNode>()->buffer_->tensor()->dtype();
            backwardRets.emplace_back(dzdx, dtype, nullptr, false);
        }
    }
    for (auto &&[y, dzdy] : provideGrads) {
        backwardParams.emplace_back(dzdy, nullptr, false);
    }
    auto backwardFunc =
        makeFunc(func->name_ + ".grad", std::move(backwardParams),
                 std::move(backwardRets), backward);

    return std::make_tuple(forwardFunc, backwardFunc, requireGrads,
                           provideGrads);
}

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncInplace(const Func &func,
                const std::unordered_set<std::string> &_requires,
                const std::unordered_set<std::string> &provides,
                const std::unordered_set<ID> &tapes) {
    return gradFuncImpl<true>(func, _requires, provides, tapes);
}

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncOutOfPlace(const Func &func,
                   const std::unordered_set<std::string> &_requires,
                   const std::unordered_set<std::string> &provides,
                   const std::unordered_set<ID> &tapes) {
    return gradFuncImpl<false>(func, _requires, provides, tapes);
}

static std::vector<ID> _findTapeDefs(const Stmt &op, GradTapeMode mode) {
    switch (mode) {
    case GradTapeMode::All: {
        std::vector<ID> ret;
        for (auto &&[id, name] :
             allDefs(op, {AccessType::Cache, AccessType::Output,
                          AccessType::InOut})) {
            ret.emplace_back(id);
        }
        return ret;
    }
    case GradTapeMode::Nothing:
        return {};
    case GradTapeMode::NoReuseOnly:
        return allNoReuseDefs(
            op, {AccessType::Cache, AccessType::Output, AccessType::InOut});
    default:
        ASSERT(false);
    }
}

static std::unordered_set<ID> findTapeDefs(const Stmt &op, GradTapeMode mode) {
    auto ret = _findTapeDefs(op, mode);
    return std::unordered_set<ID>(ret.begin(), ret.end());
}

std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<ID, std::string>>
gradBody(const Stmt &op, const std::unordered_set<std::string> &_requires,
         const std::unordered_set<std::string> &provides,
         GradTapeMode tapeMode) {
    return gradBody(op, _requires, provides, findTapeDefs(op, tapeMode));
}

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncInplace(const Func &func,
                const std::unordered_set<std::string> &_requires,
                const std::unordered_set<std::string> &provides,
                GradTapeMode tapeMode) {
    return gradFuncInplace(func, _requires, provides,
                           findTapeDefs(func->body_, tapeMode));
}

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncOutOfPlace(const Func &func,
                   const std::unordered_set<std::string> &_requires,
                   const std::unordered_set<std::string> &provides,
                   GradTapeMode tapeMode) {
    return gradFuncOutOfPlace(func, _requires, provides,
                              findTapeDefs(func->body_, tapeMode));
}

} // namespace freetensor
