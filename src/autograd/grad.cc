#include <analyze/all_defs.h>
#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <autograd/all_no_reuse_defs.h>
#include <autograd/clear_mark_version.h>
#include <autograd/dedup_tape_names.h>
#include <autograd/grad.h>
#include <autograd/merge_tape_input.h>
#include <autograd/output_intermediates.h>
#include <container_utils.h>
#include <pass/float_simplify.h>
#include <pass/hoist_return_vars.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/make_nested_loops.h>
#include <pass/make_reduction.h>
#include <pass/prop_one_time_use.h>
#include <pass/remove_cyclic_assign.h>
#include <pass/remove_dead_var.h>
#include <pass/remove_writes.h>
#include <pass/simplify.h>
#include <pass/tensor_prop_const.h>
#include <pass/undo_make_reduction.h>
#include <schedule/hoist_selected_var.h>

namespace freetensor {

namespace {

class RemoveTape : public Mutator {
    std::string name_;

  public:
    RemoveTape(const std::string &name) : name_(name) {}

  protected:
    Stmt visit(const VarDef &op) override {
        return op->name_ == name_ ? (*this)(op->body_) : Mutator::visit(op);
    }

    Expr visit(const Load &op) override {
        ASSERT(op->var_ != name_);
        return Mutator::visit(op);
    }

    Stmt visit(const Store &op) override {
        return op->var_ == name_ ? makeStmtSeq({}) : Mutator::visit(op);
    }

    Stmt visit(const ReduceTo &op) override {
        return op->var_ == name_ ? makeStmtSeq({}) : Mutator::visit(op);
    }
};

/**
 * Remove all use of a tape
 */
inline Stmt removeTape(const Stmt &op, const std::string &name) {
    return RemoveTape{name}(op);
}

class UndoOutputTape : public Mutator {
    std::string name_;
    AccessType atype_;

  public:
    UndoOutputTape(const std::string &name, AccessType atype)
        : name_(name), atype_(atype) {}

  protected:
    Stmt visit(const VarDef &_op) override {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        if (op->name_ == name_) {
            op->buffer_->setAtype(atype_);
        }
        return op;
    }
};

/**
 * Convert a trivial tape back to its original AccessType
 */
inline Stmt undoOutputTape(const Stmt &op, const std::string &name,
                           AccessType atype) {
    return UndoOutputTape{name, atype}(op);
}

} // Anonymous namespace

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
    for (auto &&name : _requires) {
        try {
            findStmt(op, [&](const Stmt &s) {
                return s->nodeType() == ASTNodeType::VarDef &&
                       s.as<VarDefNode>()->name_ == name;
            });
        } catch (const UnexpectedQueryResult &e) {
            throw InvalidAutoGrad("Input variable requesting for gradient `" +
                                  name + "` is not found or duplicated");
        }
    }
    for (auto &&name : provides) {
        try {
            findStmt(op, [&](const Stmt &s) {
                return s->nodeType() == ASTNodeType::VarDef &&
                       s.as<VarDefNode>()->name_ == name;
            });
        } catch (const UnexpectedQueryResult &e) {
            throw InvalidAutoGrad("Output variable providing gradient `" +
                                  name + "` is not found or duplicated");
        }
    }

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

Expr ReplaceBySaved::replaceForwardValue(const Expr &_equLoad) {
    auto __equLoad = deepCopy(_equLoad);
    ASSERT(__equLoad->nodeType() == ASTNodeType::Load);
    auto equLoad = __equLoad.as<LoadNode>();
    if (intermediatesMap_.count(symbolTable_.def(equLoad->var_)->id()) &&
        versions_.count(parent_->id())) {
        auto savedVar =
            intermediatesMap_.at(symbolTable_.def(equLoad->var_)->id());
        if (savedVar != equLoad->var_) {
            equLoad->var_ = savedVar;
            equLoad->indices_.insert(equLoad->indices_.begin(),
                                     versions_.at(parent_->id()));
        }
    }
    return equLoad;
}

Expr ReplaceBySaved::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (intermediatesMap_.count(symbolTable_.def(_op->var_)->id()) &&
        versions_.count(StmtOrExprID(_op, parent_))) {
        auto savedVar = intermediatesMap_.at(symbolTable_.def(_op->var_)->id());
        if (savedVar != op->var_) {
            op->var_ = savedVar;
            op->indices_.insert(op->indices_.begin(),
                                versions_.at(StmtOrExprID(_op, parent_)));
        }
    }
    return op;
}

Expr ReplaceLoadAtVersion::visit(const LoadAtVersion &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::LoadAtVersion);
    auto op = __op.as<LoadAtVersionNode>();
    auto &&[var, version] = userVersions_.at(op->tapeName_);
    if (auto it = intermediatesMap_.find(symbolTable_.def(var)->id());
        it != intermediatesMap_.end()) {
        auto &&savedVar = it->second;
        if (savedVar != var) { // Non-trivial saved vars
            std::vector<Expr> indices = {version};
            indices.insert(indices.end(), op->indices_.begin(),
                           op->indices_.end());
            return makeLoad(savedVar, std::move(indices),
                            symbolTable_.buffer(var)->tensor()->dtype());
        } else { // Trivial saved vars
            return makeLoad(var, op->indices_,
                            symbolTable_.buffer(var)->tensor()->dtype());
        }
    } else { // Input vars
        return makeLoad(var, op->indices_,
                        symbolTable_.buffer(var)->tensor()->dtype());
    }
}

ReplaceBySaved Grad::getReplacer(const Stmt &stmt) const {
    return {*this, intermediatesMap_, versions_, stmt};
}

Stmt Grad::doVisitStmt(const Stmt &s) {
    if (isRecompute_) {
        return BaseClass::visitStmt(s);
    } else {
        // Check for users' backward. Try insert the users' backward at the last
        // statement in the range (which we will meet first, because we are
        // traversing reversedly). If it is a VarDef, we will then skip it and
        // try inner statements
        if (auto it = std::find_if(
                userGrads_.begin(), userGrads_.end(),
                [&](auto &&item) { return item.oriEnd_ == s->id(); });
            it != userGrads_.end()) {
            if (!userGradOpen_.has_value()) {
                userGradOpen_ = *it;
                userGradInsertPos_ = it->oriEnd_;
                userGrads_.erase(it);
            } else {
                throw InvalidAutoGrad(
                    "Ranges of different custom gradients should not overlap");
            }
        }

        Stmt ret;
        // We are in the custom backward range. Use users' backward only. No
        // automatic backward
        if (userGradOpen_.has_value()) {
            // Insert users' backward
            if (userGradInsertPos_ == s->id()) {
                ASSERT(userGradOpen_.has_value());
                auto &&[_1, _2, bwdBody] = *userGradOpen_;

                // 1. Hoist `VarDef`s so `LoadAtVersion` can acutally load from
                // something. Specifically, hoist `VarDef` nodes in this subtree
                // which have non-`VarDef` parents, (TODO: We currently hoist
                // all `VarDef`s, but we only need to hoist those used by
                // `LoadAtVersion`s)
                Stmt hoisted = hoistSelectedVar(s, "<-!<VarDef>");

                // 2. We need `visit(VarDef)` to handle all the hoisted `VarDef`
                // nodes, so we skip them
                Stmt sub = hoisted;
                while (sub->nodeType() == ASTNodeType::VarDef) {
                    sub = sub.as<VarDefNode>()->body_;
                }
                if (sub != hoisted) {
                    userGradInsertPos_ = sub->id();
                    ret = (*this)(hoisted);
                } else {

                    // 3. Plug in the user-defined backward
                    ReplaceLoadAtVersion replacer{*this, intermediatesMap_,
                                                  userVersions_};
                    ret = replacer(bwdBody);
                    userGradInsertPos_ = ID(); // Mark the insertion is done
                }
            } else { // In the range of custom backward, but not at the position
                     // to insert
                if (!s->children().empty()) { // Enter scopes as usual
                    ret = BaseClass::visitStmt(s);
                } else { // Ignore automatic backward for leaf nodes
                    ret = makeStmtSeq({});
                }
            }
        } else { // Automatic backward
            ret = BaseClass::visitStmt(s);
        }

        // Check for the end (actually the beginning, because we are traversing
        // reversedly) of users' backward
        if (userGradOpen_.has_value() && userGradOpen_->oriBegin_ == s->id()) {
            if (userGradInsertPos_.isValid()) {
                ERROR("Failed to insert custom backward for statements " +
                      toString(userGradOpen_->oriBegin_) + " to " +
                      toString(userGradOpen_->oriEnd_));
            }
            userGradOpen_ = std::nullopt;
        }

        return ret;
    }
}

Stmt Grad::visitStmt(const Stmt &s) {
    // Trigger recomputation here. We only trigger for the first-level scope
    // of the program, and inside each `VarDef` nest:
    //
    // // TOP LEVEL RECOMPUTE
    // StmtSeq {
    //   StmtSeq {
    //     VarDef {
    //       VarDef {
    //         // RECOMPUTE
    //         StmtSeq {
    //           StmtSeq {
    //             ...
    // }}}}}}
    //
    // In other words, we don't need to recompute in inner non-`VarDef` scopes,
    // because things are already recomputed. We don't need to recompute inside
    // a `VarDef` node either if its only body is another `VarDef` node, because
    // everything we have recomputed will be dropped after we exit the
    // sub-`VarDef` scope.

    if (isRecompute_) { // Already recomputing
        return doVisitStmt(s);
    } else {
        if (s->nodeType() != ASTNodeType::VarDef) {
            if (auto &&p = s->parentStmt();
                !p.isValid() || p->nodeType() == ASTNodeType::VarDef) {
                isRecompute_ = true;
                auto recomp = doVisitStmt(s);
                isRecompute_ = false;
                auto grad = doVisitStmt(s);
                return makeStmtSeq({recomp, grad});
            }
        }
        return doVisitStmt(s);
    }
}

Stmt Grad::visit(const StmtSeq &op) {
    if (isRecompute_) {
        return BaseClass::visit(op);
    } else {
        std::vector<Stmt> stmts;
        stmts.reserve(op->stmts_.size());
        for (auto &&stmt : views::reverse(op->stmts_)) {
            stmts.emplace_back((*this)(stmt));
        }
        return makeStmtSeq(std::move(stmts), makeMetadata("grad", op));
    }
}

Stmt Grad::visit(const For &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    auto replaceBySaved = getReplacer(op);
    if (isRecompute_) {
        op->begin_ = replaceBySaved(op->begin_);
        op->end_ = replaceBySaved(op->end_);
        op->step_ = replaceBySaved(op->step_);
        op->len_ = replaceBySaved(op->len_);
    } else {
        auto noDeps = op->property_->noDeps_;
        for (auto &&fwdVar : op->property_->noDeps_) {
            if (hasDef(fwdVar) && affectedDefs_.count(def(fwdVar)->id())) {
                noDeps.emplace_back(fwdVar + ".grad");
            }
        }
        auto begin = replaceBySaved(
            makeAdd(op->begin_,
                    makeMul(op->step_, makeSub(op->len_, makeIntConst(1)))));
        auto end = replaceBySaved(makeSub(op->begin_, op->step_));
        auto step = replaceBySaved(makeSub(makeIntConst(0), op->step_));
        auto len = replaceBySaved(op->len_);

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
    auto replaceBySaved = getReplacer(op);
    op->cond_ = replaceBySaved(op->cond_);
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
    auto replaceBySaved = getReplacer(op);
    op->cond_ = replaceBySaved(op->cond_);
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

    VarDef ret = op;
    if (!isRecompute_) {
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
                // Initialize gradients to 0. Later when we compute gradients
                // for each statements, we add the local result to it. This is
                // because when `y = f(x)` and `z = g(x)` both exist, `dw/dx =
                // dw/dy dy/dx + dw/dz dz/dx`
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
                auto init = makeNestedLoops(
                    iters, views::repeat(makeIntConst(0)),
                    op->buffer_->tensor()->shape(),
                    views::repeat(makeIntConst(1)),
                    op->buffer_->tensor()->shape(),
                    views::repeat(Ref<ForProperty>::make()),
                    makeStore(gradName, std::move(indices), makeIntConst(0)));
                grad = makeStmtSeq({init, grad});
            }

            grad = makeVarDef(gradName, op->buffer_, std::nullopt, grad,
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

            ret = makeVarDef(op->name_, op->buffer_, op->viewOf_, grad,
                             op->pinned_, op->metadata(), op->id())
                      .as<VarDefNode>();
        }

        if (ret->buffer_->atype() == AccessType::Output ||
            ret->buffer_->atype() == AccessType::InOut) {
            ret->buffer_->setAtype(AccessType::Cache);
        }
    }

    if (tapes_.count(op->id()) && intermediatesMap_.count(op->id())) {
        auto tapeVar = intermediatesMap_.at(op->id());
        if (tapeVar != ret->name_) {
            ret = makeVarDef(tapeVar, ret->buffer_, std::nullopt, ret,
                             ret->pinned_, makeMetadata("tape", ret))
                      .as<VarDefNode>();
            auto &shape = ret->buffer_->tensor()->shape();
            shape.insert(shape.begin(), totLens_.at(op->id()));
        }
        ret.as<VarDefNode>()->buffer_->setAtype(AccessType::Input);
    }

    return ret;
}

Stmt Grad::visit(const Store &op) {
    auto &&b = buffer(op->var_);
    auto replaceBySaved = getReplacer(op);
    if (isRecompute_) {
        bool recomputed =
            recomputed_.count(op->var_) && recomputed_.at(op->var_).count(op);
        if (!recomputed && !taped_.count(op->var_)) {
            recomputed_[op->var_].insert(op);
            return replaceBySaved(op);
        } else {
            return makeStmtSeq({});
        }
    } else {
        // This is the statement used to store all versions of recomputation. No
        // use in gradients
        if (saveLocalStmts_.count(op->id())) {
            return makeStmtSeq({});
        }

        auto newMetadata = makeMetadata("grad", op);
        std::vector<Stmt> stmts;
        if (gradNames_.count(op->var_)) {
            auto &&grad = gradNames_.at(op->var_);
            auto &&indices = op->indices_;
            if (!allReads(op->expr_).count(op->var_)) {
                // Quick path for acyclic assignment
                GradExpr exprVisitor(
                    replaceBySaved, gradNames_, op->expr_,
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

                GradExpr exprVisitor(
                    replaceBySaved, gradNames_, op->expr_,
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
                    std::nullopt, makeStmtSeq(std::move(stmts)), false);
            }
        } else {
            return makeStmtSeq({});
        }
    }
}

Stmt Grad::visit(const ReduceTo &op) {
    auto &&b = buffer(op->var_);
    auto replaceBySaved = getReplacer(op);
    if (isRecompute_) {
        bool recomputed =
            recomputed_.count(op->var_) && recomputed_.at(op->var_).count(op);
        if (!recomputed && b->atype() == AccessType::Cache &&
            !taped_.count(op->var_)) {
            recomputed_[op->var_].insert(op);
            return replaceBySaved(op);
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
                // Quick path for canonical reduce sum
                GradExpr exprVisitor(
                    replaceBySaved, gradNames_, op->expr_,
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
        appends_.push_back(makeReduceTo(
            gradNames_.at(op->var_),
            ranges::to<std::vector>(op->indices_ |
                                    views::transform([this](const auto &idx) {
                                        return useForwardVal(idx);
                                    })),
            ReduceOp::Add, gradExprs_.at(op), false));
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
         const std::unordered_set<ID> &tapes,
         const std::vector<RangeToUserGrad> &userGrads) {

    // expand the scope of each local variable, to avoid unnecessary recomputing
    auto op = hoistVarOverStmtSeq(_op);

    // Simplify before grad. E.g. grad of x^2 is much simpler than x * x
    op = floatSimplify(op);

    // Reduce min and reduce max may need the intermediate value for
    // gradients, but reduce add does not
    op = undoMakeReduction(op); // Because we need to record loadMap
    op = makeReduction(op, {ReduceOp::Add}, true);

    // Since we are outputing all tapes, and output variables can't have same
    // names, we need to deduplicate the names of variables that needs to be
    // taped
    op = dedupTapeNames(op, tapes);

    // Save all versions of intermediates variables. If the variables are set to
    // be in tapes, save it in an output tensor globally in the forward pass.
    // The saved tensor in this case is called tapes. If not, save it in an
    // temporary (cache) tensor locally in the backward pass during
    // recomputation
    auto allIntermediates = allDefs(op, {AccessType::Cache});
    auto [forward, tapeMap, versionsGlobal, totLensGlobal, _,
          userVersionsGlobal] =
        outputIntermediates(op, tapes, OutputIntermediatesStage::Forward,
                            ".tape");
    auto [backward, intermediatesMapLocal, versionsLocal, totLensLocal,
          saveLocalStmts, userVersionsLocal] =
        outputIntermediates(
            op,
            diff(ranges::to<std::unordered_set>(allIntermediates | views::keys),
                 tapes),
            OutputIntermediatesStage::Backward, ".recomp");
    auto intermediatesMap = tapeMap;
    intermediatesMap.merge(std::move(intermediatesMapLocal));
    auto versions = std::move(versionsGlobal);
    versions.merge(std::move(versionsLocal));
    auto totLens = std::move(totLensGlobal);
    totLens.merge(std::move(totLensLocal));
    auto userVersions = std::move(userVersionsGlobal);
    userVersions.merge(std::move(userVersionsLocal));

    auto affectedDefs = intersect(PropagateProvides::propagateUntilConverge(
                                      backward, _requires, provides),
                                  PropagateRequires::propagateUntilConverge(
                                      backward, _requires, provides));

    std::unordered_set<Stmt> notSingleWrite;
    auto foundWAW = [&](const Dependence &d) {
        notSingleWrite.insert(d.earlier().as<StmtNode>());
    };
    FindDeps().type(DEP_WAW).ignoreReductionWAW(false)(backward, foundWAW);

    Grad mutator(_requires, provides, tapes, affectedDefs, intermediatesMap,
                 versions, userVersions, totLens, saveLocalStmts,
                 notSingleWrite, userGrads);
    backward = mutator(backward);

    // A backward program may re-input the same taped variable multiple times.
    // We need to merge these "input" VarDef nodes as one
    backward = mergeTapeInput(backward);

    // Some intermediate variables might be taped but turned out to be unneeded
    // in the final backward program. For example, suppose `t` and `u` are an
    // intermediate variables, the gradient of `t` and `u` in `y = t + u` depend
    // only on the gradient of `y`, and is irrelavent to the forward value of
    // `u` or `t`. The fact of whether the taped value is needed or unneeded
    // depend on the math function involved, which is revealed only after doing
    // `Grad`. So, we have to first do `Grad` and then remove unused tapes here
    backward = removeDeadVar(backward);
    auto backwardAllReads = allReads(backward);
    for (auto it = tapeMap.begin(); it != tapeMap.end();) {
        auto &&[oriDefId, tapeName] = *it;
        if (!backwardAllReads.count(tapeName)) {
            auto possibleOldDefNode = findAllStmt(_op, oriDefId);
            if (!possibleOldDefNode.empty()) {
                // This intermediate variable has only one version, which was
                // output directly, and we need to convert it back to its
                // original AccessType
                ASSERT(possibleOldDefNode.size() == 1);
                auto &&oldDefNode = possibleOldDefNode.front();
                ASSERT(oldDefNode->nodeType() == ASTNodeType::VarDef);
                auto oldAType = oldDefNode.as<VarDefNode>()->buffer_->atype();
                forward = undoOutputTape(forward, tapeName, oldAType);
            } else {
                // This intermediate variable has multiple versions, which was
                // copied to a dedicated tape variable, and we can simply delete
                // the tape variable
                forward = removeTape(forward, tapeName);
            }
            backward = removeTape(backward, tapeName);
            it = tapeMap.erase(it);
        } else {
            it++;
        }
    }

    // Clear unused MarkVersion nodes
    forward = clearMarkVersion(forward);
    backward = clearMarkVersion(backward);

    // We do some basic simplifications here, to reduce burden on auto-schedule
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
             const std::unordered_set<ID> &tapes, bool tapeInClosure,
             const std::vector<RangeToUserGrad> &userGrads) {
    auto [forward, backward, requireGrads, provideGrads, tapeMap] =
        gradBody(func->body_, _requires, provides, tapes, userGrads);

    std::vector<FuncParam> forwardParams, backwardParams;
    std::vector<FuncRet> backwardRets;
    for (auto &&param : func->params_) {
        auto node = findStmt(func->body_, [&](const Stmt &stmt) {
            return stmt->nodeType() == ASTNodeType::VarDef &&
                   stmt.as<VarDefNode>()->name_ == param.name_;
        });
        if (node.template as<VarDefNode>()->buffer_->atype() ==
            AccessType::Input) {
            if (tapeInClosure) {
                auto closureArr = Ref<Ref<Array>>::make();
                // Redirect input arguments from forward to backward
                forwardParams.emplace_back(param.name_, closureArr, true);
                backwardParams.emplace_back(param.name_, closureArr, false);
            } else {
                forwardParams.emplace_back(param.name_, nullptr, false);
                backwardParams.emplace_back(param.name_, nullptr, false);
            }
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
        if (tapeInClosure) {
            auto tapeArr = Ref<Ref<Array>>::make(nullptr);
            if (auto iter = std::find_if(
                    forwardReturns.begin(), forwardReturns.end(),
                    [&](const FuncRet &r) { return r.name_ == tapeName; });
                iter == forwardReturns.end()) {
                forwardReturns.emplace_back(tapeName, tapeDType, tapeArr,
                                            false);
            } else {
                // The tape is already a return value
                ASSERT(!iter->isInClosure());
                iter->closure_ = tapeArr;
                iter->returnClosure_ = true;
            }
            backwardParams.emplace_back(tapeName, tapeArr, false);
        } else {
            if (std::find_if(forwardReturns.begin(), forwardReturns.end(),
                             [&](const FuncRet &r) {
                                 return r.name_ == tapeName;
                             }) == forwardReturns.end()) {
                forwardReturns.emplace_back(tapeName, tapeDType, nullptr,
                                            false);
            }
            backwardParams.emplace_back(tapeName, nullptr, false);
        }
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
                const std::unordered_set<ID> &tapes, bool tapeInClosure,
                const std::vector<RangeToUserGrad> &userGrads) {
    return gradFuncImpl<true>(func, _requires, provides, tapes, tapeInClosure,
                              userGrads);
}

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncOutOfPlace(const Func &func,
                   const std::unordered_set<std::string> &_requires,
                   const std::unordered_set<std::string> &provides,
                   const std::unordered_set<ID> &tapes, bool tapeInClosure,
                   const std::vector<RangeToUserGrad> &userGrads) {
    return gradFuncImpl<false>(func, _requires, provides, tapes, tapeInClosure,
                               userGrads);
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
         const std::unordered_set<std::string> &provides, GradTapeMode tapeMode,
         const std::vector<RangeToUserGrad> &userGrads) {
    return gradBody(op, _requires, provides, findTapeDefs(op, tapeMode),
                    userGrads);
}

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncInplace(const Func &func,
                const std::unordered_set<std::string> &_requires,
                const std::unordered_set<std::string> &provides,
                GradTapeMode tapeMode, bool tapeInClosure,
                const std::vector<RangeToUserGrad> &userGrads) {
    return gradFuncInplace(func, _requires, provides,
                           findTapeDefs(func->body_, tapeMode), tapeInClosure,
                           userGrads);
}

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncOutOfPlace(const Func &func,
                   const std::unordered_set<std::string> &_requires,
                   const std::unordered_set<std::string> &provides,
                   GradTapeMode tapeMode, bool tapeInClosure,
                   const std::vector<RangeToUserGrad> &userGrads) {
    return gradFuncOutOfPlace(func, _requires, provides,
                              findTapeDefs(func->body_, tapeMode),
                              tapeInClosure, userGrads);
}

} // namespace freetensor
