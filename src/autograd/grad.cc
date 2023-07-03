#include <analyze/all_defs.h>
#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <autograd/all_no_reuse_defs.h>
#include <autograd/clear_mark_version.h>
#include <autograd/dedup_tape_names.h>
#include <autograd/find_tape_or_recomp_stmts.h>
#include <autograd/grad.h>
#include <autograd/merge_tape_input.h>
#include <autograd/output_intermediates.h>
#include <autograd/propagate_defs_need_grad.h>
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
#include <pass/scalar_prop_const.h>
#include <pass/simplify.h>
#include <pass/tensor_prop_const.h>
#include <pass/undo_make_reduction.h>
#include <schedule/hoist_selected_var.h>

namespace freetensor {

Expr InsertUserGrad::visit(const LoadAtVersion &_op) {
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

Stmt InsertUserGrad::visit(const Store &op) {
    if (localVarDefNames_.count(op->var_) ||
        std::ranges::count(views::values(gradNames_), op->var_)) {
        return Mutator::visit(op);
    } else {
        return makeStmtSeq({});
    }
}

Stmt InsertUserGrad::visit(const ReduceTo &op) {
    if (localVarDefNames_.count(op->var_) ||
        std::ranges::count(views::values(gradNames_), op->var_)) {
        return Mutator::visit(op);
    } else {
        return makeStmtSeq({});
    }
}

Stmt InsertUserGrad::visit(const VarDef &op) {
    localVarDefNames_.emplace(op->name_);
    auto ret = Mutator::visit(op);
    localVarDefNames_.erase(op->name_);
    return ret;
}

ReplaceBySaved Grad::getReplacer(const Stmt &stmt,
                                 const Store &alreadyStored) const {
    return {*this, intermediatesMap_, versions_, stmt->id(), alreadyStored};
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
                    InsertUserGrad replacer{*this, intermediatesMap_,
                                            userVersions_, gradNames_};
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

        // Insert inversion
        if (auto it = inverseStmts_.find(s->id()); it != inverseStmts_.end()) {
            auto replacer = getReplacer(s);
            auto inv = replacer.recomp(it->second.inv_);
            if (inv->nodeType() == ASTNodeType::Store) {
                inverselyUpdated_.insert(def(inv.as<StoreNode>()->var_)->id());
            } else if (inv->nodeType() == ASTNodeType::ReduceTo) {
                inverselyUpdated_.insert(
                    def(inv.as<ReduceToNode>()->var_)->id());
            } else {
                ASSERT(false);
            }
            if (it->second.cond_.isValid()) {
                inv = makeIf(replacer.recomp(it->second.cond_), std::move(inv));
            }
            // Invert first, and then compute gradient. (TODO: this will break
            // after we support inverting Store nodes)
            ret = makeStmtSeq({inv, ret});
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
        op->begin_ = replaceBySaved.recomp(op->begin_);
        op->end_ = replaceBySaved.recomp(op->end_);
        op->step_ = replaceBySaved.recomp(op->step_);
        op->len_ = replaceBySaved.recomp(op->len_);
    } else {
        auto noDeps = op->property_->noDeps_;
        for (auto &&fwdVar : op->property_->noDeps_) {
            if (hasDef(fwdVar) && defsNeedGrad_.count(def(fwdVar)->id())) {
                noDeps.emplace_back(fwdVar + ".grad");
            }
        }
        auto begin = replaceBySaved.grad(
            makeAdd(op->begin_,
                    makeMul(op->step_, makeSub(op->len_, makeIntConst(1)))));
        auto end = replaceBySaved.grad(makeSub(op->begin_, op->step_));
        auto step = replaceBySaved.grad(makeSub(makeIntConst(0), op->step_));
        auto len = replaceBySaved.grad(op->len_);

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
    if (isRecompute_) {
        op->cond_ = replaceBySaved.recomp(op->cond_);
    } else {
        op->cond_ = replaceBySaved.grad(op->cond_);
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
    if (isRecompute_) {
        op->cond_ = replaceBySaved.recomp(op->cond_);
    } else {
        op->cond_ = replaceBySaved.grad(op->cond_);
        op->metadata() = makeMetadata("grad", op);
        op->setId();
    }
    return op;
}

Stmt Grad::visit(const VarDef &_op) {
    ASSERT(!gradNames_.count(_op->name_));
    ASSERT(!recomputed_.count(_op->name_));
    std::string gradName;
    if (defsNeedGrad_.count(_op->id())) {
        gradName = gradNames_[_op->name_] = _op->name_ + ".grad";
    }
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    gradNames_.erase(op->name_);
    recomputed_.erase(op->name_);

    VarDef ret = op;
    if (isRecompute_) {
        ret->buffer_->setAtype(removeOutputting(ret->buffer_->atype()));
    } else {
        if (defsNeedGrad_.count(_op->id())) {
            if (requires_.count(op->name_)) {
                requireGrads_[op->name_] = gradName;
            }
            if (provides_.count(op->name_)) {
                provideGrads_[op->name_] = gradName;
            }

            Stmt grad = op->body_;
            if (!isOutputting(op->buffer_->atype())) {
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
            grad.as<VarDefNode>()->buffer_->tensor()->setDType(
                grad.as<VarDefNode>()->buffer_->tensor()->dtype().base());
            if (isOutputting(op->buffer_->atype())) {
                grad.as<VarDefNode>()->buffer_->setAtype(
                    resetProvidedGrad_ ? AccessType::InOut
                                       : AccessType::InputMutable);
            } else if (isInputting(op->buffer_->atype())) {
                grad.as<VarDefNode>()->buffer_->setAtype(AccessType::Output);
            }

            ret = makeVarDef(op->name_, op->buffer_, op->viewOf_, grad,
                             op->pinned_, op->metadata(), op->id())
                      .as<VarDefNode>();
        }

        ret->buffer_->setAtype(removeOutputting(ret->buffer_->atype()));
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
        ret.as<VarDefNode>()->buffer_->setAtype(
            inverselyUpdated_.count(op->id()) ? AccessType::InputMutable
                                              : AccessType::Input);
    }

    return ret;
}

Stmt Grad::visit(const Store &op) {
    auto &&b = buffer(op->var_);
    auto replaceBySaved = getReplacer(op, op);
    if (isRecompute_) {
        bool recomputed =
            recomputed_.count(op->var_) && recomputed_.at(op->var_).count(op);
        if (!recomputed && !tapes_.count(def(op->var_)->id())) {
            recomputed_[op->var_].insert(op);
            return replaceBySaved.recomp(op);
        } else {
            return makeStmtSeq({});
        }
    } else {
        // This is the statement used to store all versions of recomputation. No
        // use in gradients
        if (saveLocalStmts_.count(op->id())) {
            return makeStmtSeq({});
        }

        // May be different by sign with forward variable
        auto gradDType = b->tensor()->dtype().base();

        auto newMetadata = makeMetadata("grad", op);
        std::vector<Stmt> stmts;
        if (gradNames_.count(op->var_)) {
            auto &&grad = gradNames_.at(op->var_);
            auto &&indices = op->indices_;
            if (!allReads(op->expr_).count(op->var_)) {
                // Quick path for acyclic assignment
                if (auto it = derivatives_.find(StmtOrExprID{op->expr_, op});
                    it != derivatives_.end()) {
                    for (auto &&stmt : it->second.genGrads(
                             intermediatesMap_, versions_, gradNames_,
                             makeLoad(grad, indices, gradDType))) {
                        stmt->metadata() = newMetadata;
                        stmts.emplace_back(stmt);
                    }
                }
                if (notSingleWrite_.count(op) ||
                    (isOutputting(b->atype()) && resetProvidedGrad_)) {
                    // Reset gradient of `y` in assignement `y = f(x)` if:
                    // 1. `y` is assigned more than once.
                    // OR
                    // 2. `y` is a final output and `resetProvidedGrad_` is
                    // true.
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
                stmts.emplace_back(makeStore(oldGrad, {},
                                             makeLoad(grad, indices, gradDType),
                                             newMetadata));
                stmts.emplace_back(
                    makeStore(grad, indices, makeIntConst(0), newMetadata));

                if (auto it = derivatives_.find(StmtOrExprID{op->expr_, op});
                    it != derivatives_.end()) {
                    for (auto &&stmt : it->second.genGrads(
                             intermediatesMap_, versions_, gradNames_,
                             makeLoad(oldGrad, {}, gradDType))) {
                        stmt->metadata() = newMetadata;
                        stmts.emplace_back(stmt);
                    }
                }
                return makeVarDef(oldGrad,
                                  makeBuffer(makeTensor({}, gradDType),
                                             AccessType::Cache, b->mtype()),
                                  std::nullopt, makeStmtSeq(std::move(stmts)),
                                  false);
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
            !tapes_.count(def(op->var_)->id())) {
            recomputed_[op->var_].insert(op);
            return replaceBySaved.recomp(op);
        } else {
            return makeStmtSeq({});
        }
    } else {
        // Quick path for reduce sum/min/max. Other reductions have been
        // converted back to Store

        // May be different by sign with forward variable
        auto gradDType = b->tensor()->dtype().base();

        auto newMetadata = makeMetadata("grad", op);
        if (gradNames_.count(op->var_)) {
            auto &&grad = gradNames_.at(op->var_);
            auto &&indices = op->indices_;
            ASSERT(!allReads(op->expr_).count(
                op->var_)); // Canonical reductions only, see
                            // pass/make_reduction
            switch (op->op_) {
            case ReduceOp::Add: {
                std::vector<Stmt> stmts;
                if (auto it = derivatives_.find(StmtOrExprID{op->expr_, op});
                    it != derivatives_.end()) {
                    for (auto &&stmt : it->second.genGrads(
                             intermediatesMap_, versions_, gradNames_,
                             makeLoad(grad, indices, gradDType))) {
                        stmt->metadata() = newMetadata;
                        stmts.emplace_back(stmt);
                    }
                    return makeStmtSeq(std::move(stmts));
                }
            }

            case ReduceOp::Min:
            case ReduceOp::Max: {
                std::vector<Stmt> stmts;
                if (auto it = derivatives_.find(StmtOrExprID{op->expr_, op});
                    it != derivatives_.end()) {
                    for (auto &&stmt : it->second.genGrads(
                             intermediatesMap_, versions_, gradNames_,
                             makeLoad(grad, indices, gradDType))) {
                        ASSERT(stmt->nodeType() == ASTNodeType::ReduceTo);
                        auto &&oldReduceGrad = stmt.as<ReduceToNode>();
                        ASSERT(oldReduceGrad->op_ == ReduceOp::Add);
                        auto xi = replaceBySaved.grad(op->expr_);
                        // We need to pretend as if we are re-loading the final
                        // value of `y` from a `Store` node. So `ReplaceBySaved`
                        // can recognize it to use the `Store` versions, not
                        // `Load` versions
                        auto fakeFinalYVal =
                            makeIntrinsic("fake_final_y_val", {},
                                          b->tensor()->dtype(), false);
                        auto fakeFinalYStore =
                            makeStore(op->var_, indices, fakeFinalYVal,
                                      op->metadata(), op->id())
                                .as<StoreNode>();
                        auto y =
                            ReplaceBySaved{*this, intermediatesMap_, versions_,
                                           op->id(), fakeFinalYStore}
                                .grad(fakeFinalYVal);
                        // dz/dx[i] += x[i] == y ? dz/dy : 0
                        auto dxi = makeReduceTo(
                            oldReduceGrad->var_, oldReduceGrad->indices_,
                            ReduceOp::Add,
                            makeIfExpr(makeEQ(xi, y), oldReduceGrad->expr_,
                                       makeIntConst(0)),
                            false, oldReduceGrad->metadata());
                        // dz/dy = x[i] == y ? 0 : dz/dy (Reset dz/dy to 0 after
                        // use)
                        auto dy = makeStore(
                            grad, indices,
                            makeIfExpr(makeEQ(xi, y), makeIntConst(0),
                                       makeLoad(grad, indices, gradDType)));
                        auto newStmt = makeStmtSeq({dxi, dy});
                        newStmt->metadata() = newMetadata;
                        stmts.emplace_back(newStmt);
                    }
                }
                return makeStmtSeq(std::move(stmts));
            }

            default:
                // 1. ReduceOp::LAnd and ReduceOp::LOr should not be operating
                // floating point operands
                // 2. For ReduceOp::Mul, theoratically it is better to compute
                // AD for `y *= x[i]` by `dz/dx += dz/dy * y / x[i]`, but how
                // should we update `dz/dy` for each statement? For example,
                // what if we have two statement like `y *= x1[i]; y *= x2[i]`?
                // We also need to ensure `x[i] != 0`. (TODO)
                ASSERT(false);
            }
        } else {
            return makeStmtSeq({});
        }
    }
}

std::tuple<Stmt, Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>,
           std::unordered_map<ID, std::string>>
gradBody(const Stmt &_op, const std::unordered_set<std::string> &_requires,
         const std::unordered_set<std::string> &provides,
         const TapeStrategy &tapeStrategy, bool resetProvidedGrad, bool invert,
         const std::vector<StmtSetToUserGrad> &stmtSetToUserGrads) {
    auto &&tapes = tapeStrategy.getIdsToTape(_op);

    // expand the scope of each local variable, to avoid unnecessary recomputing
    auto op = hoistVarOverStmtSeq(_op);

    // Simplify before grad. E.g. grad of x^2 is much simpler than x * x
    op = floatSimplify(op);

    // Reduce mul may need the intermediate value for gradients, but reduce
    // add/sub/min/max does not
    op = undoMakeReduction(op); // Because we need to record loadMap
    op = makeReduction(op, {ReduceOp::Add, ReduceOp::Min, ReduceOp::Max}, true);

    // Since we are outputing all tapes, and output variables can't have same
    // names, we need to deduplicate the names of variables that needs to be
    // taped
    op = dedupTapeNames(op, tapes);

    Derivative derivative;
    derivative(op);
    auto derivatives = derivative.derivatives();

    auto defsNeedGrad = propagateDefsNeedGrad(op, _requires, provides);

    auto &&[idsToTape, idsToRecomp] =
        findTapeOrRecompStmts(op, tapes, defsNeedGrad, derivatives);

    // We can reduce the number of statements in `idsToRecomp` by recovering
    // them through the inversion of their follower statements.
    std::unordered_map<ID, InversionInfo> toInvert;
    if (invert) {
        std::unordered_map<ID, InversionInfo> toInvertFromTape,
            toInvertFromRecomp;
        std::tie(op, toInvertFromTape) =
            invertStmts(op, &idsToTape, &derivatives);
        std::tie(op, toInvertFromRecomp) =
            invertStmts(op, &idsToRecomp, &derivatives);
        toInvert = toInvertFromTape;
        toInvert.merge(toInvertFromRecomp);
    }

    // Save all versions of intermediates variables. If the variables are set to
    // be in tapes, save it in an output tensor globally in the forward pass.
    // The saved tensor in this case is called tapes. If not, save it in an
    // temporary (cache) tensor locally in the backward pass during
    // recomputation
    auto allIntermediates = allDefs(op, {AccessType::Cache});
    auto [forward, tapeMap, versionsGlobal, totLensGlobal, _,
          userVersionsGlobal] =
        outputIntermediates(op, idsToTape, derivatives,
                            OutputIntermediatesStage::Forward, ".tape");
    auto [backward, intermediatesMapLocal, versionsLocal, totLensLocal,
          saveLocalStmts, userVersionsLocal] =
        outputIntermediates(op, idsToRecomp, derivatives,
                            OutputIntermediatesStage::Backward, ".recomp");
    auto intermediatesMap = tapeMap;
    intermediatesMap.merge(std::move(intermediatesMapLocal));
    auto versions = std::move(versionsGlobal);
    versions.merge(std::move(versionsLocal));
    auto totLens = std::move(totLensGlobal);
    totLens.merge(std::move(totLensLocal));
    auto userVersions = std::move(userVersionsGlobal);
    userVersions.merge(std::move(userVersionsLocal));

    std::unordered_set<Stmt> notSingleWrite;
    auto foundWAW = [&](const Dependence &d) {
        notSingleWrite.insert(d.earlier().as<StmtNode>());
    };
    FindDeps().type(DEP_WAW).ignoreReductionWAW(false)(backward, foundWAW);

    std::vector<RangeToUserGrad> rangeToUserGrads;
    rangeToUserGrads.reserve(stmtSetToUserGrads.size());
    for (auto &&stmtSetToUserGrad : stmtSetToUserGrads) {
        if (auto &&range =
                getRangeFromStmtSeq(backward, stmtSetToUserGrad.oriStmts_);
            range.has_value()) {
            rangeToUserGrads.emplace_back(range->first, range->second,
                                          stmtSetToUserGrad.bwdBody_);
        }
    }
    Grad mutator(derivatives, _requires, provides, tapes, defsNeedGrad,
                 intermediatesMap, versions, userVersions, totLens,
                 saveLocalStmts, notSingleWrite, resetProvidedGrad, toInvert,
                 rangeToUserGrads);
    backward = mutator(backward);

    // A backward program may re-input the same taped variable multiple times.
    // We need to merge these "input" VarDef nodes as one
    backward = mergeTapeInput(backward);

    // Clear unused MarkVersion nodes
    forward = clearMarkVersion(forward);
    backward = clearMarkVersion(backward);

    // We do some basic simplifications here, to reduce burden on auto-schedule
    backward = scalarPropConst(backward);
    backward = simplify(backward);
    backward = removeWrites(backward);
    backward = propOneTimeUse(backward);
    backward = simplify(backward);
    backward = tensorPropConst(backward);
    backward = removeCyclicAssign(backward);
    backward = removeDeadVar(backward);

    return std::make_tuple(forward, backward, mutator.requireGrads(),
                           mutator.provideGrads(), tapeMap);
}

template <bool inplace>
static std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
                  std::unordered_map<std::string, std::string>>
gradFuncImpl(const Func &func, const std::unordered_set<std::string> &_requires,
             const std::unordered_set<std::string> &provides,
             const TapeStrategy &tapeStrategy, bool tapeInClosure,
             bool resetProvidedGrad, bool invert,
             const std::vector<StmtSetToUserGrad> &userGrads) {
    auto &&tapes = tapeStrategy.getIdsToTape(func->body_);

    auto [forward, backward, requireGrads, provideGrads, tapeMap] =
        gradBody(func->body_, _requires, provides, tapes, resetProvidedGrad,
                 invert, userGrads);

    std::vector<FuncParam> forwardParams, backwardParams;
    std::vector<FuncRet> backwardRets;
    for (auto &&param : func->params_) {
        auto node = findStmt(func->body_, [&](const Stmt &stmt) {
            return stmt->nodeType() == ASTNodeType::VarDef &&
                   stmt.as<VarDefNode>()->name_ == param.name_;
        });
        auto atype = node.template as<VarDefNode>()->buffer_->atype();
        switch (atype) {
        case AccessType::Input:
        case AccessType::InputMutable:
            if (tapeInClosure) {
                auto closureArr = Ref<Ref<Array>>::make();
                // Redirect Input or InputMutable arguments from forward to
                // backward. Although we can re-store input data to a tape (our
                // taping algorithm stores to tape both on reading and writing),
                // there is no need.
                forwardParams.emplace_back(param.name_, closureArr, true);
                backwardParams.emplace_back(param.name_, closureArr, false);
            } else {
                forwardParams.emplace_back(param.name_, nullptr, false);
                backwardParams.emplace_back(param.name_, nullptr, false);
            }
            break;
        case AccessType::InOut:
            if (!tapes.count(node->id())) {
                throw InvalidAutoGrad("InOut variable " + param.name_ +
                                      " must be in tapes");
            }
            [[fallthrough]];
        case AccessType::Output:
        case AccessType::Bypass:
            // Backward does not need a froward's Output, InOut or Bypass
            // argument. If needed, it will be found in the tape
            forwardParams.emplace_back(param);
            break;
        default:
            ERROR("Parameter " + param.name_ + " should not be a \"" +
                  toString(atype) + "\" variable");
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
                // Add a new closured return
                forwardReturns.emplace_back(tapeName, tapeDType, tapeArr,
                                            false);
            } else {
                // The tape is already a return value, mark it to update closure
                ASSERT(!iter->isInClosure());
                iter->closure_ = tapeArr;
                iter->returnClosure_ = true;
            }
            backwardParams.emplace_back(tapeName, tapeArr, false);
        } else {
            // No matter if the variable is already a return value, always add a
            // new return, so we can easily identify these returns. This is OK
            // because we support returning the same variable to multiple
            // positions
            forwardReturns.emplace_back(tapeName, tapeDType, nullptr, false);
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
                const TapeStrategy &tapes, bool tapeInClosure,
                bool resetProvidedGrad, bool invert,
                const std::vector<StmtSetToUserGrad> &userGrads) {
    return gradFuncImpl<true>(func, _requires, provides, tapes, tapeInClosure,
                              resetProvidedGrad, invert, userGrads);
}

std::tuple<Func, Func, std::unordered_map<std::string, std::string>,
           std::unordered_map<std::string, std::string>>
gradFuncOutOfPlace(const Func &func,
                   const std::unordered_set<std::string> &_requires,
                   const std::unordered_set<std::string> &provides,
                   const TapeStrategy &tapes, bool tapeInClosure,
                   bool resetProvidedGrad, bool invert,
                   const std::vector<StmtSetToUserGrad> &userGrads) {
    return gradFuncImpl<false>(func, _requires, provides, tapes, tapeInClosure,
                               resetProvidedGrad, invert, userGrads);
}

} // namespace freetensor
