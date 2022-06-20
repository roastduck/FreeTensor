#include <analyze/deps.h>
#include <analyze/find_loop_variance.h>
#include <pass/merge_and_hoist_if.h>
#include <schedule/fission.h>

namespace freetensor {

Stmt HoistVar::visitStmt(const Stmt &op) {
    if (before_.isValid()) {
        isAfter_ |= op->id() == before_;
    }
    auto ret = Mutator::visitStmt(op);
    if (after_.isValid()) {
        isAfter_ |= op->id() == after_;
    }
    return ret;
}

Stmt HoistVar::visit(const For &op) {
    if (op->id() != loop_) {
        auto ret = Mutator::visit(op);
        if (inside_) {
            for (auto &&def : defStack_) {
                xLoops_[def->name_].emplace_back(op->id());
            }
            innerLoops_.emplace_back(op->id());
        } else {
            outerScopes_.emplace_back(op->id());
        }
        return ret;
    } else {
        inside_ = true, isAfter_ = false;
        auto ret = Mutator::visit(op);
        inside_ = false;
        for (auto &&def : defStack_) {
            xLoops_[def->name_].emplace_back(op->id());
        }
        innerLoops_.emplace_back(op->id());
        for (auto i = defStack_.rbegin(); i != defStack_.rend(); i++) {
            ret = makeVarDef((*i)->id(), std::move((*i)->name_),
                             std::move(((*i)->buffer_)),
                             std::move((*i)->ioTensor_), ret, (*i)->pinned_);
        }
        return ret;
    }
}

Stmt HoistVar::visit(const StmtSeq &op) {
    if (!inside_) {
        outerScopes_.emplace_back(op->id());
        return Mutator::visit(op);
    } else {
        std::vector<Stmt> before, after;
        for (auto &&_stmt : op->stmts_) {
            bool thisIsAfter = false;
            if (!after.empty()) {
                thisIsAfter = isAfter_;
            }
            auto stmt = (*this)(_stmt);
            if (!before.empty()) {
                thisIsAfter = isAfter_;
            }
            (thisIsAfter ? after : before).emplace_back(stmt);
        }
        Stmt ret;
        if (after.empty()) {
            ret = makeStmtSeq(op->id(), std::move(before));
        } else if (before.empty()) {
            ret = makeStmtSeq(op->id(), std::move(after));
        } else {
            auto beforeNode = before.size() > 1
                                  ? makeStmtSeq("", std::move(before))
                                  : before[0];
            auto afterNode =
                after.size() > 1 ? makeStmtSeq("", std::move(after)) : after[0];
            scopePairs_.emplace_back(beforeNode->id(), afterNode->id());
            ret = makeStmtSeq(op->id(), {beforeNode, afterNode});
        }
        return ret;
    }
}

Stmt HoistVar::visit(const VarDef &_op) {
    if (!inside_) {
        return Mutator::visit(_op);
    } else {
        part0Vars_.erase(_op->name_);
        part1Vars_.erase(_op->name_);
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        if (part0Vars_.count(op->name_) && part1Vars_.count(op->name_)) {
            defStack_.emplace_back(op);
            return op->body_;
        } else {
            return op;
        }
    }
}

Stmt HoistVar::visit(const Store &op) {
    recordAccess(op);
    return Mutator::visit(op);
}

Expr HoistVar::visit(const Load &op) {
    recordAccess(op);
    return Mutator::visit(op);
}

Stmt HoistVar::visit(const ReduceTo &op) {
    recordAccess(op);
    return Mutator::visit(op);
}

Stmt AddDimToVar::visit(const For &op) {
    forMap_[op->id()] = op;
    return BaseClass::visit(op);
}

Stmt AddDimToVar::visit(const VarDef &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);

    auto op = __op.as<VarDefNode>();
    if (toAdd_.count(op->id())) {
        op->buffer_ = deepCopy(op->buffer_);
        auto &shape = op->buffer_->tensor()->shape();
        for (auto &&loop : toAdd_.at(op->id())) {
            shape.insert(shape.begin(), forMap_.at(loop)->len_);
        }
    }
    return op;
}

Stmt AddDimToVar::visit(const Store &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    return doAdd(__op.as<StoreNode>());
}

Expr AddDimToVar::visit(const Load &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    return doAdd(__op.as<LoadNode>());
}

Stmt AddDimToVar::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    return doAdd(__op.as<ReduceToNode>());
}

Stmt FissionFor::visitStmt(const Stmt &op) {
    if (!inside_) {
        return Mutator::visitStmt(op);
    } else {
        auto oldAnyInside = anyInside_;
        anyInside_ = false;
        if (before_.isValid()) {
            isAfter_ |= op->id() == before_;
        }
        anyInside_ |= (isPart0_ && !isAfter_) || (!isPart0_ && isAfter_);
        auto ret = Mutator::visitStmt(op);
        if (after_.isValid()) {
            isAfter_ |= op->id() == after_;
        }
        if (!anyInside_) {
            ret = makeStmtSeq("", {});
        }
        anyInside_ |= oldAnyInside;
        return ret;
    }
}

void FissionFor::markNewId(const Stmt &op, bool isPart0) {
    ID oldId = op->id(), newId;
    if (isPart0) {
        op->setId(newId = oldId.strId() + suffix0_);
        ids0_.emplace(oldId, newId);
    } else {
        op->setId(newId = oldId.strId() + suffix1_);
        ids1_.emplace(oldId, newId);
    }
}

Stmt FissionFor::visit(const For &op) {
    if (op->id() != loop_) {
        auto ret = Mutator::visit(op);
        if (inside_) {
            markNewId(ret, isPart0_);
        }
        return ret;
    } else {
        auto begin = (*this)(op->begin_);
        auto end = (*this)(op->end_);
        auto step = (*this)(op->step_);
        auto len = (*this)(op->len_);
        inside_ = true;
        isPart0_ = true, isAfter_ = false, anyInside_ = false;
        auto part0 = (*this)(op->body_);
        isPart0_ = false, isAfter_ = false, anyInside_ = false;
        auto part1 = (*this)(op->body_);
        inside_ = false;
        auto for0 = makeFor(op->id(), op->iter_, begin, end, step, len,
                            op->property_, part0);
        auto for1 = makeFor(op->id(), op->iter_, begin, end, step, len,
                            op->property_, part1);
        markNewId(for0, true);
        markNewId(for1, false);
        return makeStmtSeq("", {for0, for1});
    }
}

Stmt FissionFor::visit(const StmtSeq &op) {
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

Stmt FissionFor::visit(const VarDef &_op) {
    if (!inside_) {
        return Mutator::visit(_op);
    } else {
        varUses_.erase(_op->name_);
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        Stmt ret = varUses_.count(op->name_) ? __op : (Stmt)op->body_;
        markNewId(ret, isPart0_);
        return ret;
    }
}

Stmt FissionFor::visit(const Store &op) {
    if (inPart()) {
        varUses_.insert(op->var_);
    }
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

Expr FissionFor::visit(const Load &op) {
    if (inPart()) {
        varUses_.insert(op->var_);
    }
    return Mutator::visit(op);
}

Stmt FissionFor::visit(const ReduceTo &op) {
    if (inPart()) {
        varUses_.insert(op->var_);
    }
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

Stmt FissionFor::visit(const If &op) {
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

Stmt FissionFor::visit(const Assert &op) {
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

std::pair<Stmt,
          std::pair<std::unordered_map<ID, ID>, std::unordered_map<ID, ID>>>
fission(const Stmt &_ast, const ID &loop, FissionSide side, const ID &splitter,
        const std::string &suffix0, const std::string &suffix1) {
    // FIXME: Check the condition is not variant when splitting an If

    if (suffix0 == suffix1) {
        throw InvalidSchedule("suffix0 cannot be the same with suffix1");
    }

    HoistVar hoist(loop, side == FissionSide::Before ? splitter : "",
                   side == FissionSide::After ? splitter : "");
    FissionFor mutator(loop, side == FissionSide::Before ? splitter : "",
                       side == FissionSide::After ? splitter : "", suffix0,
                       suffix1);

    auto ast = hoist(_ast);
    if (!hoist.found()) {
        throw InvalidSchedule("Split point " + toString(splitter) +
                              " not found inside " + toString(loop));
    }
    auto &&xLoops = hoist.xLoops();

    auto variantExpr = findLoopVariance(ast);

    // var name -> loop id
    std::vector<FindDepsDir> disjunct;
    for (const ID &inner : hoist.innerLoops()) {
        disjunct.push_back({{inner, DepDirection::Normal}});
    }
    auto isRealWrite = [&](const ID &loop, const VarDef &def) -> bool {
        return isVariant(variantExpr.second, def, loop);
    };
    std::unordered_map<ID, std::vector<ID>> toAdd;
    auto found = [&](const Dependency &d) {
        ASSERT(d.dir_.size() == 1);
        auto &&id = d.dir_[0].first.id_;
        if (!xLoops.count(d.var_) ||
            std::find(xLoops.at(d.var_).begin(), xLoops.at(d.var_).end(), id) ==
                xLoops.at(d.var_).end()) {
            throw InvalidSchedule(toString(d) + " cannot be resolved");
        }
        if (!isRealWrite(id, d.def()) &&
            d.earlier()->nodeType() == ASTNodeType::Load) {
            return;
        }
        if (!isRealWrite(id, d.def()) &&
            d.later()->nodeType() == ASTNodeType::Load) {
            return;
        }
        if (std::find(toAdd[d.defId()].begin(), toAdd[d.defId()].end(), id) ==
            toAdd[d.defId()].end()) {
            toAdd[d.defId()].emplace_back(id);
        }
    };
    FindDeps()
        .direction(disjunct)
        .filterEarlier([&](const AccessPoint &earlier) {
            for (auto &&[beforeId, afterId] : hoist.scopePairs()) {
                if (earlier.stmt_->ancestorById(afterId).isValid()) {
                    return true;
                }
            }
            return false;
        })
        .filterLater([&](const AccessPoint &later) {
            for (auto &&[beforeId, afterId] : hoist.scopePairs()) {
                if (later.stmt_->ancestorById(beforeId).isValid()) {
                    return true;
                }
            }
            return false;
        })(ast, found);

    AddDimToVar adder(toAdd);
    ast = adder(ast);

    ast = mutator(ast);
    ast = mergeAndHoistIf(ast);
    return std::make_pair(ast, std::make_pair(mutator.ids0(), mutator.ids1()));
}

} // namespace freetensor
