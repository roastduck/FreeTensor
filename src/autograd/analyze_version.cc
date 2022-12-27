#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <autograd/analyze_version.h>
#include <pass/flatten_stmt_seq.h>

namespace freetensor {

void CountScopeLen::visit(const Store &op) {
    Visitor::visit(op);
    if (op->var_ == var_ && needTapes_.count(op->id())) {
        scopeLen_[op] = makeIntConst(1);
    }
}

void CountScopeLen::visit(const ReduceTo &op) {
    Visitor::visit(op);
    if (op->var_ == var_ && needTapes_.count(op->id())) {
        scopeLen_[op] = makeIntConst(1);
    }
}

void CountScopeLen::visit(const VarDef &op) {
    if (op->id() == def_) {
        var_ = op->name_;
        Visitor::visit(op);
        var_.clear();
    } else {
        Visitor::visit(op);
    }
    if (scopeLen_.count(op->body_)) {
        scopeLen_[op] = scopeLen_.at(op->body_);
    }
}

void CountScopeLen::visit(const For &op) {
    Visitor::visit(op);
    if (scopeLen_.count(op->body_)) {
        if (affectingScopes_.count(op->id())) {
            scopeLen_[op] = makeMul(scopeLen_.at(op->body_), op->len_);
        } else {
            scopeLen_[op] = scopeLen_.at(op->body_);
        }
    }
}

void CountScopeLen::visit(const StmtSeq &op) {
    Visitor::visit(op);
    Expr len, lastLen;
    for (auto &&stmt : op->stmts_) {
        if (scopeLen_.count(stmt) &&
            (var_.empty() || !allReads(stmt).count(var_))) {
            len = lastLen;
        }
        lastLen = len;
        if (scopeLen_.count(stmt)) {
            len = len.isValid() ? makeAdd(len, scopeLen_.at(stmt))
                                : scopeLen_.at(stmt);
        }
    }
    if (len.isValid()) {
        scopeLen_[op] = len;
    }
}

void CountScopeLen::visit(const If &op) {
    Visitor::visit(op);
    Expr len;
    if (scopeLen_.count(op->thenCase_)) {
        len = scopeLen_.at(op->thenCase_);
    }
    if (op->elseCase_.isValid() && scopeLen_.count(op->elseCase_)) {
        len = len.isValid() ? makeMax(len, scopeLen_.at(op->elseCase_))
                            : scopeLen_.at(op->elseCase_);
    }
    if (len.isValid()) {
        scopeLen_[op] = len;
    }
}

void CountScopeLen::visit(const Assert &op) {
    Visitor::visit(op);
    if (scopeLen_.count(op->body_)) {
        scopeLen_[op] = scopeLen_.at(op->body_);
    }
}

void AnalyzeVersion::visit(const Load &op) {
    BaseClass::visit(op);
    if (op->var_ == var_) {
        versions_[StmtOrExprID(op, curStmt())] =
            makeSub(offset_, makeIntConst(1));
    }
}

void AnalyzeVersion::visit(const Store &op) {
    BaseClass::visit(op);
    if (op->var_ == var_ && needTapes_.count(op->id())) {
        versions_[op->id()] = offset_;
    }
}

void AnalyzeVersion::visit(const ReduceTo &op) {
    BaseClass::visit(op);
    if (op->var_ == var_ && needTapes_.count(op->id())) {
        versions_[op->id()] = offset_;
    }
}

void AnalyzeVersion::visit(const VarDef &op) {
    if (op->id() == def_) {
        var_ = op->name_;
        BaseClass::visit(op);
        var_.clear();
    } else {
        BaseClass::visit(op);
    }
}

void AnalyzeVersion::visit(const For &op) {
    if (affectingScopes_.count(op->id())) {
        auto oldOffset = offset_;
        offset_ = makeAdd(
            offset_,
            makeMul(makeFloorDiv(makeSub(makeVar(op->iter_), op->begin_),
                                 op->step_),
                    scopeLen_.at(op->body_)));
        BaseClass::visit(op);
        offset_ = oldOffset;
    } else {
        BaseClass::visit(op);
    }
}

void AnalyzeVersion::visit(const StmtSeq &op) {
    auto oldOffset = offset_;
    auto lastOffset = offset_;
    for (auto &&stmt : op->stmts_) {
        if (scopeLen_.count(stmt) &&
            (var_.empty() || !allReads(stmt).count(var_))) {
            offset_ = lastOffset;
        }
        lastOffset = offset_;
        if (scopeLen_.count(stmt)) {
            (*this)(stmt);
            offset_ = makeAdd(offset_, scopeLen_.at(stmt));
        } else {
            (*this)(stmt);
        }
    }
    offset_ = oldOffset;
}

std::tuple<std::unordered_map<StmtOrExprID, Expr>, std::unordered_map<ID, Expr>,
           std::unordered_set<ID>>
analyzeVersion(const Stmt &_op, const std::unordered_set<ID> &intermediates,
               bool localVersionsOnly) {
    auto op = flattenStmtSeq(_op);

    std::vector<FindDepsDir> direction;
    for (auto &&scope : findAllStmt(op, "<For>")) {
        direction.push_back({{scope->id(), DepDirection::Normal}});
    }
    std::unordered_map<ID, std::unordered_set<ID>> affectingScopes, needTapes;
    auto found1 = [&](const Dependence &d) {
        ASSERT(d.earlier()->nodeType() != ASTNodeType::Load);
        if (d.later()->nodeType() != ASTNodeType::ReduceTo) {
            needTapes[d.defId()].insert(d.earlier().as<StmtNode>()->id());
        }
    };
    auto found2 = [&](const Dependence &d) {
        if ((d.earlier()->nodeType() == ASTNodeType::Load &&
             d.later()->nodeType() != ASTNodeType::Load) ||
            (d.later()->nodeType() == ASTNodeType::Load &&
             d.earlier()->nodeType() != ASTNodeType::Load)) {
            ASSERT(d.dir_.size() == 1);
            affectingScopes[d.defId()].insert(d.dir_[0].first.id_);
        }
    };
    FindDeps()
        .type(DEP_RAW)
        .filterAccess([&](const auto &acc) {
            return intermediates.count(acc.def_->id());
        })
        .eraseOutsideVarDef(localVersionsOnly)(op, found1);
    FindDeps()
        .direction(direction)
        .type(DEP_RAW | DEP_WAR)
        .filterAccess(
            [&](const auto &acc) { return needTapes.count(acc.def_->id()); })
        .filter([&](const AccessPoint &later, const AccessPoint &earlier) {
            return needTapes.at(earlier.def_->id())
                       .count(earlier.stmt_->id()) ||
                   needTapes.at(earlier.def_->id()).count(later.stmt_->id());
        })
        .eraseOutsideVarDef(localVersionsOnly)(op, found2);

    std::unordered_map<StmtOrExprID, Expr> versions;
    std::unordered_map<ID, Expr> totLens;
    for (auto &&defId : intermediates) {
        auto &&scopes = affectingScopes[defId];
        auto &&needTape = needTapes[defId];
        CountScopeLen counter(defId, scopes, needTape);
        counter(op);
        auto &&scopeLen = counter.scopeLen();
        auto totLen = totLens[defId] =
            scopeLen.count(op) ? scopeLen.at(op) : makeIntConst(1);
        AnalyzeVersion analyzer(defId, scopes, needTape, scopeLen, totLen,
                                versions);
        analyzer(op);
    }

    std::unordered_set<ID> trivials = intermediates;
    FindDeps()
        .type(DEP_WAW)
        .filterAccess(
            [&](const auto &acc) { return needTapes.count(acc.def_->id()); })
        .filterEarlier([&](const auto &earlier) {
            return needTapes.at(earlier.def_->id())
                .count(earlier.op_.template as<StmtNode>()->id());
        })
        .eraseOutsideVarDef(localVersionsOnly)(
            op, [&](const Dependence &dep) { trivials.erase(dep.defId()); });

    return {versions, totLens, trivials};
}

} // namespace freetensor
