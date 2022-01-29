#include <analyze/analyze_version.h>
#include <analyze/deps.h>
#include <analyze/find_all_loops.h>

namespace ir {

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
    Expr len;
    for (auto &&stmt : op->stmts_) {
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
    Visitor::visit(op);
    if (op->var_ == var_) {
        versions_[op] = makeSub(offset_, makeIntConst(1));
    }
}

void AnalyzeVersion::visit(const Store &op) {
    Visitor::visit(op);
    if (op->var_ == var_ && needTapes_.count(op->id())) {
        versions_[op] = offset_;
    }
}

void AnalyzeVersion::visit(const ReduceTo &op) {
    Visitor::visit(op);
    if (op->var_ == var_ && needTapes_.count(op->id())) {
        versions_[op] = offset_;
    }
}

void AnalyzeVersion::visit(const VarDef &op) {
    if (op->id() == def_) {
        var_ = op->name_;
        Visitor::visit(op);
        var_.clear();
    } else {
        Visitor::visit(op);
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
        Visitor::visit(op);
        offset_ = oldOffset;
    } else {
        Visitor::visit(op);
    }
}

void AnalyzeVersion::visit(const StmtSeq &op) {
    auto oldOffset = offset_;
    for (auto &&stmt : op->stmts_) {
        if (scopeLen_.count(stmt)) {
            (*this)(stmt);
            offset_ = makeAdd(offset_, scopeLen_.at(stmt));
        } else {
            (*this)(stmt);
        }
    }
    offset_ = oldOffset;
}

std::pair<std::unordered_map<AST, Expr>, std::unordered_map<std::string, Expr>>
analyzeVersion(const Stmt &op,
               const std::unordered_set<std::string> &intermediates) {
    std::vector<FindDepsCond> conds;
    for (auto &&scope : findAllLoops(op)) {
        conds.push_back({{scope, DepDirection::Normal}});
    }
    std::unordered_map<std::string, std::unordered_set<std::string>>
        affectingScopes, needTapes;
    auto filter1 = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return intermediates.count(earlier.def_->id());
    };
    auto found1 = [&](const Dependency &d) {
        ASSERT(d.earlier()->nodeType() != ASTNodeType::Load);
        if (d.later()->nodeType() != ASTNodeType::ReduceTo) {
            needTapes[d.defId()].insert(d.earlier().as<StmtNode>()->id());
        }
    };
    auto filter2 = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return needTapes.count(earlier.def_->id()) &&
               (needTapes.at(earlier.def_->id()).count(earlier.cursor_.id()) ||
                needTapes.at(earlier.def_->id()).count(later.cursor_.id()));
    };
    auto found2 = [&](const Dependency &d) {
        if ((d.earlier()->nodeType() == ASTNodeType::Load &&
             d.later()->nodeType() != ASTNodeType::Load) ||
            (d.later()->nodeType() == ASTNodeType::Load &&
             d.earlier()->nodeType() != ASTNodeType::Load)) {
            ASSERT(d.cond_.size() == 1);
            affectingScopes[d.defId()].insert(d.cond_[0].first.name_);
        }
    };
    findDeps(op, {{}}, found1, FindDepsMode::Dep, DEP_RAW, filter1, true,
             false);
    findDeps(op, conds, found2, FindDepsMode::Dep, DEP_RAW | DEP_WAR, filter2,
             true, false);

    std::unordered_map<AST, Expr> versions;
    std::unordered_map<std::string, Expr> totLens;
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
    return std::make_pair(versions, totLens);
}

} // namespace ir
