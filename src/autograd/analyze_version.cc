#include <functional>

#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <analyze/symbol_table.h>
#include <autograd/analyze_version.h>
#include <pass/flatten_stmt_seq.h>

namespace freetensor {

namespace {

class ReplaceMarkVersionByFakeLoad : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

  protected:
    using BaseClass::visit;

    Stmt visit(const MarkVersion &op) override {
        auto &&tensor = buffer(op->var_)->tensor();
        return makeEval(
            makeLoad(op->var_,
                     std::vector<Expr>(
                         tensor->shape().size(),
                         makeIntrinsic("__any__", {}, DataType::Int32, false)),
                     tensor->dtype()),
            op->metadata(), op->id());
    }
};

class FindWrites : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    std::function<void(const Expr & /* expr */, const ID & /* id */,
                       const ID & /* defId */)>
        callback_;

  public:
    FindWrites(const auto &callback) : callback_(callback) {}

  protected:
    using BaseClass::visit;

    void visit(const Store &op) override {
        BaseClass::visit(op);
        callback_(op->expr_, op->id(), def(op->var_)->id());
    }

    void visit(const ReduceTo &op) override {
        BaseClass::visit(op);
        callback_(op->expr_, op->id(), def(op->var_)->id());
    }
};

} // Anonymous namespace

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

void AnalyzeVersion::visit(const MarkVersion &op) {
    BaseClass::visit(op);
    if (op->var_ == var_) {
        auto v = makeSub(offset_, makeIntConst(1));
        versions_[op->id()] = v;
        userVersions_[op->tapeName_] = {op->var_, v};
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

void SetUserVersionsForInputs::visit(const MarkVersion &op) {
    BaseClass::visit(op);
    if (buffer(op->var_)->atype() == AccessType::Input) {
        userVersions_[op->tapeName_] = {op->var_, nullptr};
    }
}

std::tuple<std::unordered_map<StmtOrExprID, Expr>, std::unordered_map<ID, Expr>,
           std::unordered_set<ID>,
           std::unordered_map<std::string, std::pair<std::string, Expr>>>
analyzeVersion(
    const Stmt &op, const std::unordered_set<ID> &intermediates,
    const std::unordered_map<StmtOrExprID, Derivative::LazyFullDerivative>
        &derivatives,
    bool localVersionsOnly) {
    // Find out the statements which we need to store its value as one version
    //
    // For each variable in `intermediates`, which means we are willing to save,
    // we check if we need to save it for a particular statement. There are two
    // cases:
    //
    // - (RAW) If the variable is written by statement X, and then read by
    // statement Y, a version of it on X is needed, which can be used either to
    // recompute Y's forward value, or to compute Y's gradient. Currently there
    // may be false positive in this case, because we don't know which
    // statements have to be recomputed.
    // - (RAW) If the variable is written by statement X, where the value may
    // propagate to a MarkVersion node, a version of it on X is needed. In order
    // to find such statements, we replace MarkVersion by a fake Load node with
    // relaxed indices before we do a dependence analysis.
    // - (W) If the variable is written by statement X, and then overwritten
    // by statement Y, and if we need X's result to compute X's gradient (use
    // `y` for `y = f(x)`'s gradient), then we need a version of it on X.
    std::unordered_map<ID, std::unordered_set<ID>> needTapes;
    FindDeps()
        .type(DEP_RAW)
        .filterAccess([&](const auto &acc) {
            return intermediates.count(acc.def_->id());
        })
        .eraseOutsideVarDef(localVersionsOnly)(op, [&](const Dependence &d) {
            ASSERT(d.earlier()->nodeType() != ASTNodeType::Load);
            if (d.later()->nodeType() != ASTNodeType::ReduceTo) {
                needTapes[d.defId()].insert(d.earlier().as<StmtNode>()->id());
            }
        });
    FindWrites{[&](const Expr &expr, const ID &id, const ID &defId) {
        if (derivatives.count(StmtOrExprID{expr, id})) {
            needTapes[defId].insert(id);
        }
    }}(ReplaceMarkVersionByFakeLoad{}(op));

    // Find out scopes we need to account in version numbers
    std::unordered_map<ID, std::unordered_set<ID>> affectingScopes;
    std::vector<FindDepsDir> direction;
    for (auto &&scope : findAllStmt(op, "<For>")) {
        direction.push_back({{scope->id(), DepDirection::Normal}});
    }
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
        .eraseOutsideVarDef(localVersionsOnly)(op, [&](const Dependence &d) {
            if ((d.earlier()->nodeType() == ASTNodeType::Load &&
                 d.later()->nodeType() != ASTNodeType::Load) ||
                (d.later()->nodeType() == ASTNodeType::Load &&
                 d.earlier()->nodeType() != ASTNodeType::Load)) {
                ASSERT(d.dir_.size() == 1);
                affectingScopes[d.defId()].insert(d.dir_[0].first.id_);
            }
        });

    std::unordered_map<StmtOrExprID, Expr> versions;
    std::unordered_map<std::string, std::pair<std::string, Expr>> userVersions;
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
                                versions, userVersions);
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

    SetUserVersionsForInputs{userVersions}(op);

    return {versions, totLens, trivials, userVersions};
}

} // namespace freetensor
