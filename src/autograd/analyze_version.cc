#include <functional>

#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <autograd/analyze_version.h>
#include <pass/const_fold.h>

namespace freetensor {

namespace {

/**
 * The input data is also a version. As AccessType::Input and
 * AccessType::InputMutable are directly redirected at the ft.Array level, this
 * mutator only has an impact on AccessType::InOut.
 */
class InsertFakeStoreForInputs : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

  public:
    std::unordered_set<ID> ids_;

    const auto &ids() const { return ids_; }

  protected:
    using BaseClass::visit;

    Stmt visit(const VarDef &_op) override {
        auto __op = BaseClass::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        if (isInputting(op->buffer_->atype())) {
            auto fakeStore = makeStore(
                op->name_,
                std::vector<Expr>(
                    op->buffer_->tensor()->shape().size(),
                    makeIntrinsic("__any__", {}, DataType::Int32, false)),
                makeIntrinsic("__any__", {}, op->buffer_->tensor()->dtype(),
                              false));
            ids_.emplace(fakeStore->id());
            op->body_ =
                makeStmtSeq({std::move(fakeStore), std::move(op->body_)});
        }
        return op;
    }
};

} // Anonymous namespace

void CountScopeLen::visit(const Store &op) {
    Visitor::visit(op);
    if (op->var_ == var_ &&
        (fakeStoreIds_.count(op->id()) || needVersions_.count(op->id()))) {
        scopeLen_[op] = makeIntConst(1);
    }
}

void CountScopeLen::visit(const ReduceTo &op) {
    Visitor::visit(op);
    if (op->var_ == var_ && needVersions_.count(op->id())) {
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
            if (affectingScopes_.count(op->id())) {
                len = len.isValid() ? makeAdd(len, scopeLen_.at(stmt))
                                    : scopeLen_.at(stmt);
            } else {
                len = len.isValid() ? makeMax(len, scopeLen_.at(stmt))
                                    : scopeLen_.at(stmt);
            }
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
    if (op->var_ == var_ &&
        (fakeStoreIds_.count(op->id()) || needVersions_.count(op->id()))) {
        versions_[op->id()] = offset_;
    }
}

void AnalyzeVersion::visit(const ReduceTo &op) {
    BaseClass::visit(op);
    if (op->var_ == var_ && needVersions_.count(op->id())) {
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
    if (affectingScopes_.count(op->id())) {
        // Versioning for a `StmtSeq` node is more strict than that for a `For`
        // node. This means that not only do we check if the `StmtSeq` node is
        // affected, but we also distinguish between its sub-statements.
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
    } else {
        BaseClass::visit(op);
    }
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
    const Stmt &_op,
    const std::unordered_map<ID, std::unordered_set<ID>> &needVersions,
    const std::unordered_map<StmtOrExprID, Derivative::LazyFullDerivative>
        &derivatives,
    bool localVersionsOnly) {
    InsertFakeStoreForInputs insertFakeStore;
    auto op = insertFakeStore(_op);

    // Find out scopes we need to account in version numbers
    std::unordered_map<ID, std::unordered_set<ID>> affectingScopes;
    std::vector<FindDepsDir> direction;
    for (auto &&scope : findAllStmt(op, "<For>|<StmtSeq>")) {
        // NOTE: If checking each `StmtSeq` is too slow, we can check node
        // positions in the AST in the `found` callback
        direction.push_back({{scope->id(), DepDirection::Normal}});
    }
    FindDeps()
        .direction(direction)
        .type(DEP_RAW | DEP_WAR)
        .filterAccess([&](const auto &acc) -> bool {
            if (!needVersions.count(acc.def_->id())) {
                return false;
            }
            if (acc.op_->nodeType() == ASTNodeType::Load) {
                return true;
            } else {
                return insertFakeStore.ids().count(acc.stmt_->id()) ||
                       needVersions.at(acc.def_->id()).count(acc.stmt_->id());
            }
        })
        .eraseOutsideVarDef(localVersionsOnly)(op, [&](const Dependence &d) {
            ASSERT(d.dir_.size() == 1);
            affectingScopes[d.defId()].insert(d.dir_[0].first.id_);
        });

    std::unordered_map<StmtOrExprID, Expr> versions;
    std::unordered_map<std::string, std::pair<std::string, Expr>> userVersions;
    std::unordered_map<ID, Expr> totLens;
    for (auto &&[defId, needTape] : needVersions) {
        auto &&scopes = affectingScopes[defId];
        CountScopeLen counter(defId, scopes, needTape, insertFakeStore.ids());
        counter(op);
        auto &&scopeLen = counter.scopeLen();
        auto totLen = totLens[defId] =
            scopeLen.count(op) ? scopeLen.at(op) : makeIntConst(1);
        AnalyzeVersion analyzer(defId, scopes, needTape, insertFakeStore.ids(),
                                scopeLen, totLen, versions, userVersions);
        analyzer(op);
    }

    std::unordered_set<ID> trivials;
    for (auto &&[defId, _] : needVersions) {
        if (auto it = totLens.find(defId); it != totLens.end()) {
            if (auto len = constFold(it->second);
                len->nodeType() == ASTNodeType::IntConst &&
                len.template as<IntConstNode>()->val_ == 1) {
                trivials.emplace(defId);
            }
        }
    }
    FindDeps()
        .type(DEP_WAW)
        .ignoreReductionWAW(false)
        .filterAccess(
            [&](const auto &acc) { return trivials.count(acc.def_->id()); })
        .filterEarlier([&](const auto &earlier) {
            auto accId = earlier.op_.template as<StmtNode>()->id();
            return insertFakeStore.ids().count(accId) ||
                   needVersions.at(earlier.def_->id()).count(accId);
        })
        .filterLater([&](const auto &later) {
            auto accId = later.op_.template as<StmtNode>()->id();
            return !(insertFakeStore.ids().count(accId) ||
                     needVersions.at(later.def_->id()).count(accId));
        })
        .eraseOutsideVarDef(localVersionsOnly)(
            op, [&](const Dependence &dep) { trivials.erase(dep.defId()); });

    SetUserVersionsForInputs{userVersions}(op);

    return {versions, totLens, trivials, userVersions};
}

} // namespace freetensor
