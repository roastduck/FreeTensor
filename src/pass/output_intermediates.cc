#include <analyze/deps.h>
#include <analyze/find_all_scopes.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/output_intermediates.h>
#include <pass/undo_make_reduction.h>

namespace ir {

static MemType toGlobalMemType(MemType mtype) {
    switch (mtype) {
    case MemType::CPU:
        return MemType::CPU;
    case MemType::GPULocal:
    case MemType::GPUShared:
    case MemType::GPUGlobal:
        return MemType::GPUGlobal;
    default:
        ASSERT(false);
    }
}

void CountScopeLen::visit(const Store &op) {
    Visitor::visit(op);
    if (op->var_ == var_) {
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
            if (affectingScopes_.count(op->id())) {
                len = len.isValid() ? makeAdd(len, scopeLen_.at(stmt))
                                    : scopeLen_.at(stmt);
            } else {
                len = scopeLen_.at(stmt);
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

Expr AddExtraDim::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (loadMap_.count(_op)) {
        loadMap_[op] = loadMap_.at(_op);
        loadMap_.erase(_op);
    }
    if (op->var_ == var_) {
        std::vector<Expr> newIndices(1, offset_);
        newIndices.insert(newIndices.end(), op->indices_.begin(),
                          op->indices_.end());
        loadMap_[op] = makeLoad(op->var_ + ".tape", std::move(newIndices));
    }
    return op;
}

Stmt AddExtraDim::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    if (op->var_ == var_) {
        std::vector<Expr> newIndices(1, offset_);
        newIndices.insert(newIndices.end(), op->indices_.begin(),
                          op->indices_.end());
        auto newStore = makeStore("", op->var_ + ".tape", std::move(newIndices),
                                  makeLoad(op->var_, op->indices_));
        return makeStmtSeq("", {op, newStore});
    }
    return op;
}

Stmt AddExtraDim::visit(const VarDef &_op) {
    if (_op->id() == def_) {
        // FIXME: What if the scopeLen_ is a loop-variant temporary?
        if (totLen_->nodeType() == ASTNodeType::IntConst &&
            totLen_.as<IntConstNode>()->val_ == 1) {
            // No need to create a new VarDef
            auto __op = Mutator::visit(_op);
            ASSERT(__op->nodeType() == ASTNodeType::VarDef);
            auto op = __op.as<VarDefNode>();

            tapeName_ = op->name_;
            if (op->buffer_->atype() != AccessType::InOut) {
                op->buffer_->setAtype(AccessType::Output);
            }
            return op;
        } else {
            var_ = _op->name_;
            auto __op = Mutator::visit(_op);
            ASSERT(__op->nodeType() == ASTNodeType::VarDef);
            auto op = __op.as<VarDefNode>();
            var_.clear();

            tapeName_ = op->name_ + ".tape";
            auto tensor = op->buffer_->tensor();
            tensor.shape().insert(tensor.shape().begin(), totLen_);
            return makeVarDef("", tapeName_,
                              Buffer(std::move(tensor), AccessType::Output,
                                     toGlobalMemType(op->buffer_->mtype())),
                              nullptr, op, false);
        }
    } else {
        return Mutator::visit(_op);
    }
}

Stmt AddExtraDim::visit(const For &op) {
    if (affectingScopes_.count(op->id())) {
        auto oldOffset = offset_;
        offset_ = makeAdd(offset_,
                          makeMul(makeVar(op->iter_), scopeLen_.at(op->body_)));
        auto ret = Mutator::visit(op);
        offset_ = oldOffset;
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

Stmt AddExtraDim::visit(const StmtSeq &op) {
    if (affectingScopes_.count(op->id())) {
        auto oldOffset = offset_;
        std::vector<Stmt> stmts;
        for (auto &&stmt : op->stmts_) {
            stmts.emplace_back((*this)(stmt));
            if (scopeLen_.count(stmt)) {
                offset_ = makeAdd(offset_, scopeLen_.at(stmt));
            }
        }
        return makeStmtSeq(op->id(), std::move(stmts));
    } else {
        return Mutator::visit(op);
    }
}

std::tuple<Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<Load, Expr>>
outputIntermediates(const Stmt &_op,
                    const std::unordered_set<std::string> &intermediates) {
    auto op = flattenStmtSeq(_op); // Make the added dim simpler
    op = undoMakeReduction(op);    // Because we need to record loadMap

    std::vector<FindDepsCond> conds;
    for (auto &&scope : findAllScopes(op)) {
        conds.push_back({{scope, DepDirection::Normal}});
    }
    std::unordered_map<std::string, std::unordered_set<std::string>>
        affectingScopes;
    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return intermediates.count(earlier.def_->id());
    };
    auto found = [&](const Dependency &d) {
        ASSERT(d.cond_.size() == 1);
        affectingScopes[d.defId()].insert(d.cond_[0].first);
    };
    findDeps(op, conds, found, FindDepsMode::Dep, DEP_WAR, filter, true, false);

    std::unordered_map<std::string, std::string> nameMap;
    std::unordered_map<Load, Expr> loadMap;
    for (auto &&defId : intermediates) {
        auto &&scopes = affectingScopes[defId];
        CountScopeLen counter(defId, scopes);
        counter(op);
        auto &&scopeLen = counter.scopeLen();
        AddExtraDim adder(defId, scopes, scopeLen, scopeLen.at(op), loadMap);
        op = adder(op);
        nameMap[defId] = adder.tapeName();
    }
    return std::make_tuple(op, nameMap, loadMap);
}

} // namespace ir

