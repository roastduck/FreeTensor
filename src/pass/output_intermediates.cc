#include <analyze/deps.h>
#include <analyze/find_all_loops.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/make_reduction.h>
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

Expr AddExtraDim::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (loadMap_.count(_op)) {
        loadMap_[op] = loadMap_.at(_op);
        loadMap_.erase(_op);
    }
    if (op->var_ == var_) {
        std::vector<Expr> newIndices(1, makeSub(offset_, makeIntConst(1)));
        newIndices.insert(newIndices.end(), op->indices_.begin(),
                          op->indices_.end());
        loadMap_[op] = makeLoad(op->var_ + ".tape", std::move(newIndices));
    }
    return op;
}

Stmt AddExtraDim::visit(const Store &op) {
    if (op->var_ == var_ && needTapes_.count(op->id())) {
        std::vector<Expr> indices;
        indices.reserve(op->indices_.size());
        for (auto &&index : op->indices_) {
            indices.emplace_back((*this)(index));
        }
        auto oldStore = makeStore(op->id(), op->var_, std::move(indices),
                                  (*this)(op->expr_));

        std::vector<Expr> newIndices(1, offset_);
        newIndices.insert(newIndices.end(), op->indices_.begin(),
                          op->indices_.end());
        auto newStore = makeStore("", op->var_ + ".tape", std::move(newIndices),
                                  makeLoad(op->var_, op->indices_));

        return makeStmtSeq("", {oldStore, newStore});
    } else {
        return Mutator::visit(op);
    }
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
    auto oldOffset = offset_;
    std::vector<Stmt> stmts;
    for (auto &&stmt : op->stmts_) {
        if (scopeLen_.count(stmt)) {
            stmts.emplace_back((*this)(stmt));
            offset_ = makeAdd(offset_, scopeLen_.at(stmt));
        } else {
            stmts.emplace_back((*this)(stmt));
        }
    }
    offset_ = oldOffset;
    return makeStmtSeq(op->id(), std::move(stmts));
}

std::tuple<Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<Load, Expr>>
outputIntermediates(const Stmt &_op,
                    const std::unordered_set<std::string> &intermediates) {
    auto op = flattenStmtSeq(_op); // Make the added dim simpler

    // Reduce min and reduce max may need the intermediate value for
    // gradients, but reduce add does not
    op = makeReduction(op, {ReduceOp::Add});

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

    op = undoMakeReduction(op); // Because we need to record loadMap

    std::unordered_map<std::string, std::string> nameMap;
    std::unordered_map<Load, Expr> loadMap;
    for (auto &&defId : intermediates) {
        auto &&scopes = affectingScopes[defId];
        auto &&needTape = needTapes[defId];
        CountScopeLen counter(defId, scopes, needTape);
        counter(op);
        auto &&scopeLen = counter.scopeLen();
        AddExtraDim adder(
            defId, scopes, needTape, scopeLen,
            scopeLen.count(op) ? scopeLen.at(op) : makeIntConst(1), loadMap);
        op = adder(op);
        nameMap[defId] = adder.tapeName();
    }
    return std::make_tuple(op, nameMap, loadMap);
}

} // namespace ir

