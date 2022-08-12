#include <analyze/all_defs.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/hoist_var_over_stmt_seq.h>

namespace freetensor {

Stmt HoistVarOverStmtSeq::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (rename_.count(op->name_)) {
        op->name_ = rename_.at(op->name_);
    }
    return op;
}

Expr HoistVarOverStmtSeq::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (rename_.count(op->var_)) {
        op->var_ = rename_.at(op->var_);
    }
    return op;
}

Stmt HoistVarOverStmtSeq::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    if (rename_.count(op->var_)) {
        op->var_ = rename_.at(op->var_);
    }
    return op;
}

Stmt HoistVarOverStmtSeq::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    if (rename_.count(op->var_)) {
        op->var_ = rename_.at(op->var_);
    }
    return op;
}

Stmt HoistVarOverStmtSeq::visit(const StmtSeq &op) {
    std::unordered_map<std::string, int> namesCnt, ioNamesCnt;
    for (auto &&[id, name] : allDefs(op)) {
        namesCnt[name]++;
    }

    std::vector<Stmt> stmts;
    std::vector<VarDef> defs;
    for (auto &&stmt : op->stmts_) {
        if (stmt->nodeType() == ASTNodeType::VarDef) {
            isFixPoint_ = false;
            auto def = stmt.as<VarDefNode>();
            Stmt _newDef;
            if (namesCnt.at(def->name_) > 1) {
                if (def->buffer_->atype() == AccessType::Cache) {
                    ASSERT(!rename_.count(def->name_));
                    rename_[def->name_] =
                        def->name_ + "." + toString(def->id());
                    _newDef = (*this)(stmt);
                    rename_.erase(def->name_);
                } else {
                    if (++ioNamesCnt[def->name_] > 1) {
                        throw InvalidProgram(
                            "Multiple I/O variables bound to the same name " +
                            def->name_);
                    }
                    _newDef = (*this)(stmt);
                }
            } else {
                _newDef = (*this)(stmt);
            }
            ASSERT(_newDef->nodeType() == ASTNodeType::VarDef);
            auto newDef = _newDef.as<VarDefNode>();
            defs.emplace_back(newDef);
            stmts.emplace_back(newDef->body_);
        } else {
            stmts.emplace_back((*this)(stmt));
        }
    }
    auto ret = makeStmtSeq(std::move(stmts));
    for (auto i = defs.rbegin(); i != defs.rend(); i++) {
        auto &&def = *i;
        ret =
            makeVarDef(def->name_, def->buffer_, def->ioTensor_, std::move(ret),
                       def->pinned_, def->metadata(), def->id());
    }
    return ret;
}

Stmt hoistVarOverStmtSeq(const Stmt &_op) {
    auto op = _op;
    for (int i = 0;; i++) {
        if (i > 100) {
            WARNING("HoistVarOverStmtSeq iterates over 100 rounds. Maybe there "
                    "is a bug");
            break;
        }
        HoistVarOverStmtSeq mutator;
        op = flattenStmtSeq(op);
        op = mutator(op);
        if (mutator.isFixPoint()) {
            break;
        }
    }
    return op;
}

} // namespace freetensor
