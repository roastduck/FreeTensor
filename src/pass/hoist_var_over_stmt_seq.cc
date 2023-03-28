#include <algorithm>

#include <analyze/all_defs.h>
#include <analyze/all_uses.h>
#include <analyze/find_stmt.h>
#include <container_utils.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/hoist_var_over_stmt_seq.h>

namespace freetensor {

Stmt HoistVarOverStmtSeq::visit(const StmtSeq &op) {
    auto parentAllWrites = allWrites(op);

    std::unordered_map<std::string, int> namesCnt, ioNamesCnt;
    for (auto &&[id, name] : allDefs(op)) {
        namesCnt[name]++;
    }

    std::vector<Stmt> stmts;
    std::vector<VarDef> defs;
    for (auto &&stmt : op->stmts_) {
        if (stmt->nodeType() == ASTNodeType::VarDef) {
            auto def = stmt.as<VarDefNode>();

            std::unordered_set<std::string> shapeAllReads;
            for (auto &&dim : def->buffer_->tensor()->shape()) {
                shapeAllReads = uni(shapeAllReads, allReads(dim));
            }
            if (hasIntersect(parentAllWrites, shapeAllReads)) {
                goto no_hoist;
            }
            if (togetherIds_.has_value()) {
                auto togetherInside = findAllStmt(def, [&](const Stmt &s) {
                    return std::find(togetherIds_->begin(), togetherIds_->end(),
                                     s->id()) != togetherIds_->end();
                });
                if (togetherInside.empty() ||
                    togetherInside.size() == togetherIds_->size()) {
                    goto no_hoist;
                }
            }

            isFixPoint_ = false;
            Stmt _newDef;
            if (namesCnt.at(def->name_) > 1) {
                if (!isInputting(def->buffer_->atype()) &&
                    !isOutputting(def->buffer_->atype())) {
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
            continue;
        }

    no_hoist:
        stmts.emplace_back((*this)(stmt));
    }
    auto ret = makeStmtSeq(std::move(stmts));
    for (auto i = defs.rbegin(); i != defs.rend(); i++) {
        auto &&def = *i;
        ret = makeVarDef(def->name_, def->buffer_, def->viewOf_, std::move(ret),
                         def->pinned_, def->metadata(), def->id());
    }
    return ret;
}

Stmt hoistVarOverStmtSeq(const Stmt &_op,
                         const std::optional<std::vector<ID>> &togetherIds) {
    auto op = _op;
    for (int i = 0;; i++) {
        if (i > 100) {
            WARNING("HoistVarOverStmtSeq iterates over 100 rounds. Maybe there "
                    "is a bug");
            break;
        }
        HoistVarOverStmtSeq mutator(togetherIds);
        op = flattenStmtSeq(op);
        op = mutator(op);
        if (mutator.isFixPoint()) {
            break;
        }
    }
    return op;
}

} // namespace freetensor
