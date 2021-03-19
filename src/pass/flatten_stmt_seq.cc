#include <pass/flatten_stmt_seq.h>

namespace ir {

Stmt FlattenStmtSeq::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();

    std::vector<Stmt> stmts;
    stmts.reserve(op->stmts_.size());
    // Also move VarDef outer. This is mainly for Schedule::moveTo
    std::vector<VarDef> defStack;
    for (Stmt item : op->stmts_) {
        if (popVarDef_) {
            while (item->nodeType() == ASTNodeType::VarDef) {
                defStack.emplace_back(item.as<VarDefNode>());
                item = item.as<VarDefNode>()->body_;
            }
        }
        if (item->nodeType() == ASTNodeType::StmtSeq) {
            for (auto &&subitem : item.as<StmtSeqNode>()->stmts_) {
                stmts.emplace_back(subitem);
            }
        } else {
            stmts.emplace_back(item);
        }
    }

    auto ret =
        stmts.size() == 1 ? stmts[0] : makeStmtSeq(op->id(), std::move(stmts));
    for (auto it = defStack.rbegin(); it != defStack.rend(); it++) {
        auto &&def = *it;
        ret = makeVarDef(def->id(), def->name_, *def->buffer_, def->sizeLim_,
                         ret);
    }
    return ret;
}

} // namespace ir

