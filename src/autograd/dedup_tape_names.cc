#include <autograd/dedup_tape_names.h>
#include <pass/rename_var.h>

namespace freetensor {

void CountNames::visit(const VarDef &op) {
    Visitor::visit(op);
    usedCnt_[op->name_]++;
}

void CountNames::visit(const For &op) {
    Visitor::visit(op);
    usedCnt_[op->iter_]++;
}

Stmt DedupTapeNames::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (tapes_.count(op->id()) && usedCnt_.at(op->name_) > 1) {
        std::string name;
        do {
            name = op->name_ + "." + std::to_string(++dedupNumber_);
        } while (usedCnt_.count(name));
        return renameVar(op, op->name_, name);
    }
    return op;
}

Stmt dedupTapeNames(const Stmt &op, const std::unordered_set<ID> &tapes) {
    CountNames counter;
    counter(op);
    return DedupTapeNames(tapes, counter.usedCnt())(op);
}

} // namespace freetensor
