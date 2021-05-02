#include <analyze/check_all_defined.h>
#include <pass/hoist_if.h>
#include <pass/simplify.h>

namespace ir {

Stmt HoistIf::visit(const If &op) {
    if (!op->elseCase_.isValid()) {
        for (auto &item : ifStack_) {
            item.emplace_back(op); // Use the old one
        }
    }
    return Mutator::visit(op);
}

Stmt HoistIf::visit(const For &op) {
    def_.insert(op->iter_);
    ifStack_.emplace_back();
    auto ret = Mutator::visit(op);
    auto ifList = std::move(ifStack_.back());
    ifStack_.pop_back();
    def_.erase(op->iter_);

    for (auto &&branch : ifList) {
        if (checkAllDefined(def_, branch->cond_)) {
            ret = makeIf("", branch->cond_, ret);
        }
    }
    return ret;
}

Stmt HoistIf::visit(const VarDef &op) {
    def_.insert(op->name_);
    auto ret = Mutator::visit(op);
    def_.erase(op->name_);
    return ret;
}

Stmt HoistIf::visit(const StmtSeq &op) {
    if (op->stmts_.size() > 1) {
        // Not pure nested
        auto ifStackBak = ifStack_;
        auto ret = Mutator::visit(op);
        ifStack_ = std::move(ifStackBak);
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

Stmt hoistIf(const Stmt &_op) {
    auto op = HoistIf()(_op);
    op = simplifyPass(op);
    return op;
}

} // namespace ir

