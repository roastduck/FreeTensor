#include <algorithm>
#include <unordered_map>

#include <analyze/deps.h>
#include <schedule/swap.h>

namespace ir {

Stmt Swap::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();

    std::unordered_map<ID, size_t> pos;
    for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
        pos[op->stmts_[i]->id()] = i;
    }
    size_t start = op->stmts_.size();
    for (auto &&item : order_) {
        if (!pos.count(item)) {
            return op;
        }
        start = std::min(start, pos.at(item));
    }

    std::vector<Stmt> stmts = op->stmts_;
    for (size_t i = 0, iEnd = order_.size(); i < iEnd; i++) {
        auto p = pos.at(order_[i]);
        if (p >= start + order_.size()) {
            return op;
        }
        stmts[start + i] = op->stmts_[p];
    }
    scope_ = _op; // the old one
    return makeStmtSeq(op->id(), std::move(stmts));
}

Stmt swap(const Stmt &_ast, const std::vector<ID> &order) {
    Swap mutator(order);
    auto ast = mutator(_ast);
    auto scope = mutator.scope();
    if (!scope.isValid()) {
        throw InvalidSchedule("Statements not found or not consecutive");
    }

    auto findParentStmt = [&](const Stmt &stmt) -> Stmt {
        for (auto &&item : order) {
            auto ret = stmt->ancestorById(item);
            if (ret.isValid()) {
                return ret;
            }
        }
        return nullptr;
    };
    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        auto s0 = findParentStmt(later.stmt_);
        auto s1 = findParentStmt(earlier.stmt_);
        if (!s0.isValid() || !s1.isValid()) {
            return false;
        }
        auto old0 =
            std::find_if(scope->stmts_.begin(), scope->stmts_.end(),
                         [&](const Stmt &s) { return s->id() == s0->id(); });
        auto old1 =
            std::find_if(scope->stmts_.begin(), scope->stmts_.end(),
                         [&](const Stmt &s) { return s->id() == s1->id(); });
        auto new0 = std::find_if(order.begin(), order.end(),
                                 [&](const ID &id) { return id == s0->id(); });
        auto new1 = std::find_if(order.begin(), order.end(),
                                 [&](const ID &id) { return id == s1->id(); });
        return (old0 < old1) != (new0 < new1);
    };
    auto found = [&](const Dependency &d) {
        throw InvalidSchedule(toString(d) + " cannot be resolved");
    };
    findDeps(ast, {{}}, found, FindDepsMode::Dep, DEP_ALL, filter);
    return ast;
}

} // namespace ir
