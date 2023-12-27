#include <algorithm>
#include <unordered_map>

#include <analyze/deps.h>
#include <container_utils.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/sink_var.h>
#include <schedule.h>
#include <schedule/check_not_in_lib.h>
#include <schedule/hoist_selected_var.h>
#include <schedule/swap.h>
#include <selector.h>

namespace freetensor {

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
    return makeStmtSeq(std::move(stmts), op->metadata(), op->id());
}

Stmt swap(const Stmt &_ast, const std::vector<ID> &order) {
    for (auto &&item : order) {
        checkNotInLib(_ast, item);
    }

    // Hoist all VarDef nodes covering any of the statement but not covering
    // some other statements, to cover all statements
    auto insides = order | views::transform([](const ID &id) {
                       return "->>" + toString(id);
                   });
    auto allIn = insides | join("&");
    auto anyIn = insides | join("|");
    auto oldAst = hoistSelectedVar(_ast, "(" + anyIn + ")&!(" + allIn + ")");
    oldAst = flattenStmtSeq(oldAst);

    Swap mutator(order);
    auto ast = mutator(
        oldAst); // Don't destory `oldAst`, or `.scope()` will become an orphan
    auto scope = mutator.scope();
    if (!scope.isValid()) {
        throw InvalidSchedule("Statements not found or not consecutive");
    }

    FindDepsDir dir;
    for (auto outer = scope->parentStmt(); outer.isValid();
         outer = outer->parentStmt()) {
        if (outer->nodeType() == ASTNodeType::For) {
            dir.emplace_back(outer->id(), DepDirection::Same);
        }
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
    FindDeps().direction({dir}).filter([&](const AccessPoint &later,
                                           const AccessPoint &earlier) {
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
    })(ast, [&](const Dependence &d) {
        throw InvalidSchedule(toString(d) + " cannot be resolved");
    });

    return sinkVar(ast);
}

void Schedule::swap(const std::vector<ID> &order) {
    beginTransaction();
    auto log = appendLog(MAKE_SCHEDULE_LOG(Swap, freetensor::swap, order));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
