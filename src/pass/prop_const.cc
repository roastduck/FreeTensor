#include <analyze/deps.h>
#include <pass/prop_const.h>
#include <pass/simplify.h>

namespace ir {

Expr ReplaceLoads::visit(const Load &op) {
    if (replacement_.count(op)) {
        return (*this)(replacement_.at(op));
    } else {
        return Mutator::visit(op);
    }
}

Stmt propConst(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
        op = simplifyPass(op);

        std::unordered_map<Load, int> mayDepCnt;
        std::unordered_map<Load, std::vector<Stmt>> r2w;
        auto foundMay = [&](const Dependency &d) {
            ASSERT(d.later()->nodeType() == ASTNodeType::Load);
            mayDepCnt[d.later().as<LoadNode>()]++;
        };
        auto filterMust = [&](const AccessPoint &later,
                              const AccessPoint &earlier) {
            if (earlier.op_->nodeType() != ASTNodeType::Store) {
                return false;
            }
            ASSERT(later.op_->nodeType() == ASTNodeType::Load);
            if (!mayDepCnt.count(later.op_.as<LoadNode>()) ||
                mayDepCnt.at(later.op_.as<LoadNode>()) > 1) {
                return false;
            }
            auto &&expr = earlier.op_.as<StoreNode>()->expr_;
            return expr->nodeType() == ASTNodeType::IntConst ||
                   expr->nodeType() == ASTNodeType::FloatConst ||
                   expr->nodeType() == ASTNodeType::BoolConst;
        };
        auto foundMust = [&](const Dependency &d) {
            ASSERT(d.later()->nodeType() == ASTNodeType::Load);
            r2w[d.later().as<LoadNode>()].emplace_back(
                d.earlier().as<StmtNode>());
        };
        findDeps(op, {{}}, foundMay, FindDepsMode::Dep, DEP_RAW);
        findDeps(op, {{}}, foundMust, FindDepsMode::KillLater, DEP_RAW,
                 filterMust);

        std::unordered_map<Load, Expr> replacement;
        for (auto &&item : r2w) {
            ASSERT(item.second.size() == 1);
            ASSERT(item.second.front()->nodeType() == ASTNodeType::Store);
            const Load &load = item.first;
            const Store &store = item.second.front().as<StoreNode>();
            replacement[load] = store->expr_;
        }

        if (replacement.empty() || i > 100) {
            if (i > 100) {
                WARNING(
                    "propConst iterates over 100 rounds. Maybe there is a bug");
            }
            break;
        }
        op = ReplaceLoads(replacement)(op);
    }

    return op;
}

} // namespace ir

