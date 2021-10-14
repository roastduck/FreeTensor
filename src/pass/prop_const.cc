#include <analyze/deps.h>
#include <pass/prop_const.h>
#include <pass/replace_uses.h>
#include <pass/simplify.h>

namespace ir {

Stmt propConst(const Stmt &_op) {
    auto op = _op;

    for (int i = 0;; i++) {
        op = simplifyPass(op);

        std::unordered_map<AST, std::vector<Stmt>> r2w, r2wMay;
        auto foundMay = [&](const Dependency &d) {
            r2wMay[d.later()].emplace_back(d.earlier().as<StmtNode>());
        };
        auto filterMust = [&](const AccessPoint &later,
                              const AccessPoint &earlier) {
            if (earlier.op_->nodeType() != ASTNodeType::Store) {
                return false;
            }
            if (!r2wMay.count(later.op_) || r2wMay.at(later.op_).size() > 1 ||
                r2wMay.at(later.op_)[0] != earlier.op_.as<StmtNode>()) {
                return false;
            }
            auto &&expr = earlier.op_.as<StoreNode>()->expr_;
            return expr->nodeType() == ASTNodeType::IntConst ||
                   expr->nodeType() == ASTNodeType::FloatConst ||
                   expr->nodeType() == ASTNodeType::BoolConst;
        };
        auto foundMust = [&](const Dependency &d) {
            r2w[d.later()].emplace_back(d.earlier().as<StmtNode>());
        };
        findDeps(op, {{}}, foundMay, FindDepsMode::Dep, DEP_RAW, nullptr,
                 false);
        findDeps(op, {{}}, foundMust, FindDepsMode::KillLater, DEP_RAW,
                 filterMust);

        std::unordered_map<Load, Expr> replaceLoad;
        std::unordered_map<ReduceTo, Expr> replaceReduceTo;
        for (auto &&item : r2w) {
            ASSERT(item.second.size() == 1);
            ASSERT(item.second.front()->nodeType() == ASTNodeType::Store);
            auto &&store = item.second.front().as<StoreNode>();
            if (item.first->nodeType() == ASTNodeType::Load) {
                auto &&load = item.first.as<LoadNode>();
                replaceLoad[load] = store->expr_;
            } else {
                ASSERT(item.first->nodeType() == ASTNodeType::ReduceTo);
                auto &&reduceTo = item.first.as<ReduceToNode>();
                replaceReduceTo[reduceTo] = store->expr_;
            }
        }

        if ((replaceLoad.empty() && replaceReduceTo.empty()) || i > 100) {
            if (i > 100) {
                WARNING(
                    "propConst iterates over 100 rounds. Maybe there is a bug");
            }
            break;
        }
        op = ReplaceUses(replaceLoad, replaceReduceTo)(op);
    }

    return op;
}

} // namespace ir

