#include <analyze/deps.h>
#include <pass/prop_const.h>

namespace ir {

Expr ReplaceLoads::visit(const Load &op) {
    if (replacement_.count(op)) {
        return (*this)(replacement_.at(op));
    } else {
        return Mutator::visit(op);
    }
}

Stmt propConst(const Stmt &op) {
    std::unordered_map<Load, int> mayDepCnt;
    std::unordered_map<Load, std::vector<Stmt>> r2w;
    auto foundMay = [&](const Dependency &d) {
        ASSERT(d.later()->nodeType() == ASTNodeType::Load);
        mayDepCnt[d.later().as<LoadNode>()]++;
    };
    auto foundMust = [&](const Dependency &d) {
        ASSERT(d.later()->nodeType() == ASTNodeType::Load);
        r2w[d.later().as<LoadNode>()].emplace_back(d.earlier().as<StmtNode>());
    };
    findDeps(op, {{}}, foundMay, FindDepsMode::Dep, DEP_RAW);
    findDeps(op, {{}}, foundMust, FindDepsMode::KillLater, DEP_RAW);

    std::unordered_map<Load, Expr> replacement;
    for (auto &&item : r2w) {
        if (mayDepCnt.at(item.first) > 1) {
            continue;
        }
        ASSERT(item.second.size() == 1);
        if (item.second.front()->nodeType() != ASTNodeType::Store) {
            continue;
        }

        const Load &load = item.first;
        const Store &store = item.second.front().as<StoreNode>();
        switch (store->expr_->nodeType()) {
        case ASTNodeType::IntConst:
        case ASTNodeType::FloatConst:
        case ASTNodeType::BoolConst:
            replacement[load] = store->expr_;
            break;
        default:
            continue;
        }
    }

    return ReplaceLoads(replacement)(op);
}

} // namespace ir

