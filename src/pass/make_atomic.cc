#include <analyze/deps.h>
#include <pass/disambiguous.h>
#include <pass/make_atomic.h>
#include <pass/make_reduction.h>

namespace ir {

void FindAllParallel::visit(const For &op) {
    Visitor::visit(op);
    if (!op->parallel_.empty()) {
        results_.emplace_back(op->id());
    }
}

Stmt MakeAtomic::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    if (toAlter_.count(op->id())) {
        op->atomic_ = true;
    }
    return op;
}

Stmt makeAtomic(const Stmt &_op) {
    auto op = makeReduction(_op);
    op = disambiguous(op); // findDeps need it

    std::vector<std::vector<std::pair<std::string, FindDepsMode>>> cond;
    FindAllParallel finder;
    finder(op);
    for (auto &&loop : finder.results()) {
        cond.push_back({{loop, FindDepsMode::Normal}});
    }

    std::unordered_set<std::string> toAlter;
    auto found = [&](const std::vector<std::pair<std::string, FindDepsMode>> &,
                     const std::string &, const AST &later, const AST &earlier,
                     const Cursor &, const Cursor &) {
        if (later->nodeType() == ASTNodeType::ReduceTo &&
            earlier->nodeType() == ASTNodeType::ReduceTo) {
            toAlter.insert(later.as<ReduceToNode>()->id());
            toAlter.insert(earlier.as<ReduceToNode>()->id());
        }
    };
    findDeps(op, cond, found, false);

    return MakeAtomic(toAlter)(op);
}

} // namespace ir

