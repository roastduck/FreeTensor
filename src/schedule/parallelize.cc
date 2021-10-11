#include <analyze/deps.h>
#include <pass/make_reduction.h>
#include <schedule/parallelize.h>

namespace ir {

Stmt Parallelize::visit(const For &_op) {
    loopStack_.emplace_back(_op->id());
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    loopStack_.pop_back();

    if (op->id() == loop_) {
        op->parallel_ = parallel_;
        outerLoops_ = loopStack_;
        done_ = true;
    }
    return op;
}

Stmt parallelize(const Stmt &_ast, const std::string &loop,
                 const std::string &parallel) {
    Parallelize mutator(loop, parallel);
    auto ast = makeReduction(_ast);
    auto oldAst = ast;
    ast = mutator(ast);
    if (!mutator.done()) {
        throw InvalidSchedule("Loop " + loop + " not found");
    }
    FindDepsCond findDepsCond{{loop, DepDirection::Normal}};
    for (auto &&outerLoop : mutator.outerLoops()) {
        findDepsCond.push_back({outerLoop, DepDirection::Same});
    }
    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return earlier.cursor_.getParentById(loop).isValid() &&
               later.cursor_.getParentById(loop).isValid();
    };
    auto found = [&](const Dependency &d) {
        throw InvalidSchedule(dep2Str(loop, d.var_, d.later(), d.earlier()));
    };
    findDeps(oldAst, {findDepsCond}, found, FindDepsMode::Dep, DEP_ALL, filter);
    return ast;
}

} // namespace ir
