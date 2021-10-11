#include <analyze/deps.h>
#include <schedule/vectorize.h>

namespace ir {

Stmt Vectorize::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (op->id() == loop_) {
        op->vectorize_ = true;
        done_ = true;
    }
    return op;
}

Stmt vectorize(const Stmt &_ast, const std::string &loop) {
    Vectorize mutator(loop);
    auto ast = mutator(_ast);
    if (!mutator.done()) {
        throw InvalidSchedule("Loop " + loop + " not found");
    }
    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return earlier.cursor_.getParentById(loop).isValid() &&
               later.cursor_.getParentById(loop).isValid();
    };
    auto found = [&](const Dependency &d) {
        throw InvalidSchedule(dep2Str(loop, d.var_, d.later(), d.earlier()));
    };
    findDeps(ast, {{{loop, DepDirection::Normal}}}, found, FindDepsMode::Dep,
             DEP_ALL, filter);
    return ast;
}

} // namespace ir
