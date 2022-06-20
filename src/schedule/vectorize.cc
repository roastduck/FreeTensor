#include <analyze/deps.h>
#include <schedule/vectorize.h>

namespace freetensor {

Stmt Vectorize::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (op->id() == loop_) {
        op->property_->vectorize_ = true;
        done_ = true;
    }
    return op;
}

Stmt vectorize(const Stmt &_ast, const ID &loop) {
    Vectorize mutator(loop);
    auto ast = mutator(_ast);
    if (!mutator.done()) {
        throw InvalidSchedule("Loop " + toString(loop) + " not found");
    }
    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return earlier.stmt_->ancestorById(loop).isValid() &&
               later.stmt_->ancestorById(loop).isValid();
    };
    auto found = [&](const Dependency &d) {
        throw InvalidSchedule(toString(d) + " cannot be resolved");
    };
    FindDeps()
        .direction({{{loop, DepDirection::Normal}}})
        .filter(filter)(ast, found);
    return ast;
}

} // namespace freetensor
