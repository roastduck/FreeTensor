#include <analyze/deps.h>
#include <schedule.h>
#include <schedule/check_not_in_lib.h>
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
    checkNotInLib(_ast, loop);
    Vectorize mutator(loop);
    auto ast = mutator(_ast);
    if (!mutator.done()) {
        throw InvalidSchedule("Loop " + toString(loop) + " not found");
    }
    auto found = [&](const Dependence &d) {
        throw InvalidSchedule(toString(d) + " cannot be resolved");
    };
    FindDeps()
        .direction({{{loop, DepDirection::Normal}}})
        .ignoreReductionWAW(false)
        .filterSubAST(loop)(ast, found);
    return ast;
}

void Schedule::vectorize(const ID &loop) {
    beginTransaction();
    auto log =
        appendLog(MAKE_SCHEDULE_LOG(Vectorize, freetensor::vectorize, loop));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
