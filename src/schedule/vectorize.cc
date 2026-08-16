#include <analyze/deps.h>
#include <schedule.h>
#include <schedule/check_not_in_lib.h>
#include <schedule/subprocess_utils.h>
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
        throw InvalidSchedule(FT_MSG << "Loop " << loop << " not found");
    }
    auto found = [&](const Dependence &d) {
        throw InvalidSchedule(FT_MSG << d << " cannot be resolved");
    };
    FindDeps()
        .direction({{{loop, DepDirection::Normal}}})
        .ignoreReductionWAW(false)
        .filterSubAST(loop)(ast, found);
    return ast;
}

void Schedule::vectorize(const ID &loop,
                         const std::optional<bool> &asSubprocess,
                         const std::optional<double> &timeout) {
    if (shouldRunInSubprocess(asSubprocess, timeout)) {
        auto result = runTransformSubprocess(
            "vectorize", ast(), subprocessArgs({"--loop", idArg(loop)}),
            timeout);
        applySubprocessResult(result);
        return;
    }
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
