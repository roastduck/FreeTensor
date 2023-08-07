#include <analyze/deps.h>
#include <schedule.h>
#include <schedule/parallelize.h>

namespace freetensor {

Stmt Parallelize::visit(const For &_op) {
    auto thisParallel =
        _op->id() == loop_ ? parallel_ : _op->property_->parallel_;
    Stmt __op;
    loopStack_.emplace_back(_op->id());
    if (std::holds_alternative<CUDAScope>(thisParallel)) {
        if (para2var_.count(thisParallel)) {
            auto oldVar = para2var_.at(thisParallel);
            para2var_[thisParallel] = _op->iter_;
            hiddenVars_.insert(oldVar);
            __op = Mutator::visit(_op);
            hiddenVars_.erase(oldVar);
            para2var_[thisParallel] = oldVar;
        } else {
            para2var_[thisParallel] = _op->iter_;
            __op = Mutator::visit(_op);
            para2var_.erase(thisParallel);
        }
    } else {
        __op = Mutator::visit(_op);
    }
    loopStack_.pop_back();

    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();

    if (op->id() == loop_) {
        op->property_->parallel_ = parallel_;
        outerLoops_ = loopStack_;
        done_ = true;
    }
    return op;
}

Expr Parallelize::visit(const Var &op) {
    if (hiddenVars_.count(op->name_)) {
        throw InvalidSchedule("Unable to bind multiple loops, which are used "
                              "simultaneously, all to " +
                              toString(parallel_));
    }
    return Mutator::visit(op);
}

Stmt parallelize(const Stmt &_ast, const ID &loop,
                 const ParallelScope &parallel, bool allowReduction) {
    if (getenv("PAPER_IS_BWD") && getenv("PAPER_NO_PAR_REDUCE")) {
        allowReduction = false;
    }

    Parallelize mutator(loop, parallel);
    auto ast = _ast;
    auto oldAst = ast;
    ast = mutator(ast);
    if (!mutator.done()) {
        throw InvalidSchedule("Loop " + toString(loop) + " not found");
    }

    {
        FindDepsDir findDepsDir{{loop, DepDirection::Normal}};
        for (auto &&outerLoop : mutator.outerLoops()) {
            findDepsDir.push_back({outerLoop, DepDirection::Same});
        }
        auto found = [&](const Dependence &d) {
            throw InvalidSchedule(toString(d) + " cannot be resolved");
        };
        FindDeps()
            .direction({findDepsDir})
            .filterSubAST(loop)
            .ignoreReductionWAW(allowReduction)(oldAst, found);
    }

    {
        auto filter = [&](const AccessPoint &later,
                          const AccessPoint &earlier) {
            bool earlierInLoop = earlier.stmt_->ancestorById(loop).isValid();
            bool laterInLoop = later.stmt_->ancestorById(loop).isValid();
            if ((earlierInLoop && !laterInLoop) ||
                (!earlierInLoop && laterInLoop)) {
                if (std::holds_alternative<CUDAScope>(parallel) &&
                    std::get<CUDAScope>(parallel).level_ == CUDAScope::Thread) {
                    return later.def_->buffer_->mtype() == MemType::GPULocal;
                }
                if (std::holds_alternative<CUDAScope>(parallel) &&
                    std::get<CUDAScope>(parallel).level_ == CUDAScope::Block) {
                    return later.def_->buffer_->mtype() == MemType::GPULocal ||
                           later.def_->buffer_->mtype() == MemType::GPUShared;
                }
            }
            return false;
        };
        auto found = [&](const Dependence &d) {
            ASSERT(d.dir_.size() == 1);
            throw InvalidSchedule(toString(d) + " cannot be resolved");
        };
        FindDeps()
            .direction(
                {{{NodeIDOrParallelScope(parallel), DepDirection::Different}}})
            .filter(filter)
            .ignoreReductionWAW(allowReduction)(ast, found);
    }
    return ast;
}

void Schedule::parallelize(const ID &loop, const ParallelScope &parallel,
                           bool allowReduction) {
    beginTransaction();
    auto log = appendLog(MAKE_SCHEDULE_LOG(Parallelize, freetensor::parallelize,
                                           loop, parallel, allowReduction));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
