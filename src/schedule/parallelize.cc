#include <analyze/deps.h>
#include <pass/make_reduction.h>
#include <schedule/parallelize.h>

namespace ir {

Stmt Parallelize::visit(const For &_op) {
    auto thisParallel =
        _op->id() == loop_ ? parallel_ : _op->property_.parallel_;
    Stmt __op;
    loopStack_.emplace_back(_op->id());
    if (thisParallel.substr(0, 10) == "threadIdx." ||
        thisParallel.substr(0, 9) == "blockIdx.") {
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
        op->property_.parallel_ = parallel_;
        outerLoops_ = loopStack_;
        done_ = true;
    }
    return op;
}

Expr Parallelize::visit(const Var &op) {
    if (hiddenVars_.count(op->name_)) {
        throw InvalidSchedule("Unable to bind multiple loops, which are used "
                              "simultaneously, all to " +
                              parallel_);
    }
    return Mutator::visit(op);
}

Stmt parallelize(const Stmt &_ast, const ID &loop,
                 const std::string &parallel) {
    Parallelize mutator(loop, parallel);
    auto ast = makeReduction(_ast);
    auto oldAst = ast;
    ast = mutator(ast);
    if (!mutator.done()) {
        throw InvalidSchedule("Loop " + toString(loop) + " not found");
    }

    {
        FindDepsCond findDepsCond{{loop, DepDirection::Normal}};
        for (auto &&outerLoop : mutator.outerLoops()) {
            findDepsCond.push_back({outerLoop, DepDirection::Same});
        }
        auto filter = [&](const AccessPoint &later,
                          const AccessPoint &earlier) {
            return earlier.cursor_.getParentById(loop).isValid() &&
                   later.cursor_.getParentById(loop).isValid();
        };
        auto found = [&](const Dependency &d) {
            throw InvalidSchedule(toString(d) + " cannot be resolved");
        };
        findDeps(oldAst, {findDepsCond}, found, FindDepsMode::Dep, DEP_ALL,
                 filter);
    }

    {
        auto filter = [&](const AccessPoint &later,
                          const AccessPoint &earlier) {
            bool earlierInLoop = earlier.cursor_.getParentById(loop).isValid();
            bool laterInLoop = later.cursor_.getParentById(loop).isValid();
            if ((earlierInLoop && !laterInLoop) ||
                (!earlierInLoop && laterInLoop)) {
                if (parallel.substr(0, 10) == "threadIdx.") {
                    return later.def_->buffer_->mtype() == MemType::GPULocal;
                }
                if (parallel.substr(0, 9) == "blockIdx.") {
                    return later.def_->buffer_->mtype() == MemType::GPULocal ||
                           later.def_->buffer_->mtype() == MemType::GPUShared;
                }
            }
            return false;
        };
        auto found = [&](const Dependency &d) {
            ASSERT(d.cond_.size() == 1);
            throw InvalidSchedule(toString(d) + " cannot be resolved");
        };
        findDeps(ast,
                 {{{NodeIDOrParallelScope(parallel, false),
                    DepDirection::Different}}},
                 found, FindDepsMode::Dep, DEP_ALL, filter);
    }
    return ast;
}

} // namespace ir
