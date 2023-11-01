#include <analyze/deps.h>
#include <analyze/find_loop_variance.h>
#include <analyze/find_stmt.h>
#include <analyze/symbol_table.h>
#include <analyze/track_stmt.h>
#include <lazy.h>
#include <schedule.h>
#include <schedule/check_not_in_lib.h>
#include <schedule/parallelize.h>

namespace freetensor {

namespace {

class CheckNestedCudaScope : public SymbolTable<TrackStmt<Visitor>> {
    typedef SymbolTable<TrackStmt<Visitor>> BaseClass;

    const LoopVariExprMap &variantExprs_;
    const LoopVariUniqVarMap &variantVars_;
    ID loop1_, loop2_;
    bool inLoop1_ = false, inLoop2_ = false;

  public:
    CheckNestedCudaScope(const LoopVariExprMap &variantExprs,
                         const LoopVariUniqVarMap &variantVars, const ID &loop1,
                         const ID &loop2)
        : variantExprs_(variantExprs), variantVars_(variantVars), loop1_(loop1),
          loop2_(loop2) {}

  protected:
    using BaseClass::visit;

    void visitExpr(const Expr &expr) {
        // No need to recurse
        if (inLoop1_ && inLoop2_ &&
            isVariant(variantExprs_, StmtOrExprID{expr, curStmt()}, loop1_) &&
            isVariant(variantExprs_, StmtOrExprID{expr, curStmt()}, loop2_)) {
            throw InvalidSchedule(
                "Nested loop bound to the same CUDA scope is not allowed, "
                "except when they are invariant");
        }
    }

    void visit(const Store &op) {
        // No need to recurse
        if (inLoop1_ && inLoop2_ &&
            isVariant(variantVars_, def(op->var_)->id(), loop1_) &&
            isVariant(variantVars_, def(op->var_)->id(), loop2_)) {
            throw InvalidSchedule(
                "Nested loop bound to the same CUDA scope is not allowed, "
                "except when they are invariant");
        }
    }

    void visit(const ReduceTo &op) {
        // No need to recurse
        if (inLoop1_ && inLoop2_ &&
            isVariant(variantVars_, def(op->var_)->id(), loop1_) &&
            isVariant(variantVars_, def(op->var_)->id(), loop2_)) {
            throw InvalidSchedule(
                "Nested loop bound to the same CUDA scope is not allowed, "
                "except when they are invariant");
        }
    }

    void visit(const For &op) {
        bool oldInLoop1 = inLoop1_, oldInLoop2 = inLoop2_;
        if (op->id() == loop1_) {
            inLoop1_ = true;
        }
        if (op->id() == loop2_) {
            inLoop2_ = true;
        }
        BaseClass::visit(op);
        inLoop1_ = oldInLoop1;
        inLoop2_ = oldInLoop2;
    }
};

} // Anonymous namespace

Stmt Parallelize::visit(const For &_op) {
    loopStack_.emplace_back(_op->id());
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    loopStack_.pop_back();

    if (op->id() == loop_) {
        op->property_->parallel_ = parallel_;
        outerLoops_ = loopStack_;
        done_ = true;
    }
    return op;
}

Stmt parallelize(const Stmt &_ast, const ID &loop,
                 const ParallelScope &parallel, bool allowReduction) {
    if (getenv("PAPER_IS_BWD") && getenv("PAPER_NO_PAR_REDUCE")) {
        allowReduction = false;
    }
    checkNotInLib(_ast, loop);

    Parallelize mutator(loop, parallel);
    auto ast = _ast;
    auto oldAst = ast;
    ast = mutator(ast);
    if (!mutator.done()) {
        throw InvalidSchedule("Loop " + toString(loop) + " not found");
    }

    // Make sure there is no illegal cross-thread dependence
    FindDepsDir findDepsDir{{loop, DepDirection::Normal}};
    for (auto &&outerLoop : mutator.outerLoops()) {
        findDepsDir.push_back({outerLoop, DepDirection::Same});
    }
    FindDeps()
        .direction({findDepsDir})
        .filterSubAST(loop)
        .ignoreReductionWAW(allowReduction)(oldAst, [&](const Dependence &d) {
            throw InvalidSchedule(toString(d) + " cannot be resolved");
        });

    // Make sure no thread-local variable is used by another thread
    FindDeps()
        .direction(
            {{{NodeIDOrParallelScope(parallel), DepDirection::Different}}})
        .filter([&](const AccessPoint &later, const AccessPoint &earlier) {
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
        })
        .ignoreReductionWAW(allowReduction)(ast, [&](const Dependence &d) {
            ASSERT(d.dir_.size() == 1);
            throw InvalidSchedule(toString(d) + " cannot be resolved");
        });

    // Check illegal cases even in our extended fork-join model. See the
    // doc-string of schedule/parallelize
    if (std::holds_alternative<CUDAScope>(parallel)) {
        auto variants = LAZY(findLoopVariance(ast));
        for (auto &&other :
             findAllStmt(ast, "(<For><<-" + toString(loop) + ")|(<For>->>" +
                                  toString(loop) + ")")) {
            if (other.as<ForNode>()->property_->parallel_ == parallel) {
                CheckNestedCudaScope{variants->first, variants->second, loop,
                                     other->id()}(ast);
            }
        }
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
