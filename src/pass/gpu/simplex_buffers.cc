#ifdef FT_WITH_CUDA

#include <analyze/all_uses.h>
#include <container_utils.h>
#include <pass/gpu/simplex_buffers.h>
#include <pass/replace_iter.h>
#include <pass/shrink_var.h>

namespace freetensor {

namespace gpu {

Ref<SimplexOffset> FindSimplexOffset::getSimplexOffset(
    const std::unordered_set<ParallelScope> &filter, const Expr &expr) {
    Ref<SimplexOffset> ret = Ref<SimplexOffset>::make();
    analyzeLinear_(expr);
    for (auto &&[k, a] : analyzeLinear_.result().at(expr).coeff_) {
        if (allUses(a).empty()) {
            std::unordered_map<std::string, Expr> replace;
            for (auto &&iter : allIters(a)) {
                auto &&para = loop(iter)->property_->parallel_;
                if (!filter.count(para)) {
                    goto fail;
                }
                replace[iter] = makeVar(".simplex_buffers." + toString(para));
            }
            auto expr = makeMul(makeIntConst(k), ReplaceIter{replace}(a));
            ASSERT(!ret->offset_.count(expr));
            ret->offset_.emplace(expr);
        }
    fail:;
    }
    return ret;
}

Stmt ApplySimplexOffset::visit(const For &op) {
    if (op->property_->parallel_ == serialScope) {
        return BaseClass::visit(op);
    } else {
        auto tmpName = ".simplex_buffers." + toString(op->property_->parallel_);
        if (!para2var_.count(tmpName)) {
            para2var_[tmpName] = makeVar(op->iter_);
            auto ret = BaseClass::visit(op);
            para2var_.erase(tmpName);
            return ret;
        } else {
            auto oldVar = para2var_[tmpName];
            para2var_[tmpName] = makeVar(op->iter_);
            auto ret = BaseClass::visit(op);
            para2var_[tmpName] = oldVar;
            return ret;
        }
    }
}

Stmt simplexBuffers(const Stmt &_op, const ID &defId) {
    FindSimplexOffset visitor(defId);
    visitor(_op);
    ApplySimplexOffset mutator(visitor.offsets());
    auto op = mutator(_op);
    return shrinkVar(op);
}

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA
