#ifndef FREE_TENSOR_SINK_VAR_H
#define FREE_TENSOR_SINK_VAR_H

#include <unordered_set>
#include <vector>

#include <analyze/find_loop_variance.h>
#include <func.h>
#include <lazy.h>
#include <mutator.h>
#include <visitor.h>

namespace freetensor {

/**
 * Make the scope of a local variable smaller
 *
 * If you don't want a variable to be sinked, please set VarDefNode::pinned_
 */
class SinkVar : public Mutator {
    const std::unordered_set<std::pair<ID, ID>> &deps_; // {(vardef, loop)}
    const std::unordered_set<ID> &analyzedDeps_;        // {vardef}
    std::unordered_set<ID> &needDepAnalysis_;           // {vardef}
    Lazy<LoopVariUniqVarMap> variantMap_;
    bool isFixPoint_ = true;

  private:
    bool hasDep(const ID &vardef, const ID &loop);

  public:
    SinkVar(const std::unordered_set<std::pair<ID, ID>> &deps,
            const std::unordered_set<ID> &analyzedDeps,
            std::unordered_set<ID> &needDepAnalysis,
            const Lazy<LoopVariUniqVarMap> &variantMap)
        : deps_(deps), analyzedDeps_(analyzedDeps),
          needDepAnalysis_(needDepAnalysis), variantMap_(variantMap) {}

    bool isFixPoint() const { return isFixPoint_; }

  protected:
    Stmt visit(const VarDef &op) override;
};

Stmt sinkVar(const Stmt &op);

DEFINE_PASS_FOR_FUNC(sinkVar)

} // namespace freetensor

#endif // FREE_TENSOR_SINK_VAR_H
