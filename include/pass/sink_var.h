#ifndef SINK_VAR_H
#define SINK_VAR_H

#include <unordered_set>
#include <vector>

#include <analyze/find_loop_variance.h>
#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

/**
 * Make the scope of a local variable smaller
 *
 * If you don't want a variable to be sinked, please set VarDefNode::pinned_
 */
class SinkVar : public Mutator {
    const std::unordered_set<std::pair<std::string, ID>>
        &deps_; // {(var, loop)}
    const LoopVariUniqVarMap &variantMap_;
    bool isFixPoint_ = true;

  public:
    SinkVar(const std::unordered_set<std::pair<std::string, ID>> &deps,
            const LoopVariUniqVarMap &variantMap)
        : deps_(deps), variantMap_(variantMap) {}

    bool isFixPoint() const { return isFixPoint_; }

  protected:
    Stmt visit(const VarDef &op) override;
};

Stmt sinkVar(const Stmt &op);

DEFINE_PASS_FOR_FUNC(sinkVar)

} // namespace ir

#endif // SINK_VAR_H
