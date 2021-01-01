#ifndef SINK_VAR_H
#define SINK_VAR_H

#include <unordered_set>

#include <mutator.h>

namespace ir {

/**
 * Make the scope of a local variable smaller
 */
class SinkVar : public Mutator {
    std::unordered_set<std::string> used_;
    bool isFixPoint_ = true;

  public:
    bool isFixPoint() const { return isFixPoint_; }

  protected:
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const AddTo &op) override;
    Stmt visit(const VarDef &op) override;
};

Stmt sinkVar(const Stmt &op);

} // namespace ir

#endif // SINK_VAR_H
