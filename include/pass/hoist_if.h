#ifndef HOIST_IF_H
#define HOIST_IF_H

#include <unordered_set>

#include <mutator.h>

namespace ir {

class HoistIf : public Mutator {
    std::unordered_set<std::string> def_;
    std::vector<std::vector<If>> ifStack_;

  protected:
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const StmtSeq &op) override;
};

/**
 * Hoist a If node to outside a For node if the loop iterator is not in the
 * condition
 *
 * We do not handle else-cases in this pass. Otherwise, the resulting code will
 * be too long. This is different from pass/seperate_tail
 */
Stmt hoistIf(const Stmt &op);

} // namespace ir

#endif // HOIST_IF_H
