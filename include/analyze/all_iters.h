#ifndef ALL_ITERS_H
#define ALL_ITERS_H

#include <unordered_set>

#include <visitor.h>

namespace ir {

/**
 * Record all buffers that are read in an AST
 */
class AllIters : public Visitor {
    std::unordered_set<std::string> iters_;

  public:
    const std::unordered_set<std::string> &iters() const { return iters_; }

  protected:
    void visit(const Var &op) override;
};

std::unordered_set<std::string> allIters(const AST &op);

} // namespace ir

#endif // ALL_ITERS_H
