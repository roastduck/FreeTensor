#ifndef ALL_WRITES_H
#define ALL_WRITES_H

#include <unordered_set>

#include <visitor.h>

namespace ir {

/**
 * Record all buffers that are written in an AST
 */
class AllWrites : public Visitor {
    std::unordered_set<std::string> writes_;

  public:
    const std::unordered_set<std::string> &writes() const { return writes_; }

  protected:
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
};

std::unordered_set<std::string> allWrites(const AST &op);

} // namespace ir

#endif // ALL_WRITES_H
