#ifndef SWAP_H
#define SWAP_H

#include <string>
#include <vector>

#include <mutator.h>

namespace ir {

class Swap : public Mutator {
    std::vector<std::string> order_;
    StmtSeq scope_;

  public:
    Swap(const std::vector<std::string> &order) : order_(order) {}

    const StmtSeq &scope() const { return scope_; }

  protected:
    Stmt visit(const StmtSeq &op) override;
};

} // namespace ir

#endif // SWAP_H
