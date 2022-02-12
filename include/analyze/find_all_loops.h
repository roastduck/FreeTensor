#ifndef FIND_ALL_LOOPS_H
#define FIND_ALL_LOOPS_H

#include <visitor.h>

namespace ir {

class FindAllLoops : public Visitor {
    std::vector<ID> loops_;

  public:
    const std::vector<ID> &loops() const { return loops_; }

  protected:
    void visit(const For &op) override;
};

/**
 * Collect IDs of all For nodes in the AST
 */
std::vector<ID> findAllLoops(const Stmt &op);

} // namespace ir

#endif // FIND_ALL_LOOPS_H
