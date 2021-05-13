#ifndef FIND_ALL_LOOPS_H
#define FIND_ALL_LOOPS_H

#include <visitor.h>

namespace ir {

class FindAllLoops : public Visitor {
    std::vector<std::string> loops_;

  public:
    const std::vector<std::string> &loops() const { return loops_; }

  protected:
    void visit(const For &op) override;
};

} // namespace ir

#endif // FIND_ALL_LOOPS_H
