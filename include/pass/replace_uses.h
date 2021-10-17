#ifndef REPLACE_USES_H
#define REPLACE_USES_H

#include <unordered_map>

#include <mutator.h>

namespace ir {

class ReplaceUses : public Mutator {
    const std::unordered_map<Load, Expr> &replaceLoad_;
    const std::unordered_map<ReduceTo, Expr> &replaceReduceTo_;

  public:
    ReplaceUses(const std::unordered_map<Load, Expr> &replaceLoad,
                const std::unordered_map<ReduceTo, Expr> &replaceReduceTo)
        : replaceLoad_(replaceLoad), replaceReduceTo_(replaceReduceTo) {}

  protected:
    Expr visit(const Load &op) override;
    Stmt visit(const ReduceTo &op) override;
};

} // namespace ir

#endif // REPLACE_USES_H
