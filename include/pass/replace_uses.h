#ifndef REPLACE_USES_H
#define REPLACE_USES_H

#include <unordered_map>

#include <mutator.h>

namespace ir {

class ReplaceUses : public Mutator {
    const std::unordered_map<Load, Expr> &replaceLoad_;

  public:
    ReplaceUses(const std::unordered_map<Load, Expr> &replaceLoad)
        : replaceLoad_(replaceLoad) {}

  protected:
    Expr visit(const Load &op) override;
};

} // namespace ir

#endif // REPLACE_USES_H
