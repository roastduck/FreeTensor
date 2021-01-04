#ifndef SHRINK_VAR_H
#define SHRINK_VAR_H

#include <unordered_map>

#include <mutator.h>

namespace ir {

class ShrinkVar : public Mutator {
    std::unordered_map<std::string, std::vector<Expr>> offset_;

  private:
    template <class T> T modifyAccess(const T &op) {
        if (offset_.count(op->var_)) {
            auto &&offset = offset_.at(op->var_);
            ASSERT(offset.size() == op->indices_.size());
            for (size_t i = 0, iEnd = offset.size(); i < iEnd; i++) {
                op->indices_[i] = makeSub(op->indices_[i], offset[i]);
            }
        }
        return op;
    }

  protected:
    Stmt visit(const VarDef &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const AddTo &op) override;
};

Stmt shrinkVar(const Stmt &op);

} // namespace ir

#endif // SHRINK_VAR_H
